# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Tacotron modules. """

import tensorflow as tf
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention
from tacotron.modules import PreNet, CBHG
from tacotron.rnn_wrappers import OutputAndStopTokenWrapper, AttentionRNN, OutputProjectionWrapper
from tacotron.helpers import StopTokenBasedInferenceHelper, TrainingHelper, ValidationHelper
from tacotron.rnn_impl import GRUImpl, gru_cell_factory
from functools import reduce
from typing import Tuple


class EncoderV1(tf.layers.Layer):

    def __init__(self, is_training,
                 cbhg_out_units=256, conv_channels=128, max_filter_width=16,
                 projection1_out_channels=128,
                 projection2_out_channels=128,
                 num_highway=4,
                 prenet_out_units=(256, 128), drop_rate=0.5,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(EncoderV1, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.prenet_out_units = prenet_out_units

        self.prenets = [PreNet(out_unit, is_training, drop_rate, dtype=dtype) for out_unit in prenet_out_units]

        self.cbhg = CBHG(cbhg_out_units,
                         conv_channels,
                         max_filter_width,
                         projection1_out_channels,
                         projection2_out_channels,
                         num_highway,
                         is_training,
                         dtype=dtype)

    def build(self, input_shape):
        embed_dim = input_shape[2].value
        with tf.control_dependencies([tf.assert_equal(self.prenet_out_units[0], embed_dim)]):
            self.built = True

    def call(self, inputs, input_lengths=None, **kwargs):
        prenet_output = reduce(lambda acc, pn: pn(acc), self.prenets, inputs)
        cbhg_output = self.cbhg(prenet_output, input_lengths=input_lengths)
        return cbhg_output

    def compute_output_shape(self, input_shape):
        return self.cbhg.compute_output_shape(input_shape)


# ToDo: remove this function and use attention mechanism factory
def AttentionRNNV1(num_units, prenets: Tuple[PreNet],
                   memory, memory_sequence_length, gru_impl=GRUImpl.GRUCell, dtype=None):
    rnn_cell = gru_cell_factory(gru_impl, num_units)
    attention_mechanism = BahdanauAttention(num_units, memory, memory_sequence_length, dtype=dtype)
    return AttentionRNN(rnn_cell, prenets, attention_mechanism, dtype=dtype)


class DecoderRNNV1(tf.nn.rnn_cell.RNNCell):

    def __init__(self, out_units, attention_cell: AttentionRNN,
                 gru_impl=GRUImpl.GRUCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(DecoderRNNV1, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)

        self._cell = tf.nn.rnn_cell.MultiRNNCell([
            OutputProjectionWrapper(attention_cell, out_units),
            tf.nn.rnn_cell.ResidualWrapper(gru_cell_factory(gru_impl, out_units, dtype=dtype)),
            tf.nn.rnn_cell.ResidualWrapper(gru_cell_factory(gru_impl, out_units, dtype=dtype)),
        ], state_is_tuple=True)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state):
        return self._cell(inputs, state)


class DecoderV1(tf.layers.Layer):

    def __init__(self, prenet_out_units=(256, 128), drop_rate=0.5,
                 attention_out_units=256,
                 decoder_out_units=256,
                 num_mels=80,
                 outputs_per_step=2,
                 max_iters=200,
                 n_feed_frame=1,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(DecoderV1, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._prenet_out_units = prenet_out_units
        self._drop_rate = drop_rate
        self.attention_out_units = attention_out_units
        self.decoder_out_units = decoder_out_units
        self.num_mels = num_mels
        self.outputs_per_step = outputs_per_step
        self.max_iters = max_iters
        self.stop_token_fc = tf.layers.Dense(1, dtype=dtype)
        self.n_feed_frame = n_feed_frame

    def build(self, _):
        self.built = True

    def call(self, source, is_training=None, is_validation=None, teacher_forcing=False, memory_sequence_length=None,
             target=None):
        assert is_training is not None

        prenets = tuple([PreNet(out_unit, is_training, self._drop_rate)
                         for out_unit in self._prenet_out_units])

        batch_size = tf.shape(source)[0]
        attention_cell = AttentionRNNV1(self.attention_out_units, prenets, source, memory_sequence_length,dtype=self.dtype)
        decoder_cell = DecoderRNNV1(self.decoder_out_units, attention_cell, dtype=self.dtype)
        output_and_done_cell = OutputAndStopTokenWrapper(decoder_cell, self.num_mels * self.outputs_per_step,dtype=self.dtype)

        decoder_initial_state = output_and_done_cell.zero_state(batch_size, dtype=source.dtype)

        helper = TrainingHelper(target,
                                self.num_mels,
                                self.outputs_per_step,
                                n_feed_frame=self.n_feed_frame) if is_training \
            else ValidationHelper(target, batch_size,
                                  self.num_mels,
                                  self.outputs_per_step,
                                  n_feed_frame=self.n_feed_frame,
                                  teacher_forcing=teacher_forcing) if is_validation \
            else StopTokenBasedInferenceHelper(batch_size,
                                               self.num_mels,
                                               self.outputs_per_step,
                                               n_feed_frame=self.n_feed_frame,
                                               dtype=source.dtype)

        ((decoder_outputs, stop_token), _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            BasicDecoder(output_and_done_cell, helper, decoder_initial_state), maximum_iterations=self.max_iters)

        code_output = tf.reshape(decoder_outputs, [batch_size, -1, self.num_mels])
        return code_output, stop_token, final_decoder_state



