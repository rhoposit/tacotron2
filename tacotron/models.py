# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Models. """

import tensorflow as tf
from tacotron.modules import Embedding
from tacotron.tacotron_v1 import EncoderV1, DecoderV1
from tacotron.hooks import MetricsSaver, PostNetMetricsSaver
from util.audio import Audio


class SingleSpeakerTacotronV1Model(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            embedding = Embedding(params.num_symbols, embedding_dim=params.embedding_dim)

            encoder = EncoderV1(is_training,
                                cbhg_out_units=params.cbhg_out_units,
                                conv_channels=params.conv_channels,
                                max_filter_width=params.max_filter_width,
                                projection1_out_channels=params.projection1_out_channels,
                                projection2_out_channels=params.projection2_out_channels,
                                num_highway=params.num_highway,
                                prenet_out_units=params.encoder_prenet_out_units,
                                drop_rate=params.encoder_prenet_drop_rate)

            decoder = DecoderV1(prenet_out_units=params.decoder_prenet_out_units,
                                drop_rate=params.decoder_prenet_drop_rate,
                                attention_out_units=params.attention_out_units,
                                decoder_out_units=params.decoder_out_units,
                                num_codes=params.num_codes,
                                outputs_per_step=params.outputs_per_step,
                                max_iters=params.max_iters,
                                n_feed_frame=params.n_feed_frame)

            target = labels.codes if (is_training or is_validation) else None

            embedding_output = embedding(features.source)
            encoder_output = encoder(embedding_output)
            codes_output, stop_token, decoder_state = decoder(encoder_output,
                                                            is_training=is_training,
                                                            is_validation=is_validation,
                                                            memory_sequence_length=features.source_length,target=target)
            alignment = tf.transpose(decoder_state[0].alignment_history.stack(), [1, 2, 0])

            global_step = tf.train.get_global_step()

            if mode is not tf.estimator.ModeKeys.PREDICT:
                codes_loss = self.codes_loss(code_output, labels.codes,
                                          labels.codes_loss_mask)
                done_loss = self.binary_loss(stop_token, labels.done, labels.binary_loss_mask)
                loss = code_loss + done_loss

            if is_training:
                lr = self.learning_rate_decay(
                    params.initial_learning_rate, global_step) if params.decay_learning_rate else tf.convert_to_tensor(
                    params.initial_learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.add_training_stats(loss, codes_loss, done_loss, lr)
                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                    summary_writer = tf.summary.FileWriter(model_dir)
                    alignment_saver = MetricsSaver([alignment],
                                                   global_step,
                                                   codes_output,
                                                   labels.codes,
                                                   labels.target_length,
                                                   features.id,
                                                   features.text,
                                                   params.alignment_save_steps,
                                                   mode, summary_writer,
                                                   params.save_training_time_metrics,
                                                   params.keep_eval_results_max_epoch)
                    hooks = [alignment_saver]
                    if params.record_profile:
                        profileHook = tf.train.ProfilerHook(save_steps=params.profile_steps, output_dir=model_dir,
                                                            show_dataflow=True, show_memory=True)
                        hooks.append(profileHook)
                    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                      training_hooks=hooks)

            if is_validation:
                # validation with teacher forcing
                codes_output_with_teacher, stop_token_with_teacher, _ = decoder(encoder_output,
                                                                              is_training=is_training,
                                                                              is_validation=is_validation,
                                                                              memory_sequence_length=features.source_length,
                                                                              target=target,
                                                                              teacher_forcing=True)
                codes_loss_with_teacher = self.spec_loss(codes_output_with_teacher, labels.codes, labels.codes_loss_mask)
                done_loss_with_teacher = self.binary_loss(stop_token_with_teacher, labels.done, labels.binary_loss_mask)
                loss_with_teacher = codes_loss_with_teacher + done_loss_with_teacher
                eval_metric_ops = self.get_validation_metrics(codes_loss, done_loss, loss_with_teacher, codes_loss_with_teacher, done_loss_with_teacher)

                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = MetricsSaver([alignment],
                                               global_step,
                                               codes_output,
                                               labels.codes,
                                               labels.target_length,
                                               features.id,
                                               features.text,
                                               1,
                                               mode, summary_writer,
                                               params.save_training_time_metrics,
                                               params.keep_eval_results_max_epoch)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  evaluation_hooks=[alignment_saver],
                                                  eval_metric_ops=eval_metric_ops)

            if is_prediction:
                return tf.estimator.EstimatorSpec(mode, predictions={
                    "id": features.id,
                    "codes": codes_output,
                    "alignment": alignment,
                    "source": features.source,
                    "text": features.text,
                })

        super(SingleSpeakerTacotronV1Model, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def codes_loss(y_hat, y, mask, n_priority_freq=None, priority_w=0):
        l1_loss = tf.abs(y_hat - y)

        # Priority L1 loss
        if n_priority_freq is not None and priority_w > 0:
            priority_loss = tf.abs(y_hat[:, :, :n_priority_freq] - y[:, :, :n_priority_freq])
            l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

        return tf.losses.compute_weighted_loss(l1_loss, weights=tf.expand_dims(mask, axis=2))

    @staticmethod
    def binary_loss(done_hat, done, mask):
        return tf.losses.sigmoid_cross_entropy(done, tf.squeeze(done_hat, axis=-1), weights=mask)

    @staticmethod
    def learning_rate_decay(init_rate, global_step):
        warmup_steps = 4000.0
        step = tf.to_float(global_step + 1)
        return init_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    @staticmethod
    def add_training_stats(loss, codes_loss, done_loss, learning_rate):
        if loss is not None:
            tf.summary.scalar("loss_with_teacher", loss)
        if codes_loss is not None:
            tf.summary.scalar("codes_loss", codes_loss)
            tf.summary.scalar("codes_loss_with_teacher", codes_loss)
        if done_loss is not None:
            tf.summary.scalar("done_loss", done_loss)
            tf.summary.scalar("done_loss_with_teacher", done_loss)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(codes_loss, done_loss, loss_with_teacher, codes_loss_with_teacher, done_loss_with_teacher):
        metrics = {}
        if codes_loss is not None:
            metrics["codes_loss"] = tf.metrics.mean(codes_loss)
        if done_loss is not None:
            metrics["done_loss"] = tf.metrics.mean(done_loss)
        if loss_with_teacher is not None:
            metrics["loss_with_teacher"] = tf.metrics.mean(loss_with_teacher)
        if codes_loss_with_teacher is not None:
            metrics["codes_loss_with_teacher"] = tf.metrics.mean(codes_loss_with_teacher)
        if done_loss_with_teacher is not None:
            metrics["done_loss_with_teacher"] = tf.metrics.mean(done_loss_with_teacher)
        return metrics

