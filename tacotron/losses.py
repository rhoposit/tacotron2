# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
import sys

def l1_loss(y_hat, y, mask):
    loss = tf.abs(y_hat - y)
#    loss = tf.Print(loss, [tf.shape(loss)], "l1 loss: loss\n")
#    return tf.losses.compute_weighted_loss(loss, weights=tf.expand_dims(mask, axis=2))
    return tf.losses.compute_weighted_loss(loss, weights=mask)


def mse_loss(y_hat, y, mask):
    loss = tf.losses.mean_squared_error(y, y_hat, weights=tf.expand_dims(mask, axis=2))
    # tf.losses.mean_squared_error cast output to float32 so the output is casted back to the original precision
    if loss.dtype is not y.dtype:
        return tf.cast(loss, dtype=y.dtype)
    else:
        return loss


def codes_loss(y_hat, y, mask, codes_loss_type):
    if codes_loss_type == "l1":
#        y_hat = tf.Print(y_hat, [tf.shape(y_hat)], "\ncodes loss: yhat")
        y_hat = tf.squeeze(y_hat)
#        y_hat = tf.Print(y_hat, [tf.shape(y_hat)], "\ncodes loss: yhat")        
#        y = tf.Print(y, [tf.shape(y)], "\ncodes loss: y")
#        mask = tf.Print(mask, [tf.shape(mask)], "\ncodes loss: mask")
        
        return l1_loss(y_hat, y, mask)
    elif codes_loss_type == "mse":
        return mse_loss(y_hat, y, mask)
    else:
        raise ValueError(f"Unknown loss type: {codes_loss_type}")


def classification_loss(y_hat, y, mask):
    return tf.losses.softmax_cross_entropy(y, y_hat, weights=mask)


def binary_loss(done_hat, done, mask):
#    done_hat = tf.Print(done_hat, [tf.shape(done_hat)], "\nbinary loss: done_hat")
#    done = tf.Print(done, [tf.shape(done)], "\nbinary loss: done")
#    mask = tf.Print(mask, [tf.shape(mask)], "\nbinary loss: mask")
    a = tf.squeeze(done_hat, axis=-1)
#    a = tf.Print(a, [tf.shape(a)], "\nbinary loss: a")
    return tf.losses.sigmoid_cross_entropy(done, a, weights=mask)
