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
    return tf.losses.compute_weighted_loss(loss, weights=tf.expand_dims(mask, axis=2))

def l1_loss(y_hat, y, mask):
    loss = tf.abs(y_hat - y)
    return tf.losses.compute_weighted_loss(loss, weights=tf.expand_dims(mask, axis=2))


def mse_loss(y_hat, y, mask):
    loss = tf.losses.mean_squared_error(y, y_hat, weights=tf.expand_dims(mask, axis=2))
    # tf.losses.mean_squared_error cast output to float32 so the output is casted back to the original precision
    if loss.dtype is not y.dtype:
        return tf.cast(loss, dtype=y.dtype)
    else:
        return loss


def codes_loss(y_hat, y, mask, codes_loss_type):
    if codes_loss_type == "l1":
        y_hat = tf.squeeze(y_hat)
        
        return l1_loss(y_hat, y, mask)
    elif codes_loss_type == "mse":
        return mse_loss(y_hat, y, mask)
    else:
        raise ValueError(f"Unknown loss type: {codes_loss_type}")


def classification_loss(y_hat, y, mask):
    return tf.losses.softmax_cross_entropy(y, y_hat, weights=mask)


def binary_loss(done_hat, done, mask):
#    done = tf.Print(done, [tf.shape(done)], "done")
#    done_hat = tf.Print(done_hat, [tf.shape(done_hat)], "done_hat")
#    mask = tf.Print(mask, [tf.shape(mask)], "mask")   
    return tf.losses.sigmoid_cross_entropy(done, tf.squeeze(done_hat, axis=-1), weights=mask)
