# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


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


def codes_lossBAD(y_hat, y, mask, codes_loss_type):
    cce = tf.keras.losses.CategoricalCrossentropy()
    y = tf.Print(y, [tf.shape(y)], "\n* y before reshape\n")
    y_hat = tf.Print(y_hat, [tf.shape(y_hat), tf.shape(y_hat)[0], tf.shape(y_hat)[1], tf.shape(y_hat)[2]], "\n* yhat before reshape\n")
    batch = tf.shape(y_hat)[0]
    length = tf.shape(y_hat)[1]
    width = tf.shape(y_hat)[2]
    y =tf.reshape(y, tf.shape(y_hat))
    y = tf.Print(y, [tf.shape(y)], "\n* y after reshape\n")
    return cce(y, y_hat)



def codes_loss(y_hat, y, mask, codes_loss_type):
#    y = tf.Print(y, [tf.shape(y)], "\n* y shape\n")
#    y_hat = tf.Print(y_hat, [tf.shape(y_hat)], "\n* y_hat shape\n")
    return tf.losses.softmax_cross_entropy(y, y_hat, weights=mask)


def classification_loss(y_hat, y, mask):
    return tf.losses.softmax_cross_entropy(y, y_hat, weights=mask)


def binary_loss(done_hat, done, mask):
    done_hat = tf.squeeze(done_hat, axis=-1)
    done = tf.Print(done, [tf.shape(done), done], "done", summarize=-1)
    done_hat = tf.Print(done_hat, [tf.shape(done_hat), done_hat], "done_hat", summarize=-1)
    mask = tf.Print(mask, [tf.shape(mask), mask], "mask")   
    return tf.losses.sigmoid_cross_entropy(done, done_hat, weights=mask)
