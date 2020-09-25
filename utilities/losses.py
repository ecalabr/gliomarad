import tensorflow as tf
import numpy as np


# get built in locals
start_globals = list(globals().keys())


# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/


# MSE loss
def MSE(y_true, y_pred, sample_weight=None):
    return tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred,
                                                                                      sample_weight=sample_weight)


# Pixelwise weigthed MSE loss
def wMSE(y_true_weights, y_pred):
    # if weight tensor is included as last dimension of y_true then extract
    if y_true_weights.shape[-1] == 2:  # consider editing in case labels is multi channel
        y_true = y_true_weights[..., 0, None]
        weights = y_true_weights[..., -1, None]
    # if weight tensor is not included then just calculate without weights (this happens during eval even when weigthed)
    else:
        y_true = y_true_weights
        weights = None
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred, sample_weight=weights)


# 2.5 dimensional MSE loss
def MSE25d(y_true, y_pred):
    # weight loss such that middle slice is is strongest weighted [b, x, y, z, c]
    z_dim = y_pred.get_shape().as_list()[3]
    w_vect = np.ones(z_dim)
    w_vect[int(round(z_dim/2.))] = z_dim - 1  # center slice is weighted as much as rest of slices combined
    weights = tf.ones_like(y_true, dtype=tf.float32) * tf.constant(np.expand_dims(w_vect, axis=(0, 1, 2, 4)),
                                                                   dtype=tf.float32)
    return tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred,
                                                                                      sample_weight=weights)


# Pixelwise weigthed MSE loss for nonzero pixels only
def wMSE_nonzero(y_true_weights, y_pred):
    # if extra weight tensor is included as last dimension of y_true then extract
    if y_true_weights.shape[-1] == 2:  # consider editing in case labels is multi channel
        y_true = y_true_weights[..., 0, None]
        # pixel weights per provided loss tensor
        y_true_weights = y_true_weights[..., -1, None]
    # if weight tensor is not included then just calculate without weights (this happens during eval even when weigthed)
    else:
        y_true = y_true_weights
        # weights are 1 where y_true is nonzero
        y_true_weights = tf.cast(tf.cast(y_true, tf.bool), tf.float32)
    # count nonzeros in y_true per sample in batch
    non_batch_axes = list(range(1, 5))  # consider editing for vairable ranks, tf.rank(y_true).numpy())) doesnt work
    nonzeros_per_sample = tf.math.count_nonzero(y_true, axis=non_batch_axes, dtype=tf.float32)
    # calculate per pixel loss with provided weights
    loss_tensor = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)\
        (y_true, y_pred, sample_weight=y_true_weights)
    # mean of nonzero entries along non batch axes, then sum results along batch axis
    # note that loss function automatically reduces the final (channel) dimension
    loss_value = tf.reduce_sum(tf.reduce_sum(loss_tensor, axis=non_batch_axes[:-1]) / nonzeros_per_sample)
    return loss_value


# MAE loss
def MAE(y_true, y_pred, sample_weight=None):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred, sample_weight=sample_weight)


# MAPE loss
def MAPE(y_true, y_pred, sample_weight=None):
    return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred, sample_weight=sample_weight)


# softmax cross entropy
def softmaxCE(y_true, y_pred):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred)


# binary cross entropy
def binaryCE(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)


# generalized DICE loss for 2D and 3D networks
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / denominator


# Tversky index (TI) is a generalization of Dice’s coefficient
def tversky_loss(y_true, y_pred):
        beta = 0.5
        numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
        return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)


# combo dice and binary cross entropy
def combo_loss3d(y_true, y_pred):
    def dice_l(y_t, y_p):
        numerator = 2 * tf.reduce_sum(y_t * y_p, axis=(1,2,3,4))
        denominator = tf.reduce_sum(y_t + y_p, axis=(1,2,3,4))
        return tf.reshape(1 - numerator / denominator, (-1, 1, 1, 1))
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_l(y_true, y_pred)


def loss_picker(params):

    # sanity checks
    if not isinstance(params.loss, str):
        raise ValueError("Loss method parameter must be a string")

    # check for specified loss method and error if not found
    if params.loss in globals():
        loss_fn = globals()[params.loss]
    else:
        methods = [k for k in globals().keys() if k not in start_globals]
        raise NotImplementedError(
            "Specified loss method: '{}' is not one of the available methods: {}".format(params.loss, methods))

    return loss_fn
