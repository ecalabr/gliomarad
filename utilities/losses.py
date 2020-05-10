import tensorflow as tf


# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/


# MSE loss
def MSE(y_true, y_pred, sample_weight=None):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred, sample_weight=sample_weight)


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


# MAE loss
def MAE(y_true, y_pred, sample_weight=None):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred, sample_weight=sample_weight)


# MAPE loss
def MAPE(y_true, y_pred, sample_weight=None):
    return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred, sample_weight=sample_weight)


# softmax cross entropy
def softmaxCE():
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


# binary cross entropy
def binaryCE():
    return tf.keras.losses.BinaryCrossentropy(from_logits=False)


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
        raise NotImplementedError("Specified loss method is not implemented in losses.py: " + params.loss)

    return loss_fn
