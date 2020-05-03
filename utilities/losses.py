import tensorflow as tf
import numpy as np

# MSE loss
def MSE(labels, predictions, weights, *_):
    loss_function = tf.compat.v1.losses.mean_squared_error(labels, predictions, weights)

    return loss_function

# MAE loss
def MAE(labels, predictions, weights, *_):
    loss_function = tf.compat.v1.losses.absolute_difference(labels, predictions, weights)

    return loss_function

# auxiliary loss
# https://www-sciencedirect-com.ucsf.idm.oclc.org/science/article/pii/S1361841518301257
def auxiliary_MAE(labels, predictions, weights, data_format):
    # predefine loss_function
    loss_function = None
    # determine dimension of channels
    dim = 1 if data_format == 'channels_first' else -1
    # loop through the different predictions and sum to create auxillary loss
    for i in range(predictions.shape[dim]):
        # isolate pred
        if data_format == 'channels_first':
            pred = tf.expand_dims(predictions[:, i, :, :], axis=1)
        else:
            pred = tf.expand_dims(predictions[:, :, :, i], axis=-1)
        # generate MAE loss
        loss = tf.compat.v1.losses.absolute_difference(labels, pred, weights)
        if i == 0:  # for first loop, use full value of loss as this is the final predictions
            loss_function = loss
        else:  # for all subsequent loops, add the loss times 0.5, as these are auxiliary losses
            loss_function = tf.add(loss_function, loss * 0.5)

    return loss_function

# 2.5D MSE loss
def MSE25D(labels, predictions, weights, data_format):
    # cast weights to float
    weights = tf.cast(weights, tf.float32)
    # handle channels last
    if data_format == 'channels_last':
        # get center slice for channels last [b, x, y, z, c] and double weights for this slice
        center_pred = predictions[:, :, :, int(round(predictions.shape[3] / 2 + 1)), :]
        center_lab = labels[:, :, :, int(round(labels.shape[3] / 2 + 1)), :]
        center_weights = tf.multiply(weights[:, :, :, int(round(weights.shape[3] / 2 + 1)), :], 2.)
    # handle channels first
    elif data_format == 'channels_first':
        # get center slice for channels last [b, c, x, y, z] and double weights for this slice
        center_pred = predictions[:, :, :, :, int(round(predictions.shape[3] / 2)) + 1]
        center_lab = labels[:, :, :, :, int(round(labels.shape[3] / 2 + 1))]
        center_weights = tf.multiply(weights[:, :, :, :, int(round(weights.shape[3] / 2 + 1))], 2.)
    else:
        raise ValueError("Data format not understood: " + str(data_format))
    # define loss as only the center slice first
    loss_function = tf.compat.v1.losses.mean_squared_error(
        center_pred, center_lab, center_weights, reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    # add loss for the rest of the slab to the loss
    loss_function = tf.add(loss_function, tf.compat.v1.losses.mean_squared_error(
        predictions, labels, weights, reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS))

    return loss_function

# 2.5D MAE loss
def MAE25D(labels, predictions, weights, data_format):
    # handle channels last
    if data_format == 'channels_last':
        # get center slice for channels last [b, x, y, z, c]
        center_pred = predictions[:, :, :, predictions.shape[3] / 2 + 1, :]
        center_lab = labels[:, :, :, labels.shape[3] / 2 + 1, :]
        center_weights = weights[:, :, :, weights.shape[3] / 2 + 1, :]
    # handle channels first
    elif data_format == 'channels_first':
        # get center slice for channels first [b, c, x, y, z]
        center_pred = predictions[:, :, :, :, predictions.shape[3] / 2 + 1]
        center_lab = labels[:, :, :, :, labels.shape[3] / 2 + 1]
        center_weights = weights[:, :, :, :, weights.shape[3] / 2 + 1]
    else:
        raise ValueError("Data format not understood: " + str(data_format))
    # define loss
    loss_function = tf.compat.v1.losses.absolute_difference(center_lab, center_pred, center_weights)
    # add remaining slices at equal weight
    loss_function = tf.add(loss_function, tf.compat.v1.losses.absolute_difference(labels, predictions, weights))

    return loss_function

# softmax cross entropy w logits
def softmaxCE(labels, predictions, *_):
    # define loss function
    loss_function = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=tf.cast(labels, tf.int32),
        logits=predictions,
        weights=1.0,  # not weighting using mask as mask is target
        reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    )

    return loss_function

# generalized DICE loss for 2D and 2.5D networks
def gen_dice(labels, predictions, _, data_format):
    # convert labels to int and softmax pred
    y = tf.cast(labels, tf.float32)
    y_pred = tf.nn.softmax(predictions, axis=-1 if data_format == 'channels_last' else 1)
    # handle 2.5D by making a triangle weight kernel weighting center slice the most and sum == 1
    if len(y_pred.shape) == 5 and data_format == 'channels_last':
        # for channels last [b, x, y, z, c]
        kerlen = y_pred.get_shape().as_list()[3]
        kernel = (float(kerlen) + 1. - np.abs(np.arange(float(kerlen)) - np.arange(float(kerlen))[::-1])) / 2.
        kernel /= kernel.sum()
        kernel = tf.reshape(tf.constant(kernel, dtype=tf.float32), shape=[1, 1, 1, kerlen, 1])
        y = tf.reduce_sum(input_tensor=tf.multiply(y, kernel), axis=3)
        y_pred = tf.reduce_sum(input_tensor=tf.multiply(y_pred, kernel), axis=3)
    # handle channels first
    elif len(y_pred.shape) == 5 and data_format == 'channels_first':
        # for channels first [b, c, x, y, z]
        kerlen = y_pred.get_shape().as_list()[4]
        kernel = (kerlen + 1 - np.abs(np.arange(kerlen) - np.arange(kerlen)[::-1])) / 2
        kernel /= kernel.sum()
        kernel = tf.reshape(tf.constant(kernel, dtype=tf.float32), shape=[1, 1, 1, 1, kerlen])
        y = tf.reduce_sum(input_tensor=tf.multiply(y, kernel), axis=4)
        y_pred = tf.reduce_sum(input_tensor=tf.multiply(y_pred, kernel), axis=4)
    # do dice calculation
    mean = 0.5  # the weight is the proportion of 1s to 0s in the label set
    w_1 = 1. / mean ** 2.
    w_0 = 1. / (1. - mean) ** 2.
    y_true_f_1 = tf.reshape(y, [-1])
    y_pred_f_1 = tf.reshape(y_pred[..., 1] if data_format == 'channels_last' else y_pred[:, 1, ...], [-1])
    y_true_f_0 = tf.reshape(1 - y, [-1])
    y_pred_f_0 = tf.reshape(y_pred[..., 0] if data_format == 'channels_last' else y_pred[:, 0, ...], [-1])
    int_0 = tf.reduce_sum(input_tensor=y_true_f_0 * y_pred_f_0)
    int_1 = tf.reduce_sum(input_tensor=y_true_f_1 * y_pred_f_1)
    dice = 2. * (w_0 * int_0 + w_1 * int_1) / (
                (w_0 * (tf.reduce_sum(input_tensor=y_true_f_0) + tf.reduce_sum(input_tensor=y_pred_f_0))) +
                (w_1 * (tf.reduce_sum(input_tensor=y_true_f_1) + tf.reduce_sum(input_tensor=y_pred_f_1))))
    loss_function = 1. - dice

    return loss_function


def loss_picker(loss_method, labels, predictions, data_format, weights=None):
    """
    Takes a string specifying the loss method and returns a tensorflow loss function
    :param loss_method: (str) the desired loss method
    :param labels: (tf.tensor) the labels tensor
    :param predictions: (tf.tensor) the features tensor
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param weights: (tf.tensor) an optional weight tensor for masking values
    :return: A tensorflow loss function
    """

    # sanity checks
    if not isinstance(loss_method, str):
        raise ValueError("Loss method parameter must be a string")
    if weights is None:
        weights = 1.0
    loss_methods = ['MSE', 'MAE', 'auxiliary_MAE', 'MSE25D', 'MAE25D', 'softmaxCE', 'weighted_dice', 'gen_dice']

    # check for specified loss method and error if not found
    if loss_method in globals():
        loss_fn = globals()[loss_method](labels, predictions, weights, data_format)
    else:
        raise NotImplementedError("Specified loss method is not implemented: " + loss_method +
                                  ". Possible options are: " + ' '.join(loss_methods))

    return loss_fn
