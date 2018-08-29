import tensorflow as tf
import numpy as np

################################################################################
# Convenience functions for building the residual networks model.
################################################################################

def batch_norm(tensor, is_training, data_format="channels_last"):
    """
    Performs a batch normalization using a standard set of parameters.
    Args:
        tensor: a tensorflow tensor - the input data tensor
        is_training: a boolean - whether or not the model is training
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tf.layers.batch_normalization layer with appropriate parameters
    """
    # define axis based on data_format
    axis = -1 if data_format == "channels_last" else 1

    return tf.layers.batch_normalization(
        inputs=tensor,
        axis=axis, # channels last, assume last channel is correct axis
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        training=is_training,
        trainable=True,
        name=None,
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=True, # use fused batch norm for performance tensorflow.org/performance/performance_guide#common_fused_ops
        virtual_batch_size=None,
        adjustment=None,
        )


def activation(tensor, acti_type="leaky_relu"): # add optional type argument
    """
    Performs non-linear activation, in this case, leaky ReLU
    Args:
        tensor: a tensorflow tensor - the input data
        acti_type: a string - specifying the type of activation function to use
    Returns:
        the activation function - tf.nn.leaky_relu
    """
    if acti_type == "leaky_relu":
        return tf.nn.leaky_relu(
            features=tensor,
            alpha=0.2,
            name=None)

    elif acti_type == "relu":
        return tf.nn.relu(
            features=tensor,
            name=None)

    else:
        raise ValueError("Provided type " + str(acti_type) + " is not a known activation type")


def fixed_padding(inputs, kernel_size, strides, data_format="channels_last"):
    """
    Pads the input along the spatial dimensions independently of input size. Works for 2D and 3D
    Args:
        inputs: a tensorflow tensor - the input data tensor
        kernel_size: an int or tuple/list - the size of the convolution kernel
        strides: a list or tuple of length 2 or 3 - the strides used for convolution
            determines which dimensions to pad, must have length == input dimensions (-batchsize)
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        A tensor with the same format as the input with the data either intact or padded (if kernel_size > 1)
    """
    # determine signle dimension kernel size (assumes isotropic kernel)
    if isinstance(kernel_size, (list, tuple)): kernel_size = kernel_size[0]

    # determine pad start and end
    pad_total = kernel_size - 1
    p_beg = pad_total // 2
    p_end = pad_total - p_beg

    # set value_error_str
    data_format_error = "'data_format' should be 'channels_first' or 'channels_last', but is " + str(data_format)

    # handle 2D strides first
    if len(strides)==2:
        if strides == [2, 2]:
            if data_format == "channels_first":
                padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [p_beg, p_end], [p_beg, p_end]])
            elif data_format == "channels_last":
                padded_inputs = tf.pad(inputs, [[0, 0], [p_beg, p_end], [p_beg, p_end], [0, 0]])
            else:
                raise ValueError(data_format_error)
        else:
            raise ValueError("2D strides must be [2, 2] but are " + str(strides))

    # handle 3D strides
    elif len(strides)==3:
        # full 3D strides
        if strides == [2, 2, 2]:
            if data_format == "channels_first":
                padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [p_beg, p_end], [p_beg, p_end], [p_beg, p_end]])
            elif data_format == "channels_last":
                padded_inputs = tf.pad(inputs, [[0, 0], [p_beg, p_end], [p_beg, p_end], [p_beg, p_end], [0, 0]])
            else:
                raise ValueError(data_format_error)
        # handle strides in x y (aka h w) only
        elif strides in [[2, 2, 1], [1, 2, 2]]:
            if data_format == "channels_first":
                padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [p_beg, p_end], [p_beg, p_end]])
            elif data_format == "channels_last":
                padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [p_beg, p_end], [p_beg, p_end], [0, 0]])
            else:
                raise ValueError(data_format_error)
        else:
            raise ValueError("3D strides must be [2, 2, 2], [2, 2, 1], or [1, 2, 2] but are " + str(strides))
    else:
        raise ValueError("Strides must be a tuple or list of length 2 or 3")

    return padded_inputs


def conv2d_fixed_pad(inputs, filters, kernel_size, strides, dilation, data_format='channels_last'):
    """
    Strided 2-D convolution with explicit padding.
    Args:
        inputs: a tensorflow tensor - the input data tensor
        filters: an int - the number of filters for the layers
        kernel_size: an int or list/tuple - the size of the convolution kernel
        strides: strides: an int or list/tuple - the stride for the convolutions
        dilation: an int or list/tuple - the dilation rate for the convolution
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tf.layers.conv2d layer with explicit padding
    """
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

    # determine if strided
    if not isinstance(strides, list):
        strides = [strides] * 2
    strided = True if any(stride > 1 for stride in strides) else False

    if strided:
        inputs = fixed_padding(inputs, kernel_size, strides, data_format)

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('valid' if strided else 'same'),
        use_bias=False, # no bias since batch norm is used
        kernel_initializer=tf.variance_scaling_initializer(),
        dilation_rate=dilation,
        data_format=data_format)


def conv3d_fixed_pad(inputs, filters, kernel_size, strides, dilation, data_format='channels_last'):
    """
    Strided 3-D convolution with explicit padding.
    The padding is consistent and is based only on `kernel_size`, not on the dimensions of `inputs`
    (as opposed to using `tf.layers.conv3d` alone).
    Args:
        inputs: a tensorflow tensor - the input data tensor
        filters: an int - the number of filters for the layers
        kernel_size: an int or tuple/list - the size of the convolution kernel
        strides: strides: an int - the stride for the convolutions
        dilation: an int - the dilation rate for the convolution
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tf.layers.conv2d layer with explicit padding
    """
    # determine if strided
    if not isinstance(strides, list):
        strides = [strides] * 3
    strided = True if any(stride > 1 for stride in strides) else False
    if strided:
        inputs = fixed_padding(inputs, kernel_size, strides, data_format)

    return tf.layers.conv3d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('same' if strided else 'valid'),
        use_bias=False, # no bias since batch norm is used
        kernel_initializer=tf.variance_scaling_initializer(),
        dilation_rate=dilation,
        data_format=data_format)


def upsample_layer(inputs, filters, kernel_size, strides, data_format):
    """
    Strided 2-D or 3-D transpose convolution upsampling using strides of 2
    Args:
        inputs: a tensorflow tensor - the input data tensor
        filters: an int - the number of filters for the layers
        kernel_size: an int or tuple/list - the size of the convolution kernel
        strides:  an int or list/tuple - the strides for  the convolution
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tf.layers.conv2d_transpose or tf.layers.conv3d_transpose layer depending on length of strides argument
    """

    # if strides is an int, assume 2D
    if isinstance(strides, int): strides = [strides, strides]

    # if list/tuple of len 2 is passed to strides, then return 2d function
    if len(strides) < 3:
        return tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,  # no bias since batch norm is used
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

    # if a len==3 list is passed, return 3d function
    elif isinstance(strides, (list, tuple)) and len(strides) == 3:
        return tf.layers.conv3d_transpose(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,  # no bias since batch norm is used
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

    # value error if strides is not an int or list/tuple of length 2 or 3
    else:
        raise ValueError("strides must be an int or list/tuple of length 2 or 3, instead got " + str(strides))


################################################################################
# Layer building blocks
################################################################################

# Simple residual layer
def residual_layer(tensor, n_filters=64, k_size=(3, 3), strides=(1, 1), dilation=(1, 1),
                   is_training=False, data_format="channels_last"):
    """
    Creates a single simple residual block with batch normalization and activation
    Modeled after: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
    Args:
        tensor: a tensorflow tensor - the input data tensor
        n_filters: an int - the number of filters for the layers
        k_size: an int or list/tuple - the size of the convolution kernel for the residual layers
        strides: an int or list/tuple - the stride for the convolutions of the first convolution layer
        dilation: an int or list/tuple - the dilation rate for the convolution
        is_training: a boolean - whether or not the model is training
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tensor: a tensorflow tensor - the output of the layer
    """
    # define shortcut
    shortcut = tensor

    # determine if strided
    if not isinstance(strides, (list, tuple)):
        strides = [strides] * 2
    strided = True if any(stride > 1 for stride in strides) else False

    # handle identity versus projection shortcut for outputs of different dimension
    if strided: # if strides are 2 for example, then project shortcut to output size
        # accomplish projection with a 1x1 convolution
        shortcut = conv2d_fixed_pad(tensor, n_filters, 1, strides, dilation=dilation, data_format=data_format)
        shortcut = batch_norm(shortcut, is_training)

    # Convolution block 1
    tensor = conv2d_fixed_pad(tensor, n_filters, k_size, strides, dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)
    tensor = activation(tensor)

    # Convolution block 2 (force strides==1)
    tensor = conv2d_fixed_pad(tensor, n_filters, k_size, strides=1, dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)

    # add shortcut to outputs and one final activation
    tensor = tf.add(tensor, shortcut)
    tensor = activation(tensor)

    return tensor


# Simple residual layer
def upsample_residual_layer(tensor, n_filters=64, k_size=(3, 3), strides=(1, 1), dilation=(1, 1),
                   is_training=False, data_format="channels_last"):
    """
    Creates a single simple residual block with batch normalization and activation followed by stride 2 upsampling
    Modeled after: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
    Args:
        tensor: a tensorflow tensor - the input data tensor
        n_filters: an int - the number of filters for the layers
        k_size: an int or list/tuple - the size of the convolution kernel for the residual layers
        strides: an int or list/tuple - the stride for the convolutions of the first convolution layer
        dilation: an int or list/tuple - the dilation rate for the convolution
        is_training: a boolean - whether or not the model is training
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tensor: a tensorflow tensor - the output of the layer
    """
    # define shortcut
    shortcut = tensor

    # determine if strided
    if not isinstance(strides, (list, tuple)):
        strides = [strides] * 2
    strided = True if any(stride > 1 for stride in strides) else False

    # handle identity versus projection shortcut for outputs of different dimension
    if strided: # if strides are 2 for example, then project shortcut to output size
        # accomplish projection with a 1x1 convolution
        shortcut = conv2d_fixed_pad(tensor, n_filters, 1, strides, dilation=dilation, data_format=data_format)
        shortcut = batch_norm(shortcut, is_training)

    # Convolution block 1
    tensor = conv2d_fixed_pad(tensor, n_filters, k_size, strides, dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)
    tensor = activation(tensor)

    # Convolution block 2 (force strides==1)
    tensor = conv2d_fixed_pad(tensor, n_filters, k_size, strides=1, dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)

    # add shortcut to outputs and one final activation
    tensor = tf.add(tensor, shortcut)
    tensor = activation(tensor)

    # upsample
    tensor = upsample_layer(tensor, n_filters, k_size, [2, 2], data_format)

    return tensor


# Bottleneck residual layer
def bneck_residual_layer(tensor, n_filters=64, k_size=(3, 3), strides=(1, 1), is_training=False, dilation=(1,1),
                         data_format="channels_last"):
    """
    Creates a single bottleneck residual block with batch normalization and activation
    Modeled after: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
    Args:
        tensor: a tensorflow tensor - the input data tensor
        n_filters: an int - the number of filters for the layers
        k_size: an int or list/tuple - the size of the convolution kernel for the residual layers
        strides: an int or list/tuple - the stride for the convolutions of the middle convolution layer
        is_training: a boolean - whether or not the model is training
        dilation: an int or list/tuple - the dilation rate for the convolution
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tensor: a tensorflow tensor - the output of the layer
    """
    # define shortcut
    shortcut = tensor

    # determine if strided
    if not isinstance(strides, (list, tuple)):
        strides = [strides] * 2
    strided = True if any(stride > 1 for stride in strides) else False

    # handle identity versus projection shortcut for outputs of different dimension
    if strided: # if strides are 2 for example, then project shortcut to output size
        # accomplish projection with a 1x1 convolution
        shortcut = conv2d_fixed_pad(tensor, n_filters, 1, strides, dilation=dilation, data_format=data_format)
        shortcut = batch_norm(shortcut, is_training)

    # Convolution block 1 (1x1 kernel)
    tensor = conv2d_fixed_pad(tensor, n_filters, kernel_size=1, strides=1, dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)
    tensor = activation(tensor)

    # Convolution block 2 (3x3)
    tensor = conv2d_fixed_pad(tensor, n_filters, k_size, strides, dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)
    tensor = activation(tensor)

    # Convolution block 3 (1x1 kernel) with 4x number of filters
    filters = 4 * n_filters
    tensor = conv2d_fixed_pad(tensor, filters, kernel_size=1, strides=1, dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)

    # add shortcut to outputs and one final activation
    tensor = tf.add(tensor, shortcut)
    tensor = activation(tensor)

    return tensor


# 3D residual block
def resid_layer3D(tensor, n_filters=64, k_size=(3, 3, 3), strides=(1, 1, 1), dilation=(1,1,1),
                  is_training=False, data_format="channels_last"):
    """
    Creates a 3D residual block with batch normalization and activation
    Modeled after: https://github.com/taigw/brats17/blob/master/util/MSNet.py
    Args:
        tensor: a tensorflow tensor - the input data tensor
        n_filters: an int - the number of filters for the layers
        k_size: an int or list/tuple - the size of the convolution kernel for the residual layers
        strides: an int or list/tuple - the stride for the convolutions of the first convolution layer
        dilation: an int or list/tuple - the dilation rate of the convolution
        is_training: a boolean - whether or not the model is training
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tensor: a tensorflow tensor - the output of the layer
    """
    # define shortcut
    shortcut = tensor

    # determine if strided
    if not isinstance(strides, (list, tuple)):
        strides = [strides] * 3
    strided = True if any(stride > 1 for stride in strides) else False

    # handle identity versus projection shortcut for outputs of different dimension
    if strided:  # if strides are 2 for example, then project shortcut to output size
        # accomplish projection with a 1x1 convolution
        shortcut = conv3d_fixed_pad(tensor, n_filters, 1, strides, dilation=dilation, data_format=data_format)
        shortcut = batch_norm(shortcut, is_training)

    # Convolution block 1
    tensor = conv3d_fixed_pad(tensor, n_filters, k_size, strides, dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)
    tensor = activation(tensor)

    # Convolution block 2 (force strides==1)
    tensor = conv3d_fixed_pad(tensor, n_filters, k_size, strides=[1, 1, 1], dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)

    # add shortcut to outputs and one final activation
    tensor = tf.add(tensor, shortcut)
    tensor = activation(tensor)

    return tensor


