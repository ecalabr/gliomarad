import tensorflow as tf


################################################################################
# Convenience functions for building the residual networks model.
################################################################################

def _fixed_padding(inputs, kernel_size, strides, data_format='channels_last', ):
    """
    Pads the input along the spatial dimensions independently of input size. Works for 2D and 3D
    :param inputs: (tf.tensor) the input tensor
    :param kernel_size: (tuple/list(int)) the convolution kernel size
    :param strides: (tuple/list(int)) the strides
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :return: A tensor with the same format as the input with the data either intact or padded (if kernel_size > 1)
    """

    # sanity checks
    if isinstance(kernel_size, (list, tuple)): kernel_size = kernel_size[0]
    if isinstance(strides, (list, tuple)): strides = strides[0]
    if not isinstance(kernel_size, int): raise ValueError("Kernel size must be an int or list/tuple of ints")
    if not isinstance(strides, int): raise ValueError("Strides must be an int or list/tuple of ints")
    if data_format not in ['channels_first', 'channels_last']:
        raise ValueError("Did not recognize data format: " + data_format)

    # determine pad start and end
    pad_total = kernel_size - 1
    p_beg = pad_total // 2
    p_end = pad_total - p_beg

    # handle 2D strides first
    if strides == 2:
        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [p_beg, p_end], [p_beg, p_end]])
        else:  # must be channels_last
            padded_inputs = tf.pad(inputs, [[0, 0], [p_beg, p_end], [p_beg, p_end], [0, 0]])
    else:
        raise ValueError("Stride must be 2 but is: " + str(strides))

    return padded_inputs


################################################################################
# Basic layer functions
################################################################################


def batch_norm(tensor, is_training, data_format='channels_last', name=None, reuse=False):
    """
    Returns a batch normalization layer with the specified parameters
    :param tensor: (tf.tensor) the input tensor
    :param is_training: (bool) whether or not the model is training
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param name: (str) the name of the operation
    :param reuse: (bool) whether or not to reuse weights for this layer
    :return: Returns a batch normalization layer with the specified parameters
    """

    # define axis based on data_format
    axis = -1 if data_format == "channels_last" else 1

    return tf.layers.batch_normalization(
        inputs=tensor,  # tensor
        axis=axis,  # axis
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
        training=is_training,  # is training?
        trainable=True,
        name=name,  # name
        reuse=reuse,  # reuse
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=True,  # use fused batch norm for performance tensorflow.org/performance/performance_guide
        virtual_batch_size=None,
        adjustment=None
    )


def activation(tensor, acti_type='leaky_relu', name=None):  # add optional type argument
    """
    Returns the specified activation function
    :param tensor: (tf.tensor) the input tensor
    :param acti_type: (str) the name of the desired activation function
    :param name: (str) the name of the operation
    :return: Returns the input tensor after activation.
    """

    if acti_type == "leaky_relu":
        act_func = tf.nn.leaky_relu(
            features=tensor,
            alpha=0.2,
            name=name)

    elif acti_type == "relu":
        act_func = tf.nn.relu(
            features=tensor,
            name=name)
    else:
        raise ValueError("Provided type " + str(acti_type) + " is not a known activation type")

    return act_func


def conv2d_fixed_pad(inputs, filters, kernel_size, strides, dilation, data_format, name=None, reuse=None):
    """
    2D strided convolutional layer with explicit padding.
    The padding is consistent and is based only on `kernel_size`, not on the
    dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    :param inputs: (tf.tensor) the input data tensor
    :param filters: (int) the number of filters used for the layer
    :param kernel_size: (tuple/list(int)) the convolution kernel size
    :param strides: (tuple/list(int)) the strides for the convolution
    :param dilation: (tuple/list(int)) the dilation used for the convolution
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param name: (str) the name of the operation
    :param reuse: (bool) whether or not to reuse weights for this layer
    :return: The 2d convolutional layer with the specified parameters
    """

    # sanity checks
    if isinstance(kernel_size, (list, tuple)): kernel_size = kernel_size[0]
    if isinstance(strides, (list, tuple)): strides = strides[0]
    if isinstance(dilation, (list, tuple)): dilation = dilation[0]
    if not isinstance(kernel_size, int): raise ValueError("Kernel size must be an int or list/tuple of ints")
    if not isinstance(strides, int): raise ValueError("Strides must be an int or list/tuple of ints")
    if not isinstance(dilation, int): raise ValueError("Dilation must be an int or list/tuple of ints")
    if data_format not in ['channels_first', 'channels_last']:
        raise ValueError("Did not recognize data format: " + data_format)

    # determine if strided
    strided = strides > 1
    if strided:
        inputs = _fixed_padding(inputs, kernel_size, strides, data_format)

    return tf.layers.conv2d(
        inputs=inputs,  # inputs
        filters=filters,  # filters
        kernel_size=kernel_size,  # kernel size
        strides=strides,  # strides
        padding=('valid' if strided else 'same'),  # if strided
        data_format=data_format,  # data format
        dilation_rate=dilation,  # dilation
        activation=None,
        use_bias=False,  # false since batch norm is used
        kernel_initializer=tf.variance_scaling_initializer(),  # custom initializer
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=name,  # name
        reuse=reuse  # reuse
    )


def upsample_layer(inputs, filters, kernel_size, strides, data_format, name=None, reuse=None):
    """
    Strided 2-D transpose convolution layer, which can be used for upsampling using strides >= 2
    :param inputs: (tf.tensor) the input data tensor
    :param filters: (int) the number of filters used for the layer
    :param kernel_size: (tuple/list(int)) the convolution kernel size
    :param strides: (tuple/list(int)) the strides for the convolution
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param name: (str) the name of the operation
    :param reuse: (bool) whether or not to reuse weights for this layer
    :return: The 2d transpose convolutional layer
    """

    # sanity checks
    if isinstance(kernel_size, (list, tuple)): kernel_size = kernel_size[0]
    if isinstance(strides, (list, tuple)): strides = strides[0]
    if not isinstance(kernel_size, int): raise ValueError("Kernel size must be an int or list/tuple of ints")
    if not isinstance(strides, int): raise ValueError("Strides must be an int or list/tuple of ints")
    if data_format not in ['channels_first', 'channels_last']:
        raise ValueError("Did not recognize data format: " + data_format)

    # return 2d conv transpose layer
    return tf.layers.conv2d_transpose(
        inputs=inputs,  # inputs
        filters=filters,  # filters
        kernel_size=kernel_size,  # kernel size
        strides=strides,  # strides
        padding='same',  # padding
        data_format=data_format,  # data format
        activation=None,
        use_bias=False,  # false since batch norm is use
        kernel_initializer=tf.variance_scaling_initializer(),  # custom initializer
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=name,
        reuse=reuse
    )


def maxpool_layer_2d(tensor, pool_size, strides, data_format, name=None):
    """
    Returns a maxpool 2d layer
    :param tensor:
    :param pool_size:
    :param strides:
    :param data_format:
    :param name:
    :return:
    """

    # sanity checks
    if isinstance(pool_size, (list, tuple)): pool_size = pool_size[0]
    if isinstance(strides, (list, tuple)): strides = strides[0]
    if not isinstance(pool_size, int): raise ValueError("Pool size must be an int or list/tuple of ints")
    if not isinstance(strides, int): raise ValueError("Strides must be an int or list/tuple of ints")
    if data_format not in ['channels_first', 'channels_last']:
        raise ValueError("Did not recognize data format: " + data_format)

    # return maxpool layer
    return tf.layers.max_pooling2d(
            inputs=tensor,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=data_format,
            name=name)


################################################################################
# Layer building blocks
################################################################################


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
