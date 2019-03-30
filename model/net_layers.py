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

    # if 1D kernel or strides provided, assume 2D and square
    if not isinstance(kernel_size, (list, tuple)): kernel_size = [kernel_size, kernel_size]
    if not isinstance(strides, (list, tuple)): strides = [strides, strides]
    if any([stride not in [1, 2] for stride in strides]) or not any([stride == 2 for stride in strides]):
        raise ValueError("To use _fixed_padding, at least one stride must be 2, and no strides can be outside [1, 2]")

    # define variable
    pads = []

    # go through each dimension of input and determine padding
    for kernel in kernel_size:
        # determine pad start and end
        pad_total = kernel - 1
        p_beg = pad_total // 2
        p_end = pad_total - p_beg
        pads.append([p_beg, p_end])

    # do padding according to data format
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0]] + pads)
    else:  # must be channels_last
        padded_inputs = tf.pad(inputs, [[0, 0]] + pads + [[0, 0]])

    return padded_inputs


################################################################################
# Basic layer functions 2D
################################################################################


def batch_norm(tensor, is_training, data_format='channels_last', name=None, reuse=None):
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
        momentum=0.9,  # set to 0.9 per https://github.com/tensorflow/tensorflow/issues/1122
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

    # no activation
    if acti_type == 'none':
        act_func = tensor

    # leaky relu
    elif acti_type == 'leaky_relu':
        act_func = tf.nn.leaky_relu(
            features=tensor,
            alpha=0.2,
            name=name)

    # normal relu
    elif acti_type == 'relu':
        act_func = tf.nn.relu(
            features=tensor,
            name=name)

     # not implemented
    else:
        raise ValueError("Provided type " + str(acti_type) + " is not a known activation type")

    return act_func


def conv2d_fixed_pad(tensor, filters, kernel_size, strides, dilation, data_format, name=None, reuse=None):
    """
    2D strided convolutional layer with explicit padding.
    The padding is consistent and is based only on `kernel_size`, not on the
    dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    :param tensor: (tf.tensor) the input data tensor
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
        tensor = _fixed_padding(tensor, kernel_size, strides, data_format)

    return tf.layers.conv2d(
        inputs=tensor,  # inputs
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


def deconv2d_layer(inputs, filters, kernel_size, strides, data_format, name=None, reuse=None):
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


def maxpool2d_layer(tensor, pool_size, strides, data_format, name=None):
    """
    Returns a maxpool 2d layer
    :param tensor: (tf.tensor) the input tensor
    :param pool_size: (tuple/list(int)) the pool size for maxpool
    :param strides: (tuple/list(int)) the strides for the convolution
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param name: (str) the name of the operation
    :return: Returns a maxpool2d layer with the specified parameters
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


def upsample2d_layer(tensor, factor, data_format, name=None):
    """
    Upsampling with bilinear interpolation.
    :param tensor: (tf.tensor) the input tensor
    :param factor: (int) the resampling factor
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param name: (str) the name of the operation
    :return: Returns an tensor with the desired upsampling factor
    """

    # handle channels first data
    if data_format == 'channels_first':
        tensor = tf.transpose(tensor, perm=[0, 2, 3, 1], name=name + '_transpose1')

    # get output shape
    size = tensor.get_shape().as_list()[1:3]
    size = [dim * factor for dim in size]

    # do resizing
    tensor = tf.image.resize_images(tensor, size=size, method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

    # handle channels first data
    if data_format == 'channels_first':
        tensor = tf.transpose(tensor, perm=[0, 1, 2, 3], name=name + '_transpose2')

    return tensor


################################################################################
# Basic layer functions 3D
################################################################################


def conv3d_fixed_pad(tensor, filters, kernel_size, strides, dilation, data_format, name=None, reuse=None):
    """
    3D strided convolutional layer with explicit padding.
    The padding is consistent and is based only on `kernel_size`, not on the
    dimensions of `inputs` (as opposed to using `tf.layers.conv3d` alone).
    :param tensor: (tf.tensor) the input data tensor
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
    if not len(kernel_size) == 3: raise ValueError("Kernel must be length 3, but is length " + str(len(kernel_size)))
    if not len(strides) == 3: raise ValueError("Strides must be length 3, but is length " + str(len(strides)))
    if not len(dilation) == 3: raise ValueError("dilation must be length 3, but is length " + str(len(dilation)))
    if data_format not in ['channels_first', 'channels_last']:
        raise ValueError("Did not recognize data format: " + data_format)

    # determine if strided
    strided = any([stride > 1 for stride in strides])
    if strided:
        tensor = _fixed_padding(tensor, kernel_size, strides, data_format)

    return tf.layers.conv3d(
        inputs=tensor,  # inputs
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


def deconv3d_layer(inputs, filters, kernel_size, strides, data_format, name=None, reuse=None):
    """
    Strided 3-D transpose convolution layer, which can be used for upsampling using strides >= 2
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
    if not len(kernel_size) == 3: raise ValueError("Kernel must be length 3, but is length " + str(len(kernel_size)))
    if not len(strides) == 3: raise ValueError("Strides must be length 3, but is length " + str(len(strides)))
    if data_format not in ['channels_first', 'channels_last']:
        raise ValueError("Did not recognize data format: " + data_format)

    # return 3d conv transpose layer
    return tf.layers.conv3d_transpose(
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


################################################################################
# Layer building blocks 2D
################################################################################


def conv2d_block(tensor, filters, kernel_size, strides, dil, data_format, name, reuse, dropout, is_training, acti_type):
    """
    Basic conv 2d block with batchnorm, activation, and optional dropout
    :param tensor: (tf.tensor) the input data tensor
    :param filters: (int) the number of filters used for the layer
    :param kernel_size: (tuple/list(int)) the convolution kernel size
    :param strides: (tuple/list(int)) the strides for the convolution
    :param dil: (tuple/list(int)) the dilation used for the convolution
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param name: (str) the name of the operation
    :param reuse: (bool) whether or not to reuse weights for this layer
    :param dropout: (float) the dropout rate. If zero, dropout is not used.
    :param is_training: (bool) whether or not the model is training
    :param acti_type: (str) the type of activation to be uzed
    :return: Returns a basic conv2d block with the specified parameters
    """
    # basic block of conv, batch norm, activation, and optional dropout
    tensor = conv2d_fixed_pad(tensor, filters, kernel_size, strides, dil, data_format, name + '_conv', reuse)
    tensor = batch_norm(tensor, is_training, data_format, name + '_bn', reuse)
    tensor = activation(tensor, acti_type, name + '_act')
    if dropout > 0.:
        tensor = tf.layers.dropout(tensor, rate=dropout, training=is_training, name=name + '_dropout')

    return tensor


def deconv2d_block(tensor, filters, kernel_size, strides, data_format, name, reuse, dropout, is_training, acti_type):
    """
    Basic transpose conv 2d block with batchnorm, activation, and optional dropout
    :param tensor: (tf.tensor) the input data tensor
    :param filters: (int) the number of filters used for the layer
    :param kernel_size: (tuple/list(int)) the convolution kernel size
    :param strides: (tuple/list(int)) the strides for the convolution
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param name: (str) the name of the operation
    :param reuse: (bool) whether or not to reuse weights for this layer
    :param dropout: (float) the dropout rate. If zero, dropout is not used.
    :param is_training: (bool) whether or not the model is training
    :param acti_type: (str) the type of activation to be uzed
    :return: Returns a basic transpose conv2d block with the specified parameters
    """
    # basic block of conv, batch norm, activation, and optional dropout
    tensor = deconv2d_layer(tensor, filters, kernel_size, strides, data_format, name + '_deconv', reuse)
    tensor = batch_norm(tensor, is_training, data_format, name + '_bn', reuse)
    tensor = activation(tensor, acti_type, name + '_act')
    if dropout > 0.:
        tensor = tf.layers.dropout(tensor, rate=dropout, training=is_training, name=name + '_dropout')
    return tensor


def resid2d_layer(tensor, filt, ksize, strides, dil, dfmt, name, reuse, dropout, is_training, act):
    """
    Creates a 2D single simple residual block with batch normalization and activation
    Modeled after: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
    :param tensor: (tf.tensor) the inputa data tensor
    :param filt: (int) the number of filters for the convolutions
    :param ksize: (list/tuple(int)) the kernel size for convolutions
    :param strides: (list/tuple(int)) the strides for convolutions
    :param dil: (list/tuple(int)) the dilation for convolutions
    :param dfmt: (str) the tf data format 'channels_first' or 'channels_last'
    :param name: (str) the name of the operation
    :param reuse: (bool) whether or not to reuse weights for this layer
    :param dropout: (float) the dropout rate, if zero, no dropout layer is applied
    :param is_training: (bool) whether or not the model is training
    :param act: (str) the name of te activation method, e.g. 'leaky_relu'
    :return: returns a residual layer as defined in the resnet example above.
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
        shortcut = conv2d_fixed_pad(tensor, filt, 1, strides, dil, dfmt, name + '_shrtct_conv', reuse)
        shortcut = batch_norm(shortcut, is_training, dfmt, name + '_shrtct_bn', reuse)

    # Convolution block 1
    tensor = conv2d_fixed_pad(tensor, filt, ksize, strides, dil, dfmt, name + '_resid_conv1', reuse)
    tensor = batch_norm(tensor, is_training, dfmt, name + '_resid_conv1_bn', reuse)
    tensor = activation(tensor, act, name + '_resid_conv1_act')

    # Convolution block 2 (force strides==1)
    tensor = conv2d_fixed_pad(tensor, filt, ksize, [1, 1], dil, dfmt, name + '_resid_conv2', reuse)
    tensor = batch_norm(tensor, is_training, dfmt, name + '_resid_conv2_bn', reuse)

    # fuse shortcut with outputs and do one final activation
    tensor = tf.add(tensor, shortcut, name + '_resid_shrtct_add')
    tensor = activation(tensor, act, name + '_resid_conv2_act')

    # optional dropout layer
    if dropout > 0.:
        tensor = tf.layers.dropout(tensor, rate=dropout, training=is_training, name=name + '_dropout')

    return tensor


def resid_us_layer(tensor, n_filters, k_size, strides, dilation, is_training, data_format, act_type, name, reuse=False):
    """
    Creates a 2D single transpose convolution residual block with batch normalization and activation
    Modeled after: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
    :param tensor: (tf.tensor) the inputa data tensor
    :param n_filters: (int) the number of filters for the convolutions
    :param k_size: (list/tuple(int)) the kernel size for convolutions
    :param strides: (list/tuple(int)) the strides for convolutions
    :param dilation: (list/tuple(int)) the dilation for convolutions
    :param is_training: (bool) whether or not the model is training
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param act_type: (str) the name of te activation method, e.g. 'leaky_relu'
    :param name: (str) the name of the operation
    :param reuse: (bool) whether or not to reuse weights for this layer
    :return: returns a transpose residual layer as defined in the resnet example above.
    """

    # ensure strided
    if isinstance(strides, int): strides = [strides] * 2
    if not isinstance(strides, (list, tuple)): raise ValueError("Strides must be an int or list/tuple of ints")
    if strides[0] <= 1: raise ValueError("Strides must be greater than or equal to 2 for upsampling layer")

    # projection shortcut for upsample layer accomplish projection with a 1x1 convolution
    shortcut = deconv2d_layer(tensor, n_filters, 1, strides, data_format, name + '_shrtct_conv', reuse)
    shortcut = batch_norm(shortcut, is_training, data_format, name + '_sc_batchnorm', reuse)

    # Convolution block 1
    tensor = deconv2d_layer(tensor, n_filters, k_size, strides, data_format, name + '_resid_conv1', reuse)
    tensor = batch_norm(tensor, is_training, data_format, name + '_resid_conv1_batchnorm', reuse)
    tensor = act_type(tensor, act_type, name + '_resid_conv1_act')

    # Convolution block 2 (force strides==1)
    tensor = conv2d_fixed_pad(tensor, n_filters, k_size, [1, 1], dilation, data_format, name + '_resid_conv2', reuse)
    tensor = batch_norm(tensor, is_training, data_format, name + '_resid_conv2_batchnorm', reuse)

    # fuse shortcut with outputs and do one final activation
    tensor = tf.add(tensor, shortcut)
    tensor = act_type(tensor, act_type, name + '_resid_conv2_act')

    return tensor


def bneck_res_layer(tensor, ksize, in_filt, resample, dropout, is_training, data_format, act_type, name, reuse=False):
    """
    Creates a 2D bottleneck residual layer with optional upsampling and downsampling
    https://vitalab.github.io/deep-learning/2017/05/08/resunet.html
    :param tensor: (tf.tensor) the inputa data tensor
    :param ksize: (list/tuple(ints)) the kernel size for the middle convolutions of bottleneck blocks
    :param in_filt: (int) the number of base filters for the input/output of the residual block
    :param resample: (int) in rage [0, 2], 0 = no resampling, 1 = downsample by 2, 2 = upsample by 2
    :param dropout: (float) the dropout rate, if zero, no dropout layer is applied
    :param is_training: (bool) whether or not the model is training
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param act_type: (str) the name of te activation method, e.g. 'leaky_relu'
    :param name: (str) the name of the operation
    :param reuse: (bool) whether or not to reuse weights for this layer
    :return: returns a bottleneck residual block with the specified parameters
            including optional upsampling or downsampling.
    """

    # sanity checks
    if isinstance(ksize, int): ksize = [ksize] * 2

    # define basic parameters
    dil = [1, 1]  # do not use dilation
    filters = int(round(in_filt / 4))  # filters for bottleneck layers are 1/4 of input/output filters

    # shortcut with projection if upsampling or downsampling, if not just identity
    if resample == 1:  # downsample projection
        shortcut = conv2d_fixed_pad(tensor, in_filt, [1, 1], [2, 2], dil, data_format, name + '_sc_us_proj', reuse)
    elif resample == 2:  # upsample projection
        shortcut = deconv2d_layer(tensor, in_filt, [1, 1], [2, 2], data_format, name + '_sc_us_proj', reuse)
    else:
        shortcut = tf.identity(tensor, name + '_sc_identity')

    # bottleneck residual block
    # first 1x1 conv block with optional downsampling
    tensor = batch_norm(tensor, is_training, data_format, name + '_bn_1', reuse)
    tensor = activation(tensor, act_type, name + '_act_1')
    if resample == 1:  # if downsampling
        tensor = conv2d_fixed_pad(tensor, filters, [1, 1], [2, 2], dil, data_format, name + '_ds_conv1x1_1', reuse)
    else:
        tensor = conv2d_fixed_pad(tensor, filters, [1, 1], [1, 1], dil, data_format, name + '_conv1x1_1', reuse)

    # 3x3 (or other specified non-unity kernel) conv block
    tensor = batch_norm(tensor, is_training, data_format, name + '_bn_2', reuse)
    tensor = activation(tensor, act_type, name + '_act_2')
    layer_name = name + '_conv' + str(ksize[0]) + 'x' + str(ksize[1]) + '_2'
    tensor = conv2d_fixed_pad(tensor, filters, ksize, [1, 1], dil, data_format, layer_name, reuse)

    # optional upsampling, in this case as an additional transpose conv, could also do linear upsamp without params?
    if resample == 2:  # if upsampling
        tensor = deconv2d_layer(tensor, filters, [1, 1], [2, 2], data_format, name + '_transpose_conv1x1', reuse)

    # second 1x1 conv block
    tensor = batch_norm(tensor, is_training, data_format, name + '_bn_3', reuse)
    tensor = activation(tensor, act_type, name + '_act_3')
    tensor = conv2d_fixed_pad(tensor, in_filt, [1, 1], [1, 1], dil, data_format, name + '_conv1x1_3', reuse)

    # optional dropout layer
    if dropout > 0.:
        tensor = tf.layers.dropout(tensor, rate=dropout, training=is_training, name=name + '_dropout')

    # shortcut fusion
    tensor = tf.add(tensor, shortcut)

    return tensor


def embedding_block(tensor, ksize, filt, drpout, is_training, dfmt, act, name, reuse=False):

    # set some basic constant params
    strd = [1, 1]
    dil = [1, 1]

    # input tensor forks to 3x3x1 conv and 3x3xfilters conv
    # transform fork
    layer_name = name + '_transform'
    trans_tensor = conv2d_block(tensor, filt, ksize, strd, dil, dfmt, layer_name, reuse, drpout, is_training, act)

    # embedding fork
    layer_name = name + '_embed'
    embed_tensor = conv2d_fixed_pad(tensor, 1, [1, 1], [1, 1], [1, 1], dfmt, layer_name, reuse)

    # concatenate transform and embedding forks
    layer_name = name + '_transform_concat'
    axis = 1 if dfmt == 'channels_first' else -1
    tensor = tf.concat([trans_tensor, embed_tensor], axis, name=layer_name)

    # final convolution
    layer_name = name + '_final_conv'
    tensor = conv2d_block(tensor, filt, ksize, strd, dil, dfmt, layer_name, reuse, drpout, is_training, act)

    return tensor, embed_tensor


################################################################################
# Layer building blocks 3D
################################################################################


def resid3d_layer(tensor, filt, ksize, strides, dil, dfmt, name, reuse, dropout, is_training, act):
    """
    Creates a 3D single simple residual block with batch normalization and activation
    Modeled after: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
    :param tensor: (tf.tensor) the inputa data tensor
    :param filt: (int) the number of filters for the convolutions
    :param ksize: (list/tuple(int)) the kernel size for convolutions
    :param strides: (list/tuple(int)) the strides for convolutions
    :param dil: (list/tuple(int)) the dilation for convolutions
    :param dfmt: (str) the tf data format 'channels_first' or 'channels_last'
    :param name: (str) the name of the operation
    :param reuse: (bool) whether or not to reuse weights for this layer
    :param dropout: (float) the dropout rate, if zero, no dropout layer is applied
    :param is_training: (bool) whether or not the model is training
    :param act: (str) the name of te activation method, e.g. 'leaky_relu'
    :return: returns a residual layer as defined in the resnet example above.
    """

    # define shortcut
    shortcut = tensor

    # determine if strided
    if not isinstance(strides, (list, tuple)):
        strides = [strides] * 3
    strided = True if any(stride > 1 for stride in strides) else False

    # handle identity versus projection shortcut for outputs of different dimension
    if strided: # if strides are 2 for example, then project shortcut to output size
        # accomplish projection with a 1x1 convolution
        shortcut = conv3d_fixed_pad(tensor, filt, [1, 1, 1], strides, dil, dfmt, name + '_shrtct_conv', reuse)
        shortcut = batch_norm(shortcut, is_training, dfmt, name + '_shrtct_bn', reuse)

    # Convolution block 1
    tensor = conv3d_fixed_pad(tensor, filt, ksize, strides, dil, dfmt, name + '_resid_conv1', reuse)
    tensor = batch_norm(tensor, is_training, dfmt, name + '_resid_conv1_bn', reuse)
    tensor = activation(tensor, act, name + '_resid_conv1_act')

    # Convolution block 2 (force strides==1)
    tensor = conv3d_fixed_pad(tensor, filt, ksize, [1, 1, 1], dil, dfmt, name + '_resid_conv2', reuse)
    tensor = batch_norm(tensor, is_training, dfmt, name + '_resid_conv2_bn', reuse)

    # fuse shortcut with outputs and do one final activation
    tensor = tf.add(tensor, shortcut, name + '_resid_shrtct_add')
    tensor = activation(tensor, act, name + '_resid_conv2_act')

    # optional dropout layer
    if dropout > 0.:
        tensor = tf.layers.dropout(tensor, rate=dropout, training=is_training, name=name + '_dropout')

    return tensor