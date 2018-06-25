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
    return tf.layers.batch_normalization(
        inputs=tensor,
        axis=-1, # channels last, assume last channel is correct axis
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
        data_format=data_format)


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
        strides: a list - strides, determines which dimensions to pad, must have length == input dimensions (-batchsize)
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        A tensor with the same format as the input with the data either intact or padded (if kernel_size > 1)
    """
    # determine pad start and end
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    # determine which dimensions to pad
    if len(strides) == 3 and all(stride > 1 for stride in strides): # full 3D
        if data_format == "channels_first":
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [pad_beg, pad_end]])
        elif data_format == "channels_last":
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        else:
            raise ValueError("'data_format' should be 'channels_first' or 'channels_last', but is " + str(data_format))
    elif 1 < len(strides) < 4 and all(stride > 1 for stride in strides[0:1]): # 2D or 3D with stride of 1 in third dim
        if data_format == "channels_first":
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
        elif data_format == "channels_last":
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        else:
            raise ValueError("'data_format' should be 'channels_first' or 'channels_last', but is " + str(data_format))
    else:
        raise ValueError("Expected strides of [2, 2], [2, 2, 2], or [2, 2, 1] but got " + str(strides))

    return padded_inputs


def conv2d_fixed_pad(inputs, filters, kernel_size, strides, dilation, data_format='channels_last'):
    """
    Strided 2-D convolution with explicit padding.
    Args:
        inputs: a tensorflow tensor - the input data tensor
        filters: an int - the number of filters for the layers
        kernel_size: an int - the size of the convolution kernel
        strides: strides: an int - the stride for the convolutions
        dilation: an int - the dilation rate for the convolution
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
        strides:
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tf.layers.conv2d_transpose or tf.layers.conv3d_transpose layer depending on length of strides argument
    """

    # if an int or list/tuple of len 2 is passed to strides, then return 2d function
    if isinstance(strides, int) or len(strides) < 3:
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
def residual_layer(tensor, n_filters=64, k_size=3, strides=1, dilation=1,
                   is_training=False, data_format="channels_last"):
    """
    Creates a single simple residual block with batch normalization and activation
    Modeled after: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
    Args:
        tensor: a tensorflow tensor - the input data tensor
        n_filters: an int - the number of filters for the layers
        k_size: an int - the size of the convolution kernel for the residual layers
        strides: an int - the stride for the convolutions of the residual layers
        dilation: an int - the dilation rate for the convolution
        is_training: a boolean - whether or not the model is training
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tensor: a tensorflow tensor - the output of the layer
    """
    # define shortcut
    shortcut = tensor

    # determine if strided
    if not isinstance(strides, list):
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

    # Convolution block 2
    tensor = conv2d_fixed_pad(tensor, n_filters, k_size, strides, dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)

    # add shortcut to outputs and one final activation
    tensor = tf.add(tensor, shortcut)
    tensor = activation(tensor)

    return tensor


# Bottleneck residual layer
def bneck_residual_layer(tensor, n_filters=64, k_size=3, strides=1, is_training=False, dilation=1,
                         data_format="channels_last"):
    """
    Creates a single bottleneck residual block with batch normalization and activation
    Modeled after: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
    Args:
        tensor: a tensorflow tensor - the input data tensor
        n_filters: an int - the number of filters for the layers
        k_size: an int - the size of the convolution kernel for the residual layers
        strides: an int - the stride for the convolutions of the residual layers
        is_training: a boolean - whether or not the model is training
        dilation: an int - the dilation rate for the convolution
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        tensor: a tensorflow tensor - the output of the layer
    """
    # define shortcut
    shortcut = tensor

    # determine if strided
    if not isinstance(strides, list):
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
        strides: an int or list/tuple - the stride for the convolutions of the residual layers
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

    # Convolution block 2
    tensor = conv3d_fixed_pad(tensor, n_filters, k_size, strides, dilation=dilation, data_format=data_format)
    tensor = batch_norm(tensor, is_training, data_format=data_format)

    # add shortcut to outputs and one final activation
    tensor = tf.add(tensor, shortcut)
    tensor = activation(tensor)

    return tensor


################################################################################
# Networks
################################################################################

def resnet(tensor, is_training, n_classes, n_blocks=16, ds=(3, 7, 13),
           base_filters=64, k_size=3, data_format="channels_last"):
    """Creates a ResNet using n=n_blocks simple resnet blocks and n=ds downsampling layers
    Args:
        tensor: a tensorflow tensor - the input data tensor
        is_training: a boolean - whether or not the model is training
        n_classes: an int - the number of output classes for logits
        n_blocks: an int - the number of residual blocks to build into the network
        ds: a list/tuple of ints - the indices (starting from 0) of the downsampling blocks (i.e. where strides=2)
        base_filters: an int - the number of filters for the first layer of the network (auto scaled for deeper layers)
        k_size: an int - the size of the convolution kernel for the residual layers
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        logits: a tensorflow tensor - unscaled logits of size n_classes
    """
    # set variable scope
    with tf.variable_scope("resnet"):

        # no dilation for resnet
        dilation = 1

        # intial convolution layer with stride=2 (output size = dim/4)
        tensor = conv2d_fixed_pad(tensor, base_filters, k_size, strides=2, dilation=dilation, data_format=data_format)
        tensor = tf.identity(tensor, "init_conv")
        tensor = batch_norm(tensor,is_training,data_format)
        tensor = activation(tensor)

        # max pool layer 1
        tensor = tf.layers.max_pooling2d(inputs=tensor, pool_size=2, strides=2, padding='SAME', data_format=data_format)
        tensor = tf.identity(tensor, "initial_max_pool")

        # build sets of residual blocks
        filters = base_filters
        for i in range(n_blocks):
            # if this is a downsampling block, strides=2 and filters is doubled, else strides=1
            if i in ds:
                strides = 2
                filters = 2 * filters
            else:
                strides = 1
            tensor = residual_layer(tensor, filters, k_size, strides, dilation, is_training, data_format)
            tensor = tf.identity(tensor, "block_layer_" + str(i).zfill(len(str(n_blocks))))

        # final average pool layer (replaced with equivalent reduce mean for performance)
        axes = [2, 3] if data_format == 'channels_first' else [1, 2]
        tensor = tf.reduce_mean(tensor, axes, keepdims=True)
        tensor = tf.identity(tensor, 'final_reduce_mean')

        # final dense layer with 1000 neurons
        final_size = np.prod(tensor.get_shape().as_list()[1:]) # not sure if this is correct
        tensor = tf.reshape(tensor, [-1, final_size]) # completely flattens tensor
        tensor = tf.layers.dense(inputs=tensor, units=1000, name="final_dense", activation=activation)

        # logits layer
        logits = tf.layers.dense(inputs=tensor, units=n_classes, name="logits_layer")

    return logits


def MSNet(tensor, is_training, n_classes, k_size= (3, 3, 1), base_filters=32, data_format="channels_last"):
    """Creates a MSNet modeled after: https://github.com/taigw/brats17/blob/master/util/MSNet.py
    Args:
        tensor: a tensorflow tensor - the input data tensor
        is_training: a boolean - whether or not the model is training
        n_classes: an int - the number of output classes for logits
        base_filters: an int - the number of filters for the first layer of the network (auto scaled for deeper layers)
        k_size: an int or list/tuple of ints - the size of the convolution kernel for the residual layers
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        logits: a tensorflow tensor - unscaled logits of size n_classes
    """
    # set variable scope
    with tf.variable_scope("resnet"):

        # set default values
        filters = base_filters
        ksize = list(k_size)
        dilation = [1, 1, 1]
        strides = [1, 1, 1]

        # first set of residual blocks with 3x3x1 itraslice convolutions
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_1_1")
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_1_2")

        # first interslice 1x1x3 convolution layer
        tensor = conv3d_fixed_pad(tensor, filters, [1, 1, 3], strides, dilation, data_format)
        tensor = tf.identity(tensor, "interslice_1")
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)

        # first downsampling layer with strides=2
        filters = base_filters * 2
        tensor = conv3d_fixed_pad(tensor, filters, ksize, [2, 2, 1], dilation, data_format)
        tensor = tf.identity(tensor, "downsample_1")
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)

        # second set of residual blocks
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_2_1")
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_2_2")

        # second interslice 1x1x3 convolution layer
        fuse1 = conv3d_fixed_pad(tensor, filters, [1, 1, 3], strides, dilation, data_format)
        fuse1 = tf.identity(fuse1, "interslice_2")
        fuse1 = batch_norm(fuse1, is_training, data_format)
        fuse1 = activation(fuse1)

        # second downsampling layer with strides=2
        filters = base_filters * 4
        tensor = conv3d_fixed_pad(fuse1, filters, ksize, [2, 2, 1], dilation, data_format)
        tensor = tf.identity(tensor, "downsample_2")
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)

        # first set of residual blocks with dilated convolutions and dilation rates of 1, 2, 3 respectively
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_3_1")
        tensor = resid_layer3D(tensor, filters, ksize, strides, [2, 2, 1], is_training, data_format)
        tensor = tf.identity(tensor, "block_3_2")
        tensor = resid_layer3D(tensor, filters, ksize, strides, [3, 3, 1], is_training, data_format)
        tensor = tf.identity(tensor, "block_3_3")

        # third interslice 1x1x3 convolution layer
        fuse2 = conv3d_fixed_pad(tensor, filters, [1, 1, 3], strides, dilation, data_format)
        fuse2 = tf.identity(fuse2, "interslice_3")
        fuse2 = batch_norm(fuse2, is_training, data_format)
        fuse2 = activation(fuse2)

        # second set of residual blocks with dilated convolutions and dilation rates of 1, 2, 3 respectively
        tensor = resid_layer3D(fuse2, filters, ksize, strides, [3, 3, 1], is_training, data_format)
        tensor = tf.identity(tensor, "block_4_1")
        tensor = resid_layer3D(tensor, filters, ksize, strides, [2, 2, 1], is_training, data_format)
        tensor = tf.identity(tensor, "block_4_2")
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_4_3")

        # fourth interslice 1x1x3 convolution layer
        fuse3 = conv3d_fixed_pad(tensor, filters, [1, 1, 3], strides, dilation, data_format)
        fuse3 = tf.identity(fuse3, "interslice_4")
        fuse3 = batch_norm(fuse3, is_training, data_format)
        fuse3 = activation(fuse3)

        # first fusion layer with 2x upsampling (n_classes outputs)
        fuse1 = upsample_layer(fuse1, n_classes, ksize, [2, 2, 1], data_format)
        fuse1 = batch_norm(fuse1, is_training, data_format)
        fuse1 = activation(fuse1)

        # second fusion layer with 4x upsampling (n_classes * 2 outputs)
        fuse2 = upsample_layer(fuse2, n_classes*2, ksize, [2, 2, 1], data_format)
        fuse2 = batch_norm(fuse2, is_training, data_format)
        fuse2 = activation(fuse2)
        fuse2 = upsample_layer(fuse2, n_classes*2, ksize, [2, 2, 1], data_format)
        fuse2 = batch_norm(fuse2, is_training, data_format)
        fuse2 = activation(fuse2)

        # third fusion layer with 4x upsampling (n_classes * 4 outputs)
        fuse3 = upsample_layer(fuse3, n_classes*4, ksize, [2, 2, 1], data_format)
        fuse3 = batch_norm(fuse3, is_training, data_format)
        fuse3 = activation(fuse3)
        fuse3 = upsample_layer(fuse3, n_classes*4, ksize, [2, 2, 1], data_format)
        fuse3 = batch_norm(fuse3, is_training, data_format)
        fuse3 = activation(fuse3)

        # concatenate fusion layers
        tensor = tf.concat([fuse1, fuse2, fuse3], name="concatenate", axis=-1 if data_format=="channels_last" else 1)

        # final layer (n_classes outputs)
        tensor = conv3d_fixed_pad(tensor, n_classes, ksize, strides, dilation, data_format)

        return tensor
