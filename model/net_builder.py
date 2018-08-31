from net_layers import *
import sys

# Define the various networks that can be used

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


def MSNet(tensor, is_training, n_classes, k_size=(1, 3, 3), base_filters=32, data_format="channels_last"):
    """Creates a MSNet modeled after: https://github.com/taigw/brats17/blob/master/util/MSNet.py
    Args:
        tensor: a tensorflow tensor - the input data tensor
        is_training: a boolean - whether or not the model is training
        n_classes: an int - the number of output classes for logits
        base_filters: an int - the number of filters for the first layer of the network (auto scaled for deeper layers)
        k_size: an int or list/tuple of ints - the size of the convolution kernel for the residual layers
        data_format: a string - "channels_first" for NCDHW data or "channels_last" for NDHWC
    Returns:
        logits: a tensorflow tensor - unscaled logits of size n_classes
    """
    # set variable scope
    with tf.variable_scope("resnet"):

        # set default values
        ksize = list(k_size) if isinstance(k_size, (tuple, list)) else [1, k_size, k_size]
        filters = base_filters
        dilation = [1, 1, 1]
        strides = [1, 1, 1]

        # first set of residual blocks with 1x3x3 (DxWxH) itraslice convolutions
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_1_1")
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_1_2")

        # first interslice 3x1x1 (DxWxH) convolution layer
        tensor = conv3d_fixed_pad(tensor, filters, [3, 1, 1], strides, dilation, data_format)
        tensor = tf.identity(tensor, "interslice_1")
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)

        # second downsampling residual block with strides==2 in HxW (original was single conv layer without residual)
        filters = base_filters * 2
        tensor = resid_layer3D(tensor, filters, ksize, [1, 2, 2], dilation, is_training, data_format)
        tensor = tf.identity(tensor, "downsample_1")

        # second set of residual blocks
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_2_1")
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_2_2")

        # second interslice 1x1x3 convolution layer
        fuse1 = conv3d_fixed_pad(tensor, filters, [3, 1, 1], strides, dilation, data_format)
        fuse1 = tf.identity(fuse1, "interslice_2")
        fuse1 = batch_norm(fuse1, is_training, data_format)
        fuse1 = activation(fuse1)

        # second downsampling residual block with strides==2 in HxW (original was single conv layer without residual)
        filters = base_filters * 4
        tensor = resid_layer3D(tensor, filters, ksize, [1, 2, 2], dilation, is_training, data_format)
        tensor = tf.identity(tensor, "downsample_2")

        # first set of residual blocks with dilated convolutions and dilation rates of 1, 2, 3 respectively in HxW
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_3_1")
        tensor = resid_layer3D(tensor, filters, ksize, strides, [1, 2, 2], is_training, data_format)
        tensor = tf.identity(tensor, "block_3_2")
        tensor = resid_layer3D(tensor, filters, ksize, strides, [1, 3, 3], is_training, data_format)
        tensor = tf.identity(tensor, "block_3_3")

        # third interslice 3x1x1 (DxWxH) convolution layer
        fuse2 = conv3d_fixed_pad(tensor, filters, [3, 1, 1], strides, dilation, data_format)
        fuse2 = tf.identity(fuse2, "interslice_3")
        fuse2 = batch_norm(fuse2, is_training, data_format)
        fuse2 = activation(fuse2)

        # second set of residual blocks with dilated convolutions and dilation rates of 1, 2, and 3 respectively in HxW
        tensor = resid_layer3D(fuse2, filters, ksize, strides, [1, 3, 3], is_training, data_format)
        tensor = tf.identity(tensor, "block_4_1")
        tensor = resid_layer3D(tensor, filters, ksize, strides, [1, 2, 2], is_training, data_format)
        tensor = tf.identity(tensor, "block_4_2")
        tensor = resid_layer3D(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "block_4_3")

        # fourth interslice 3x1x1 (DxWxH) convolution layer
        fuse3 = conv3d_fixed_pad(tensor, filters, [3, 1, 1], strides, dilation, data_format)
        fuse3 = tf.identity(fuse3, "interslice_4")
        fuse3 = batch_norm(fuse3, is_training, data_format)
        fuse3 = activation(fuse3)

        # first fusion layer with 2x upsampling (n_classes outputs)
        fuse1 = upsample_layer(fuse1, n_classes, ksize, [1, 2, 2], data_format)
        fuse1 = batch_norm(fuse1, is_training, data_format)
        fuse1 = activation(fuse1)

        # second fusion layer with 4x upsampling (n_classes * 2 outputs)
        fuse2 = upsample_layer(fuse2, n_classes*2, ksize, [1, 2, 2], data_format)
        fuse2 = batch_norm(fuse2, is_training, data_format)
        fuse2 = activation(fuse2)
        fuse2 = upsample_layer(fuse2, n_classes*2, ksize, [1, 2, 2], data_format)
        fuse2 = batch_norm(fuse2, is_training, data_format)
        fuse2 = activation(fuse2)

        # third fusion layer with 4x upsampling (n_classes * 4 outputs) [unsure why its x4 here]
        fuse3 = upsample_layer(fuse3, n_classes*4, ksize, [1, 2, 2], data_format)
        fuse3 = batch_norm(fuse3, is_training, data_format)
        fuse3 = activation(fuse3)
        fuse3 = upsample_layer(fuse3, n_classes*4, ksize, [1, 2, 2], data_format)
        fuse3 = batch_norm(fuse3, is_training, data_format)
        fuse3 = activation(fuse3)

        # concatenate fusion layers along channel dim "channels_first" for NCDHW or "channels_last" for NDHWC
        tensor = tf.concat([fuse1, fuse2, fuse3], name="concatenate", axis=-1 if data_format=="channels_last" else 1)

        # final layer (n_classes outputs)
        tensor = conv3d_fixed_pad(tensor, n_classes, ksize, strides, dilation, data_format)

        return tensor


# autoencoder residual unet
def resid_unet(tensor, is_training, n_blocks=17, ds=(2, 5, 9, 13),
           base_filters=64, k_size=(3,3), data_format="channels_last"):
    """
    Creates a ResNet Unet simimlar to https://arxiv.org/pdf/1704.07239.pdf
    using n=n_blocks simple resnet blocks and n=ds downsampling layers
    Args:
        tensor: a tensorflow tensor - the input data tensor
        is_training: a boolean - whether or not the model is training
        n_blocks: an int - the number of residual blocks to build into the network
        ds: a list/tuple of ints - the indices (starting from 0) of the downsampling blocks (i.e. where strides=2)
        base_filters: an int - the number of filters for the first layer of the network (auto scaled for deeper layers)
        k_size: an int - the size of the convolution kernel for the residual layers
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        logits: a tensorflow tensor - unscaled logits of size n_classes
    """

    # set variable scope
    with tf.variable_scope("resid_unet"):

        # set default values
        ksize = list(k_size) if isinstance(k_size, (tuple, list)) else [k_size] * 2
        filters = base_filters
        dilation = [1, 1]
        strides = [1, 1]

        # initial convolutional layer
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = tf.identity(tensor, "init_conv")
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)

        # encoder residual blocks
        for i in range(n_blocks):
            # if this is a downsampling block, strides=2 and filters is doubled, else strides=1
            if i in ds:
                strides = 2
                filters = 2 * filters
            else:
                strides = 1
            tensor = residual_layer(tensor, filters, ksize, strides, dilation, is_training, data_format)
            tensor = tf.identity(tensor, "encoder_block_layer_" + str(i).zfill(len(str(n_blocks * 2))))

        # decoder residual blocks
        filters = filters
        for i in range(n_blocks)[::-1]:
            # if this is a downsampling block, perform upsampling block, reduce nfilters
            if i in ds:
                strides = 2
                filters = filters / 2
                tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format)
                #tensor = upsample_residual_layer(tensor, filters, ksize, strides, dilation, is_training, data_format)
                tensor = tf.identity(tensor, "decoder_block_layer_" + str(i).zfill(len(str(n_blocks * 2))))
            else:
                strides = 1
                tensor = residual_layer(tensor, filters, ksize, strides, dilation, is_training, data_format)
                tensor = tf.identity(tensor, "decoder_block_layer_" + str(i).zfill(len(str(n_blocks * 2))))

        # final convolutional layer
        filters = base_filters
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = tf.identity(tensor, "final_conv")
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)

        return tensor


# autoencoder residual unet for regression
def res_unet_reg(tensor, is_training, base_filters, k_size, data_format):
    """
    Creates a ResNet Unet simimlar to https://arxiv.org/pdf/1704.07239.pdf
    using n=n_blocks simple resnet blocks and n=ds downsampling layers
    Args:
        tensor: a tensorflow tensor - the input data tensor
        is_training: a boolean - whether or not the model is training
        base_filters: an int - the number of filters for the first layer of the network (auto scaled for deeper layers)
        k_size: an int - the size of the convolution kernel for the residual layers
        data_format: a string - "channels_first" for NCHW data or "channels_last" for NHWC
    Returns:
        logits: a tensorflow tensor - unscaled logits of size n_classes
    """

    # set variable scope
    with tf.variable_scope("resid_unet"):
        # set default values
        ksize = list(k_size) if isinstance(k_size, (tuple, list)) else [k_size] * 2
        filters = base_filters
        dilation = [1, 1]
        strides = [1, 1]

        # initial convolutional layer
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = tf.identity(tensor, "init_conv_1")
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)

        # encoder residual block layer 1
        tensor = residual_layer(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "encoder_block_layer_1_1")

        # long range skip 1
        skip_1 = tf.identity(tensor, "skip_1")

        # downsample 1
        filters = filters * 2
        tensor = residual_layer(tensor, filters, ksize, [2, 2], dilation, is_training, data_format)
        tensor = tf.identity(tensor, "downsample_block_1")

        # encoder residual block layer 2
        tensor = residual_layer(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "encoder_block_layer_2_1")

        # long range skip 2
        skip_2 = tf.identity(tensor, "skip_2")

        # downsample 2
        filters = filters * 2
        tensor = residual_layer(tensor, filters, ksize, [2, 2], dilation, is_training, data_format)
        tensor = tf.identity(tensor, "downsample_block_2")

        # encoder residual block layer 3
        tensor = residual_layer(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "encoder_block_layer_3_1")

        # long range skip 3
        skip_3 = tf.identity(tensor, "skip_3")

        # downsample 3
        filters = filters * 2
        tensor = residual_layer(tensor, filters, ksize, [2, 2], dilation, is_training, data_format)
        tensor = tf.identity(tensor, "downsample_block_3")

        # encoder residual block layer 4
        tensor = residual_layer(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "encoder_block_layer_4_1")

        # upsample 3
        filters = filters / 2
        tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format)
        tensor = tf.identity(tensor, "upsample_3")

        # fuse 3
        #tensor = tf.concat([tensor, skip_3], name="concatenate_3", axis=-1 if data_format == "channels_last" else 1)
        tensor = tf.add(tensor, skip_3, name="fuse_3")

        # decoder residual block layer 3
        tensor = residual_layer(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "decoder_block_layer_3_1")

        # upsample 2
        filters = filters / 2
        tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format)
        tensor = tf.identity(tensor, "upsample_2")

        # fuse 2
        #tensor = tf.concat([tensor, skip_2], name="concatenate_2", axis=-1 if data_format == "channels_last" else 1)
        tensor = tf.add(tensor, skip_2, name="fuse_2")

        # decoder residual block layer 2
        tensor = residual_layer(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "decoder_block_layer_2_1")

        # upsample 1
        filters = filters / 2
        tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format)
        tensor = tf.identity(tensor, "upsample_1")

        # fuse 1
        #tensor = tf.concat([tensor, skip_1], name="concatenate_1", axis=-1 if data_format == "channels_last" else 1)
        tensor = tf.add(tensor, skip_1, name="fuse_1")

        # decoder residual block layer 1
        tensor = residual_layer(tensor, filters, ksize, strides, dilation, is_training, data_format)
        tensor = tf.identity(tensor, "decoder_block_layer_1_1")

        # 1x1 convolutional layer for output
        tensor = conv2d_fixed_pad(tensor, 1, [1, 1], strides, dilation, data_format)
        tensor = tf.identity(tensor, "final_conv_1")

        return tensor


def unet(tensor, is_training, base_filters, k_size, data_format):
    """
    Makes network like https://arxiv.org/pdf/1803.00131.pdf
    :param tensor:
    :param is_training:
    :param base_filters:
    :param k_size:
    :param data_format:
    :return:
    """

    # set variable scope
    with tf.variable_scope("unet"):
        # set default values
        ksize = list(k_size) if isinstance(k_size, (tuple, list)) else [k_size] * 2
        filters = base_filters
        dilation = [1, 1]
        strides = [1, 1]
        pool_size = [2, 2]
        pool_stride = [2, 2]

        # Convolution layers 1
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_1_1")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_1_2")

        # skip 1
        skip1 = tf.identity(tensor, "skip_1")

        # maxpool layer 1
        tensor = maxpool_layer_2d(tensor, ksize, pool_size, pool_stride, data_format)
        tensor = tf.identity(tensor, "maxpool_1")
        filters = base_filters * 2

        # Convolution layers 2
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_2_1")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_2_2")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_2_3")

        # skip 2
        skip2 = tf.identity(tensor, "skip_2")

        # maxpool layer 2
        tensor = maxpool_layer_2d(tensor, ksize, pool_size, pool_stride, data_format)
        tensor = tf.identity(tensor, "maxpool_2")
        filters = base_filters * 4

        # Convolution layers 3
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_3_1")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_3_2")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_3_3")

        # skip 3
        skip3 = tf.identity(tensor, "skip_3")

        # maxpool layer 3
        tensor = maxpool_layer_2d(tensor, ksize, pool_size, pool_stride, data_format)
        tensor = tf.identity(tensor, "maxpool_3")
        filters = base_filters * 8

        # Convolution layers 4
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_4_1")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_4_2")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_4_3")

        # skip 4
        skip4 = tf.identity(tensor, "skip_4")

        # maxpool layer 4
        tensor = maxpool_layer_2d(tensor, ksize, pool_size, pool_stride, data_format)
        tensor = tf.identity(tensor, "maxpool_4")
        filters = base_filters * 8

        # convolution layers 5 (bottom)
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_5_1")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_5_2")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_5_3")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_5_4")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_5_5")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_5_6")

        # deconvolution layer 1 and skip conncetion 4
        filters = base_filters * 8
        tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format)
        tensor = tf.add(tensor, skip4, name="skip_4")

        # Convolution layers 5
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_6_1")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_6_2")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_6_3")

        # deconvolution layer 2
        filters = base_filters * 4
        tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format)
        tensor = tf.add(tensor, skip3, name="skip_3")

        # Convolution layers 6
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_7_1")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_7_2")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_7_3")

        # deconvolution layer 3
        filters = base_filters * 2
        tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format)
        tensor = tf.add(tensor, skip2, name="skip_2")

        # Convolution layers 7
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_8_1")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_8_2")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_8_3")

        # deconvolution layer 3
        filters = base_filters
        tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format)
        tensor = tf.add(tensor, skip1, name="skip_1")

        # Convolution layers 8
        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_9_1")

        tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format)
        tensor = batch_norm(tensor, is_training, data_format)
        tensor = activation(tensor)
        tensor = tf.identity(tensor, "conv_9_2")

        # final convolutional layer
        tensor = conv2d_fixed_pad(tensor, 1, [1, 1], [1, 1], [1, 1], data_format)
        tensor = tf.identity(tensor, "final_conv_1")

        return tensor


def net_builder(features, params, is_training):
    """
    Builds the specified network.
    :param features: (tf.tensor) the features data
    :param params: (class: Params) the hyperparameters for the model
    :param is_training: (bool) whether or not the model is training
    :return: network - the specified network with the desired parameters
    """

    # determine network
    if params.model_name == "res_unet_reg":
        network = res_unet_reg(features, is_training, base_filters=params.base_filters, k_size=params.kernel_size,
                               data_format=params.data_format)
    elif params.model_name == "unet":
        network = unet(features, is_training, base_filters=params.base_filters, k_size=params.kernel_size,
                       data_format=params.data_format)
    else:
        sys.exit("Specified network does not exist: " + params.model_name)

    return network
