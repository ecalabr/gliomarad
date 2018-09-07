from net_layers import *


def unet(tensor, is_training, base_filters, k_size, data_format, reuse):
    """
    Makes network like https://arxiv.org/pdf/1803.00131.pdf
    :param tensor: (tf.tensor) the input tensor
    :param is_training: (bool) whether or not the model is training
    :param base_filters: (int) the number of base filters for the model
    :param k_size: (list/tuple(int)) the kernel size for convolution
    :param data_format: (str) the tensorflow data format: 'channels_last' or 'channels_first'
    :param reuse: (bool) whether or not to reuse layer weights
    :return: returns a tensorflow Unet model with the specified parameters.
    """

    # set default values
    ksize = list(k_size) if isinstance(k_size, (tuple, list)) else [k_size] * 2
    filters = base_filters
    dilation = [1, 1]
    strides = [1, 1]
    pool_size = [2, 2]
    pool_stride = [2, 2]
    act_type = 'leaky_relu'

    # Convolution layers 1
    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_1_1', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_1_1', reuse)
    tensor = activation(tensor, act_type, 'relu_1_1')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_1_2', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_1_2', reuse)
    tensor = activation(tensor, act_type, 'relu_1_2')

    # skip 1
    skip1 = tf.identity(tensor, "skip_1")

    # maxpool layer 1
    tensor = maxpool_layer_2d(tensor, pool_size, pool_stride, data_format, 'maxpool_1')
    filters = base_filters * 2

    # Convolution layers 2
    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_2_1', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_2_1', reuse)
    tensor = activation(tensor, act_type, 'relu_2_1')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_2_2', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_2_2', reuse)
    tensor = activation(tensor, act_type, 'relu_2_2')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_2_3', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_2_3', reuse)
    tensor = activation(tensor, act_type, 'relu_2_3')

    # skip 2
    skip2 = tf.identity(tensor, "skip_2")

    # maxpool layer 2
    tensor = maxpool_layer_2d(tensor, pool_size, pool_stride, data_format, 'maxpool_2')
    filters = base_filters * 4

    # Convolution layers 3
    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_3_1', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_3_1', reuse)
    tensor = activation(tensor, act_type, 'relu_3_1')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_3_2', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_3_2', reuse)
    tensor = activation(tensor, act_type, 'relu_3_2')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_3_3', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_3_3', reuse)
    tensor = activation(tensor, act_type, 'relu_3_3')

    # skip 3
    skip3 = tf.identity(tensor, "skip_3")

    # maxpool layer 3
    tensor = maxpool_layer_2d(tensor, pool_size, pool_stride, data_format, 'maxpool_3')
    filters = base_filters * 8

    # Convolution layers 4
    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_4_1', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_4_1', reuse)
    tensor = activation(tensor, act_type, 'relu_4_1')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_4_2', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_4_2', reuse)
    tensor = activation(tensor, act_type, 'relu_4_2')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_4_3', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_4_3', reuse)
    tensor = activation(tensor, act_type, 'relu_4_3')

    # skip 4
    skip4 = tf.identity(tensor, "skip_4")

    # maxpool layer 4
    tensor = maxpool_layer_2d(tensor, pool_size, pool_stride, data_format, 'maxpool_4')
    filters = base_filters * 8

    # convolution layers 5 (bottom)
    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_5_1', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_5_1', reuse)
    tensor = activation(tensor, act_type, 'relu_5_1')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_5_2', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_5_2', reuse)
    tensor = activation(tensor, act_type, 'relu_5_2')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_5_3', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_5_3', reuse)
    tensor = activation(tensor, act_type, 'relu_5_3')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_5_4', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_5_4', reuse)
    tensor = activation(tensor, act_type, 'relu_5_4')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_5_5', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_5_5', reuse)
    tensor = activation(tensor, act_type, 'relu_5_5')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_5_6', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_5_6', reuse)
    tensor = activation(tensor, act_type, 'relu_5_6')

    # deconvolution layer 1 and skip connection 4
    filters = base_filters * 8
    tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format, 'upsample_1', reuse)
    tensor = tf.add(tensor, skip4, name='fuse_4')

    # Convolution layers 6
    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_6_1', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_6_1', reuse)
    tensor = activation(tensor, act_type, 'relu_6_1')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_6_2', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_6_2', reuse)
    tensor = activation(tensor, act_type, 'relu_6_2')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_6_3', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_6_3', reuse)
    tensor = activation(tensor, act_type, 'relu_6_3')

    # deconvolution layer 2
    filters = base_filters * 4
    tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format, 'upsample_2', reuse)
    tensor = tf.add(tensor, skip3, name='fuse_3')

    # Convolution layers 7
    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_7_1', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_7_1', reuse)
    tensor = activation(tensor, act_type, 'relu_7_1')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_7_2', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_7_2', reuse)
    tensor = activation(tensor, act_type, 'relu_7_2')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_7_3', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_7_3', reuse)
    tensor = activation(tensor, act_type, 'relu_7_3')

    # deconvolution layer 3
    filters = base_filters * 2
    tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format, 'upsample_3', reuse)
    tensor = tf.add(tensor, skip2, name='fuse_2')

    # Convolution layers 8
    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_8_1', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_8_1', reuse)
    tensor = activation(tensor, act_type, 'relu_8_1')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_8_2', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_8_2', reuse)
    tensor = activation(tensor, act_type, 'relu_8_2')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_8_3', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_8_3', reuse)
    tensor = activation(tensor, act_type, 'relu_8_3')

    # deconvolution layer 3
    filters = base_filters
    tensor = upsample_layer(tensor, filters, ksize, [2, 2], data_format, 'upsample_4', reuse)
    tensor = tf.add(tensor, skip1, name='fuse_1')

    # Convolution layers 9
    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_9_1', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_9_1', reuse)
    tensor = activation(tensor, act_type, 'relu_9_1')

    tensor = conv2d_fixed_pad(tensor, filters, ksize, strides, dilation, data_format, 'conv_9_2', reuse)
    tensor = batch_norm(tensor, is_training, data_format, 'batch_norm_9_2', reuse)
    tensor = activation(tensor, act_type, 'relu_9_2')

    # final convolutional layer
    tensor = conv2d_fixed_pad(tensor, 1, [1, 1], [1, 1], [1, 1], data_format, 'final_conv_1', reuse)

    return tensor


def custom_unet(features, params, is_training, reuse=False):
    """
    Makes a deep unet with long range skip connections similar to:
    https://arxiv.org/pdf/1803.00131.pdf
    :param features: (tf.tensor) the input features
    :param params: (class Params()) the parameters for the model
    :param is_training: (bool) whether or not the model is training
    :param reuse: (bool) whether or not to reuse layer weights (mostly for eval and infer modes)
    :return: A deep unet with the specified parameters
    """

    # define fixed params
    layer_layout = params.layer_layout
    filt = params.base_filters
    dfmt = params.data_format
    # dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    strides = [1, 1]
    dil = [1, 1]

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # initial input convolution layer
    tensor = conv2d_fixed_pad(features, filt, ksize, [1, 1], [1, 1], dfmt, 'init_conv', reuse)
    tensor = batch_norm(tensor, is_training, dfmt, 'init_bn', reuse)
    tensor = activation(tensor,act, 'init_act')

    # unet encoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):
        # horizontal blocks
        for layer in range(n_layers):
            layer_name = 'enc_conv_' + str(n) + '_' + str(layer)
            tensor = conv2d_fixed_pad(tensor, filt, ksize, strides, dil, dfmt, layer_name, reuse)
            tensor = batch_norm(tensor, is_training, dfmt, 'enc_conv_bn_' + str(n) + '_' + str(layer), reuse)
            tensor = activation(tensor, act, 'enc_conv_act_' + str(n) + '_' + str(layer))
        # create skip connection
        layer_name = 'skip_' + str(n)
        skips.append(tf.identity(tensor, name=layer_name))
        # downsample block
        filt = filt * 2  # double filters before downsampling
        layer_name = 'enc_conv_downsample_' + str(n)
        tensor = conv2d_fixed_pad(tensor, filt, ksize, [2, 2], dil, dfmt, layer_name, reuse)
        tensor = batch_norm(tensor, is_training, dfmt, 'enc_conv_downsample_bn_' + str(n), reuse)
        tensor = activation(tensor, act, 'enc_conv_downsample_act_' + str(n))

    # unet horizontal (bottom) bottleneck blocks
    for layer in range(horz_layers):
        layer_name = 'horz_conv_' + str(layer)
        tensor = conv2d_fixed_pad(tensor, filt, ksize, strides, dil, dfmt, layer_name, reuse)
        tensor = batch_norm(tensor, is_training, dfmt, 'horz_conv_bn_' + str(layer), reuse)
        tensor = activation(tensor, act, 'horz_conv_act_' + str(layer))

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):
        # upsample block
        filt = filt / 2  # half filters before upsampling
        layer_name = 'dec_conv_upsample' + str(n)
        tensor = upsample_layer(tensor, filt, ksize, [2, 2], dfmt, layer_name, reuse)
        tensor = batch_norm(tensor, is_training, dfmt, 'dec_conv_upsample_bn_' + str(n), reuse)
        tensor = activation(tensor, act, 'dec_conv_upsample_act_' + str(n))
        # fuse skip connections with concatenation of features
        layer_name = 'skip_' + str(n)
        axis = 1 if dfmt == 'channels_first' else -1
        tensor = tf.concat([tensor, skips[n]], axis, name=layer_name)
        # tensor = tf.add(tensor, skips[n], name=layer_name)
        # horizontal blocks
        for layer in range(n_layers):
            layer_name = 'conv_dec_blk_' + str(n) + '_' + str(layer)
            tensor = conv2d_fixed_pad(tensor, filt, ksize, strides, dil, dfmt, layer_name, reuse)
            tensor = batch_norm(tensor, is_training, dfmt, 'dec_conv_bn_' + str(n) + '_' + str(layer), reuse)
            tensor = activation(tensor, act, 'dec_conv_act_' + str(n) + '_' + str(layer))

    # output layer
    tensor = conv2d_fixed_pad(tensor, 1, [1, 1], [1, 1], [1, 1], dfmt, 'final_conv', reuse)

    return tensor


def bneck_resunet(features, params, is_training, reuse=False):
    """
    Makes a deep bottleneck residual unet with long range skip connections similar to:
    https://arxiv.org/pdf/1704.07239.pdf
    https://vitalab.github.io/deep-learning/2017/05/08/resunet.html
    :param features: (tf.tensor) the input features
    :param params: (class Params()) the parameters for the model
    :param is_training: (bool) whether or not the model is training
    :param reuse: (bool) whether or not to reuse layer weights (mostly for eval and infer modes)
    :return: A deep bottleneck residual unet with the specified parameters
    """

    # define fixed params
    layer_layout = params.layer_layout
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # initial input convolution layer
    tensor = conv2d_fixed_pad(features, filt, ksize, [1, 1], [1, 1], dfmt, 'init_conv', reuse)

    # unet encoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):
        # horizontal blocks
        for layer in range(n_layers):
            layer_name = 'bneck_enc_blk_' + str(n) + '_' + str(layer)
            tensor = bneck_res_layer(tensor, ksize, filt, 0, dropout, is_training, dfmt, act, layer_name, reuse)
        # create skip connection
        layer_name = 'skip_' + str(n)
        skips.append(tf.identity(tensor, name=layer_name))
        # downsample block
        filt = filt * 2  # double filters before downsampling
        layer_name = 'bneck_downsample_' + str(n)
        tensor = bneck_res_layer(tensor, ksize, filt, 1, dropout, is_training, dfmt, act, layer_name, reuse)

    # unet horizontal (bottom) bottleneck blocks
    for layer in range(horz_layers):
        layer_name = 'bneck_horz_' + str(layer)
        tensor = bneck_res_layer(tensor, ksize, filt, 0, dropout, is_training, dfmt, act, layer_name, reuse)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):
        # upsample block
        filt = filt / 2  # half filters before upsampling
        layer_name = 'bneck_upsample_' + str(n)
        tensor = bneck_res_layer(tensor, ksize, filt, 2, dropout, is_training, dfmt, act, layer_name, reuse)
        # fuse skip connections
        layer_name = 'skip_' + str(n)
        tensor = tf.add(tensor, skips[n], name=layer_name)
        # horizontal blocks
        for layer in range(n_layers):
            layer_name = 'bneck_dec_blk_' + str(n) + '_' + str(layer)
            tensor = bneck_res_layer(tensor, ksize, filt, 0, dropout, is_training, dfmt, act, layer_name, reuse)

    # output layer
    tensor = conv2d_fixed_pad(tensor, 1, [1, 1], [1, 1], [1, 1], dfmt, 'final_conv', reuse)

    return tensor


def deep_embed_net(features, params, is_training, reuse=False):
    """
    Creates a deep embedding CNN similar to:
    https://www.medicalimageanalysisjournal.com/article/S1361-8415(18)30125-7/fulltext
    :param features: (tf.tensor) the input features
    :param params: (class Params()) the parameters for the model
    :param is_training: (bool) whether or not the model is training
    :param reuse: (bool) whether or not to reuse layer weights (mostly for eval and infer modes)
    :return: A deep embedding CNN with the specified parameters
    """

    # initial convolutional layer
    tensor = features

    return tensor


def net_builder(features, params, is_training, reuse=False):
    """
    Builds the specified network.
    :param features: (tf.tensor) the features data
    :param params: (class: Params) the hyperparameters for the model
    :param is_training: (bool) whether or not the model is training
    :param reuse (bool) whether or not to reuse weights for layers
    :return: network - the specified network with the desired parameters
    """

    # sanity checks
    if not isinstance(is_training, bool): raise ValueError("Parameter is_training must be a boolean")

    # determine network
    if params.model_name == 'unet':
        network = unet(features, is_training, base_filters=params.base_filters, k_size=params.kernel_size,
                       data_format=params.data_format, reuse=reuse)
    elif params.model_name == 'bneck_resunet':
        network = bneck_resunet(features, params, is_training, reuse)
    elif params.model_name == 'custom_unet':
        network = custom_unet(features, params, is_training,reuse)
    else:
        raise ValueError("Specified network does not exist: " + params.model_name)

    return network
