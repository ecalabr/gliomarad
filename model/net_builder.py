from net_layers import *
import tensorflow as tf


def custom_unet_maxpool(features, params, is_training, reuse=False):
    """
    Makes a deep unet with long range skip connections similar to:
    https://arxiv.org/pdf/1803.00131.pdf
    with maxpool downsampling and bilinear upsampling
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
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    strides = [1, 1]
    dil = [1, 1]

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # initial input convolution layer
    layer_name = 'initial_layer'
    tensor = conv2d_block(features, filt, ksize, strides, dil, dfmt, layer_name, reuse, dropout, is_training, act)

    # unet encoder
    for n, n_layers in enumerate(unet_layout):
        # horizontal limb
        for layer in range(n_layers):
            layer_name = 'encoder_block_' + str(n) + '_' + str(layer)
            tensor = conv2d_block(tensor, filt, ksize, strides, dil, dfmt, layer_name, reuse, dropout, is_training, act)

        # create skip connection
        layer_name = 'skip_' + str(n)
        skips.append(tf.identity(tensor, name=layer_name))

        # downsample layer
        filt = filt * 2  # double filters before downsampling
        layer_name = 'encoder_downsample_' + str(n)
        tensor = maxpool2d_layer(tensor, [2, 2], [2, 2], dfmt, layer_name)

    # unet horizontal (bottom) limb
    for layer in range(horz_layers):
        layer_name = 'bottom_block_' + str(layer)
        tensor = conv2d_block(tensor, filt, ksize, strides, dil, dfmt, layer_name, reuse, dropout, is_training, act)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder
    for n, n_layers in enumerate(unet_layout):
        # upsample block
        filt = filt / 2  # half filters before upsampling
        layer_name = 'decoder_upsample_' + str(n)
        # tensor = deconv2d_block(tensor, filt, ksize, [2, 2], dfmt, layer_name, reuse, dropout, is_training, act)
        tensor = upsample2d_layer(tensor, 2, dfmt, layer_name)

        # fuse skip connections with concatenation of features
        layer_name = 'skip_' + str(n)
        axis = 1 if dfmt == 'channels_first' else -1
        tensor = tf.concat([tensor, skips[n]], axis, name=layer_name)

        # horizontal limb
        for layer in range(n_layers):
            layer_name = 'decoder_block_' + str(n) + '_' + str(layer)
            tensor = conv2d_block(tensor, filt, ksize, strides, dil, dfmt, layer_name, reuse, dropout, is_training, act)

    # output layer
    tensor = conv2d_fixed_pad(tensor, 1, [1, 1], [1, 1], [1, 1], dfmt, 'output_layer_conv', reuse)

    return tensor


def custom_unet_fcnn(features, params, is_training, reuse=False):
    """
    Makes a deep unet with long range skip connections similar to:
    https://arxiv.org/pdf/1803.00131.pdf
    with fully convolutional upsampling and downsampling
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
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    strides = [1, 1]
    dil = [1, 1]

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # initial input convolution layer
    layer_name = 'initial_layer'
    tensor = conv2d_block(features, filt, ksize, strides, dil, dfmt, layer_name, reuse, dropout, is_training, act)

    # unet encoder
    for n, n_layers in enumerate(unet_layout):
        # horizontal limb
        for layer in range(n_layers):
            layer_name = 'encoder_block_' + str(n) + '_' + str(layer)
            tensor = conv2d_block(tensor, filt, ksize, strides, dil, dfmt, layer_name, reuse, dropout, is_training, act)

        # create skip connection
        layer_name = 'skip_' + str(n)
        skips.append(tf.identity(tensor, name=layer_name))

        # downsample layer
        filt = filt * 2  # double filters before downsampling
        layer_name = 'encoder_downsample_' + str(n)
        tensor = conv2d_block(tensor, filt, ksize, [2, 2], dil, dfmt, layer_name, reuse, dropout, is_training, act)

    # unet horizontal (bottom) limb
    for layer in range(horz_layers):
        layer_name = 'bottom_block_' + str(layer)
        tensor = conv2d_block(tensor, filt, ksize, strides, dil, dfmt, layer_name, reuse, dropout, is_training, act)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder
    for n, n_layers in enumerate(unet_layout):
        # upsample block
        filt = filt / 2  # half filters before upsampling
        layer_name = 'decoder_upsample_' + str(n)
        tensor = deconv2d_block(tensor, filt, ksize, [2, 2], dfmt, layer_name, reuse, dropout, is_training, act)

        # fuse skip connections with concatenation of features
        layer_name = 'skip_' + str(n)
        axis = 1 if dfmt == 'channels_first' else -1
        tensor = tf.concat([tensor, skips[n]], axis, name=layer_name)
        # tensor = tf.add(tensor, skips[n], name=layer_name)

        # horizontal limb
        for layer in range(n_layers):
            layer_name = 'decoder_block_' + str(n) + '_' + str(layer)
            tensor = conv2d_block(tensor, filt, ksize, strides, dil, dfmt, layer_name, reuse, dropout, is_training, act)

    # output layer
    tensor = conv2d_fixed_pad(tensor, 1, [1, 1], [1, 1], [1, 1], dfmt, 'output_layer_conv', reuse)

    return tensor


def custom_resid_unet(features, params, is_training, reuse=False):
    """
    Makes a deep unet with long range skip connections similar to the below link except with residual blocks
    in the horizontal limbs
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
    dpout = params.dropout_rate
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
            if layer < n_layers -1:
                layer_name = 'enc_bneck_resid_' + str(n) + '_' + str(layer)
                tensor = bneck_res_layer(tensor, ksize, filt, 0, dpout, is_training, dfmt, act, layer_name, reuse)
            else:
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
        # if 0 < layer < horz_layers - 1:
        layer_name = 'horz_bneck_resid_' + str(layer)
        tensor = bneck_res_layer(tensor, ksize, filt, 0, dpout, is_training, dfmt, act, layer_name, reuse)
        # else:
        #    layer_name = 'horz_conv_' + str(layer)
        #    tensor = conv2d_fixed_pad(tensor, filt, ksize, strides, dil, dfmt, layer_name, reuse)
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
        tensor = deconv2d_layer(tensor, filt, ksize, [2, 2], dfmt, layer_name, reuse)
        tensor = batch_norm(tensor, is_training, dfmt, 'dec_conv_upsample_bn_' + str(n), reuse)
        tensor = activation(tensor, act, 'dec_conv_upsample_act_' + str(n))
        # fuse skip connections with concatenation of features
        layer_name = 'skip_' + str(n)
        axis = 1 if dfmt == 'channels_first' else -1
        tensor = tf.concat([tensor, skips[n]], axis, name=layer_name)
        # tensor = tf.add(tensor, skips[n], name=layer_name)
        # horizontal blocks
        for layer in range(n_layers):
            if 0 < layer:
                layer_name = 'dec_bneck_resid_' + str(n) + '_' + str(layer)
                tensor = bneck_res_layer(tensor, ksize, filt, 0, dpout, is_training, dfmt, act, layer_name, reuse)
            else:
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
        layer_name = 'fuse_skip_' + str(n)
        # axis = 1 if dfmt == 'channels_first' else -1
        # tensor = tf.concat([tensor, skips[n]], axis, name=layer_name)
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

    # define fixed params
    layer_layout = params.layer_layout
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    strides = [1, 1]
    dil = [1, 1]

    # first 5 simple convolutional layers
    tensor = features
    for i in range(5):
        name = 'initial_conv_' + str(i)
        tensor = conv2d_block(tensor, filt, ksize, strides, dil, dfmt, name, reuse, dropout, is_training, act)

    # embedding blocks
    recons = None
    for i in range(layer_layout[0]):
        name = 'embedding_block_' + str(i)
        tensor, recon = embedding_block(tensor,ksize, filt, dropout, is_training, dfmt, act, name, reuse)
        if i == 0:  # handles first loop where recons has not been defined yet
            recons = recon  # tf.expand_dims(recon, axis=-1)
        else:  # handles subsequent loops
            recons = tf.concat([recons, recon], axis=-1, name=name + '_embed_concat')

     # final output layer
    tensor = conv2d_fixed_pad(tensor, 1, [1, 1], [1, 1], [1, 1], dfmt, 'final_conv', reuse)

    # concatenate all recons and return as one tensor, which will be # of embedding blocks + 1 different predicitons
    tensor = tf.concat([tensor, recons], axis=-1, name='final_embed_concat')

    return tensor


def deep_embed_resnet(features, params, is_training, reuse=False):
    """
    Creates a deep embedding CNN similar to:
    https://www.medicalimageanalysisjournal.com/article/S1361-8415(18)30125-7/fulltext
    :param features: (tf.tensor) the input features
    :param params: (class Params()) the parameters for the model
    :param is_training: (bool) whether or not the model is training
    :param reuse: (bool) whether or not to reuse layer weights (mostly for eval and infer modes)
    :return: A deep embedding CNN with the specified parameters
    """

    # define fixed params
    layer_layout = params.layer_layout
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    strides = [1, 1]
    dil = [1, 1]

    # initial conv layer block
    tensor = features
    name = 'initial_3x3_conv'
    tensor = conv2d_block(tensor, filt, ksize, strides, dil, dfmt, name, reuse, dropout, is_training, act)

    # first 5 residual layers
    for i in range(5):
        name = 'res_block_' + str(i)
        tensor = resid2d_layer(tensor, filt, ksize, strides, dil, dfmt, name, reuse, dropout, is_training, act)

    # embedding blocks
    recons = None
    for i in range(layer_layout[0]):
        name = 'embedding_block_' + str(i)
        tensor, recon = embedding_block(tensor,ksize, filt, dropout, is_training, dfmt, act, name, reuse)
        if i == 0:  # handles first loop where recons has not been defined yet
            recons = recon  # tf.expand_dims(recon, axis=-1)
        else:  # handles subsequent loops
            recons = tf.concat([recons, recon], axis=-1, name=name + '_embed_concat')

     # final output layer
    tensor = conv2d_fixed_pad(tensor, 1, [1, 1], [1, 1], [1, 1], dfmt, 'final_conv', reuse)

    # concatenate all recons and return as one tensor, which will be # of embedding blocks + 1 different predicitons
    tensor = tf.concat([tensor, recons], axis=-1, name='final_embed_concat')

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
    if params.model_name == 'custom_unet_fcnn':
        network = custom_unet_fcnn(features, params, is_training, reuse)
    elif params.model_name == 'bneck_resunet':
        network = bneck_resunet(features, params, is_training, reuse)
    elif params.model_name == 'custom_unet_maxpool':
        network = custom_unet_maxpool(features, params, is_training,reuse)
    elif params.model_name == 'custom_resid_unet':
        network = custom_resid_unet(features, params, is_training,reuse)
    elif params.model_name == 'deep_embed_net':
        network = deep_embed_net(features, params, is_training)
    elif params.model_name == 'deep_embed_resnet':
        network = deep_embed_resnet(features, params, is_training)
    else:
        raise ValueError("Specified network does not exist: " + params.model_name)

    return network
