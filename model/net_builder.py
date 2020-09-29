import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Conv2D, Conv2DTranspose, Input, MaxPool3D
from tensorflow.keras.models import Model
from model.net_layers import bneck_resid3d, bneck_resid2d, conv3d_act_bn
from tensorflow.keras.mixed_precision import experimental as mixed_precision


# get built in locals
start_globals = list(globals().keys())


# simple 3D unet with max pooling downsample, conv transpose upsample, long range concat skips, batch norm, dropout
def unet_3d(params):

    # define fixed params
    layer_layout = params.layer_layout
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    chan = len(params.data_prefix)
    train_dims = params.train_dims + [chan] if dfmt == 'channels_last' else [chan] + params.train_dims
    batch_size = params.batch_size
    output_filt = params.output_filters
    policy = params.policy

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # input layer
    inputs = Input(shape=train_dims, batch_size=batch_size, dtype='float32')

    # initial convolution layer
    x = Conv3D(filt, ksize, padding='same', data_format=dfmt, dtype=policy)(inputs)

    # unet encoder limb
    for n, n_layers in enumerate(unet_layout):
        # horizontal layers
        for layer in range(n_layers):
            # residual blocks with activation and batch norm
            x = conv3d_act_bn(x, filt, ksize, dfmt, dropout, act, policy=policy)

        # create skip connection
        skips.append(tf.identity(x))

        # downsample block, double filters
        filt = filt * 2  # double filters before downsampling
        x = MaxPool3D(pool_size=[2, 2, 2], padding='same', data_format=dfmt)(x)

    # unet horizontal (bottom) bottleneck blocks
    for layer in range(horz_layers):
        x = conv3d_act_bn(x, filt, ksize, dfmt, dropout, act, policy=policy)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):

        # upsample block
        filt = int(round(filt/2))  # half filters before upsampling
        x = Conv3DTranspose(filt, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt, dtype=policy)(x)

        # fuse skip connections with concatenation of features
        x = tf.concat([x, skips[n]], axis=-1 if dfmt == 'channels_last' else 1)

        # horizontal blocks
        for layer in range(n_layers):
            x = conv3d_act_bn(x, filt, ksize, dfmt, dropout, act, policy=policy)

    # output layer
    if params.final_layer == "conv":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
    elif params.final_layer == "sigmoid":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.sigmoid(x)
    elif params.final_layer == "softmax":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


# 3D unet with bottleneck residual blocks, conv downsample, conv transpose upsample
# long range concat skips, batch norm, dropout
def unet_3d_bneck(params):

    # define fixed params
    layer_layout = params.layer_layout
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    chan = len(params.data_prefix)
    train_dims = params.train_dims + [chan] if dfmt == 'channels_last' else [chan] + params.train_dims
    batch_size = params.batch_size
    output_filt = params.output_filters
    policy = params.policy

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # input layer
    inputs = Input(shape=train_dims, batch_size=batch_size, dtype='float32')

    # initial convolution layer
    x = Conv3D(filt, ksize, padding='same', data_format=dfmt, dtype=policy)(inputs)

    # unet encoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):
        # horizontal layers
        for layer in range(n_layers):
            # residual blocks with activation and batch norm
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy=policy)

        # create skip connection
        skips.append(tf.identity(x))

        # downsample block
        filt = filt * 2  # double filters before downsampling
        x = Conv3D(filt, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt, dtype=policy)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        # x = activation_layer(act)(x)

    # unet horizontal (bottom) bottleneck blocks
    for layer in range(horz_layers):
        x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy=policy)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):

        # upsample block
        filt = int(round(filt/2))  # half filters before upsampling
        x = Conv3DTranspose(filt, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt, dtype=policy)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        # x = activation_layer(act)(x)

        # fuse skip connections with concatenation of features
        x = tf.concat([x, skips[n]], axis=-1 if dfmt == 'channels_last' else 1)

        # horizontal blocks
        for layer in range(n_layers):
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy=policy)

    # output layer - no mixed precision data policy
    if params.final_layer == "conv":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
    elif params.final_layer == "sigmoid":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.sigmoid(x)
    elif params.final_layer == "softmax":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


# 3D unet with bottleneck residual blocks, conv downsample, conv transpose upsample
# long range concat skips, batch norm, dropout
def unet_3d_bneck2(params):

    # define fixed params
    layer_layout = params.layer_layout
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    chan = len(params.data_prefix)
    train_dims = params.train_dims + [chan] if dfmt == 'channels_last' else [chan] + params.train_dims
    batch_size = params.batch_size
    output_filt = params.output_filters
    policy = params.policy

    # additional setup for network construction
    skips = []

    # input layer
    inputs = Input(shape=train_dims, batch_size=batch_size, dtype='float32')
    x = inputs

    # unet encoder limb with residual bottleneck blocks
    for level in layer_layout:
        for block in range(level): # remove one for first layer that takes inputs
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy=policy)

        # if not the bottom level, create skip connection and then downsample
        if not level == layer_layout[-1]:
            # create long range skip
            skips.append(x)

            # downsample block
            filt = filt * 2  # double filters before strided conv downsampling
            x = Conv3D(filt, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt, dtype=policy)(x)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    layer_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, level in enumerate(layer_layout[1:]):  # bottom level of layer layout is not repeated

        # upsample block
        filt = int(round(filt/2))  # half filters before upsampling
        x = Conv3DTranspose(filt, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt, dtype=policy)(x)

        # fuse skip connections with concatenation of features
        x = tf.concat([x, skips[n]], axis=-1 if dfmt == 'channels_last' else 1)

        # horizontal blocks
        for block in range(level):
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy=policy)

    # output layer - no mixed precision data policy
    if params.final_layer == "conv":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
    elif params.final_layer == "sigmoid":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.sigmoid(x)
    elif params.final_layer == "softmax":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


# 3D unet with bottleneck residual blocks, conv downsample, conv transpose upsample
# long range concat skips, batch norm, dropout
def unet_3d_bneck_dilated(params):

    # define fixed params
    layer_layout = params.layer_layout
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    chan = len(params.data_prefix)
    train_dims = params.train_dims + [chan] if dfmt == 'channels_last' else [chan] + params.train_dims
    batch_size = params.batch_size
    output_filt = params.output_filters
    policy = params.policy

    # additional setup for network construction
    skips = []

    # input layer
    inputs = Input(shape=train_dims, batch_size=batch_size, dtype='float32')
    x = inputs

    # unet encoder limb with residual bottleneck blocks
    for level in layer_layout[:-1]:  # bottom level is reserved for dilated convs
        for block in range(level): # remove one for first layer that takes inputs
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy=policy)

        # create long range skip
        skips.append(x)

        # downsample block
        filt = filt * 2  # double filters before strided conv downsampling
        x = Conv3D(filt, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt, dtype=policy)(x)

    # dilated bottom level
    dil_out = []
    for dil_block in range(layer_layout[-1]):
        # dilation sequence is 1, 2, 4, 6, 8 etc
        if dil_block == 0:
            dil = [1] * 3
        else:
            dil = [2 * dil_block] * 3
        # dilated blocks
        x = Conv3D(filt, ksize, dilation_rate=dil, padding='same', data_format=dfmt, dtype=policy)(x)
        # concatenate all outputs
        dil_out.append(x)
    # concatenate outputs
    x = tf.concat(dil_out, axis=-1 if dfmt == 'channels_last' else 1)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    layer_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, level in enumerate(layer_layout[1:]):  # bottom level of layer layout is not repeated

        # upsample block
        filt = int(round(filt/2))  # half filters before upsampling
        x = Conv3DTranspose(filt, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt, dtype=policy)(x)

        # fuse skip connections with concatenation of features
        x = tf.concat([x, skips[n]], axis=-1 if dfmt == 'channels_last' else 1)

        # horizontal blocks
        for block in range(level):
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy=policy)

    # output layer - no mixed precision data policy
    if params.final_layer == "conv":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
    elif params.final_layer == "sigmoid":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.sigmoid(x)
    elif params.final_layer == "softmax":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


# 2.5D unet with bottleneck residual blocks, conv downsample, conv transpose upsample
# long range concat skips, batch norm, dropout
def unet_25d_bneck(params):

    # define fixed params
    layer_layout = params.layer_layout
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    chan = len(params.data_prefix)
    train_dims = params.train_dims + [chan] if dfmt == 'channels_last' else [chan] + params.train_dims
    batch_size = params.batch_size
    output_filt = params.output_filters
    policy = params.policy

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # input layer
    inputs = Input(shape=train_dims, batch_size=batch_size, dtype='float32')

    # initial convolution layer
    x = Conv3D(filt, ksize, padding='same', data_format=dfmt, dtype=policy)(inputs)

    # unet encoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):
        # horizontal layers
        for layer in range(n_layers):
            # residual blocks with activation and batch norm
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy=policy)

        # do z dimension conv
        x = bneck_resid3d(x, filt, [1, 1, 3], dfmt, dropout, act, policy=policy)

        # create skip connection
        skips.append(tf.identity(x))

        # downsample block
        filt = filt * 2  # double filters before downsampling
        x = Conv3D(filt, ksize, strides=[2, 2, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # unet horizontal (bottom) bottleneck blocks
    for layer in range(horz_layers):
        x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy=policy)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):

        # upsample block
        filt = int(round(filt/2))  # half filters before upsampling
        x = Conv3DTranspose(filt, ksize, strides=[2, 2, 1], padding='same', data_format=dfmt, dtype=policy)(x)

        # do z dimension conv
        x = bneck_resid3d(x, filt, [1, 1, 3], dfmt, dropout, act, policy=policy)

        # fuse skip connections with concatenation of features
        x = tf.concat([x, skips[n]], axis=-1 if dfmt == 'channels_last' else 1)

        # horizontal blocks
        for layer in range(n_layers):
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy=policy)

    # output layer
    if params.final_layer == "conv":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
    elif params.final_layer == "sigmoid":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.sigmoid(x)
    elif params.final_layer == "softmax":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


# 2D unet with bottleneck residual blocks, conv downsample, conv transpose upsample
# long range concat skips, batch norm, dropout
def unet_2d_bneck(params):

    # define fixed params
    layer_layout = params.layer_layout
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    act = params.activation
    chan = len(params.data_prefix)
    train_dims = params.train_dims + [chan] if dfmt == 'channels_last' else [chan] + params.train_dims
    batch_size = params.batch_size
    output_filt = params.output_filters
    policy = params.policy

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # input layer
    inputs = Input(shape=train_dims, batch_size=batch_size, dtype='float32')

    # initial convolution layer
    x = Conv2D(filt, ksize, padding='same', data_format=dfmt, dtype=policy)(inputs)

    # unet encoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):
        # horizontal blocks
        for layer in range(n_layers):
            # residual blocks with activation and batch norm
            x = bneck_resid2d(x, filt, ksize, dfmt, dropout, act, policy=policy)

        # create skip connection
        skips.append(tf.identity(x))

        # downsample block
        filt = filt * 2  # double filters before downsampling
        x = Conv2D(filt, ksize, strides=[2, 2], padding='same', data_format=dfmt, dtype=policy)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        # x = activation_layer(act)(x)

    # bottom bottleneck blocks
    for layer in range(horz_layers):
        x = bneck_resid2d(x, filt, ksize, dfmt, dropout, act, policy=policy)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):

        # upsample block
        filt = int(round(filt/2))  # half filters before upsampling
        x = Conv2DTranspose(filt, ksize, strides=[2, 2], padding='same', data_format=dfmt, dtype=policy)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        # x = activation_layer(act)(x)

        # fuse skip connections with concatenation of features
        x = tf.add(x, skips[n])

        # horizontal blocks
        for layer in range(n_layers):
            x = bneck_resid2d(x, filt, ksize, dfmt, dropout, act, policy=policy)

    # output layer
    if params.final_layer == "conv":
        x = Conv2D(filters=output_filt, kernel_size=[1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
    elif params.final_layer == "sigmoid":
        x = Conv2D(filters=output_filt, kernel_size=[1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.sigmoid(x)
    elif params.final_layer == "softmax":
        x = Conv2D(filters=output_filt, kernel_size=[1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


# Wrapper function
def net_builder(params):
    # set up mixed precision computation to take advantage of Nvidia tensor cores
    # https://www.tensorflow.org/guide/mixed_precision
    if params.mixed_precision:  # enable mixed precision and warn user
        print("WARNING: using tensorflow mixed precision... This could lead to numeric instability in some cases.")
        policy = mixed_precision.Policy('mixed_float16')
        # warn if batch size and/or nfilters is not a multpile of 8
        if not params.base_filters % 8 == 0:
            print("WARNING: parameter base_filters is not a multiple of 8, which will not use tensor cores.")
        if not params.batch_size % 8 == 0:
            print("WARNING: parameter batch_size is not a multiple of 8, which will not use tensor cores.")
    else:  # if not using mixed precision, then assume float32
        policy = mixed_precision.Policy('float32')

    # put current policy in params for use in model construction
    params.policy = policy

    # set default policy, subsequent per layer dtype can be specified
    mixed_precision.set_policy(policy)  # default policy for layers

    # determine network
    if params.model_name in globals():
        model = globals()[params.model_name](params)
    else:
        methods = [k for k in globals().keys() if k not in start_globals]
        raise NotImplementedError(
            "Specified model type: '{}' is not one of the available types: {}".format(params.model_name, methods))

    return model
