import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from model.net_layers import bneck_resid3d, bneck_resid2d


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

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # input layer
    inputs = Input(shape=train_dims, batch_size=batch_size)

    # initial convolution layer
    x = Conv3D(filt, ksize, padding='same', data_format=dfmt)(inputs)

    # unet encoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):
        # horizontal layers
        for layer in range(n_layers):
            # residual blocks with activation and batch norm
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act)

        # create skip connection
        skips.append(tf.identity(x))

        # downsample block
        filt = filt * 2  # double filters before downsampling
        x = Conv3D(filt, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        # x = activation_layer(act)(x)

    # unet horizontal (bottom) bottleneck blocks
    for layer in range(horz_layers):
        x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):

        # upsample block
        filt = int(round(filt/2))  # half filters before upsampling
        x = Conv3DTranspose(filt, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        # x = activation_layer(act)(x)

        # fuse skip connections with concatenation of features
        x = tf.add(x, skips[n])

        # horizontal blocks
        for layer in range(n_layers):
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act)

    # output layer
    if params.final_layer == "conv":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt)(x)
    elif params.final_layer == "sigmoid":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.sigmoid(x)
    elif params.final_layer == "softmax":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


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

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # input layer
    inputs = Input(shape=train_dims, batch_size=batch_size)

    # initial convolution layer
    x = Conv2D(filt, ksize, padding='same', data_format=dfmt)(inputs)

    # unet encoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):
        # horizontal blocks
        for layer in range(n_layers):
            # residual blocks with activation and batch norm
            x = bneck_resid2d(x, filt, ksize, dfmt, dropout, act)

        # create skip connection
        skips.append(tf.identity(x))

        # downsample block
        filt = filt * 2  # double filters before downsampling
        x = Conv2D(filt, ksize, strides=[2, 2], padding='same', data_format=dfmt)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        # x = activation_layer(act)(x)

    # bottom bottleneck blocks
    for layer in range(horz_layers):
        x = bneck_resid2d(x, filt, ksize, dfmt, dropout, act)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):

        # upsample block
        filt = int(round(filt/2))  # half filters before upsampling
        x = Conv2DTranspose(filt, ksize, strides=[2, 2], padding='same', data_format=dfmt)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        # x = activation_layer(act)(x)

        # fuse skip connections with concatenation of features
        x = tf.add(x, skips[n])

        # horizontal blocks
        for layer in range(n_layers):
            x = bneck_resid2d(x, filt, ksize, dfmt, dropout, act)

    # output layer
    if params.final_layer == "conv":
        x = Conv2D(filters=output_filt, kernel_size=[1, 1], padding='same', data_format=dfmt)(x)
    elif params.final_layer == "sigmoid":
        x = Conv2D(filters=output_filt, kernel_size=[1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.sigmoid(x)
    elif params.final_layer == "softmax":
        x = Conv2D(filters=output_filt, kernel_size=[1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


def net_builder(params):

    # determine network
    if params.model_name in globals():
        model = globals()[params.model_name](params)
    else:
        raise ValueError("Specified network does not exist in net_builder.py: " + params.model_name)

    return model
