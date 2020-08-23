import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Conv2D, Conv2DTranspose, Input, MaxPool3D
from tensorflow.keras.models import Model
from model.net_layers import bneck_resid3d, bneck_resid2d, conv3d_act_bn
from tensorflow.python.eager import backprop
from tensorflow.keras.mixed_precision import experimental as mixed_precision


# helper functions

# custom model class
class CustomModel(Model):
    def train_step(self, data):
        # unpack dataset data
        x = data[0]
        if isinstance(data[1], tuple):
            y = data[1][0]
            weights = data[1][1]
        else:
            y = data[1]
            weights = None

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, weights, regularization_losses=self.losses)
        # For custom training steps, users can just write:
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, weights)
        return {m.name: m.result() for m in self.metrics}

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

    # additional setup for network construction
    skips = []
    horz_layers = layer_layout[-1]
    unet_layout = layer_layout[:-1]

    # input layer
    inputs = Input(shape=train_dims, batch_size=batch_size)

    # initial convolution layer
    x = Conv3D(filt, ksize, padding='same', data_format=dfmt)(inputs)

    # unet encoder limb
    for n, n_layers in enumerate(unet_layout):
        # horizontal layers
        for layer in range(n_layers):
            # residual blocks with activation and batch norm
            x = conv3d_act_bn(x, filt, ksize, dfmt, dropout, act)

        # create skip connection
        skips.append(tf.identity(x))

        # downsample block, double filters
        filt = filt * 2  # double filters before downsampling
        x = MaxPool3D(pool_size=[2, 2, 2], padding='same', data_format=dfmt)(x)

    # unet horizontal (bottom) bottleneck blocks
    for layer in range(horz_layers):
        x = conv3d_act_bn(x, filt, ksize, dfmt, dropout, act)

    # reverse layout and skip connections for decoder limb
    skips.reverse()
    unet_layout.reverse()

    # unet decoder limb with residual bottleneck blocks
    for n, n_layers in enumerate(unet_layout):

        # upsample block
        filt = int(round(filt/2))  # half filters before upsampling
        x = Conv3DTranspose(filt, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt)(x)

        # fuse skip connections with concatenation of features
        x = tf.concat([x, skips[n]], axis=-1 if dfmt == 'channels_last' else 1)

        # horizontal blocks
        for layer in range(n_layers):
            x = conv3d_act_bn(x, filt, ksize, dfmt, dropout, act)

    # output layer
    if params.final_layer == "conv":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
    elif params.final_layer == "sigmoid":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.sigmoid(x, dtype='float32')
    elif params.final_layer == "softmax":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1, dtype='float32')
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


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
        x = tf.concat([x, skips[n]], axis=-1 if dfmt == 'channels_last' else 1)

        # horizontal blocks
        for layer in range(n_layers):
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act)

    # output layer
    if params.final_layer == "conv":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
    elif params.final_layer == "sigmoid":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.sigmoid(x, dtype='float32')
    elif params.final_layer == "softmax":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1, dtype='float32')
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


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

        # do z dimension conv
        x = bneck_resid3d(x, filt, [1, 1, 3], dfmt, dropout, act)

        # create skip connection
        skips.append(tf.identity(x))

        # downsample block
        filt = filt * 2  # double filters before downsampling
        x = Conv3D(filt, ksize, strides=[2, 2, 1], padding='same', data_format=dfmt)(x)

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
        x = Conv3DTranspose(filt, ksize, strides=[2, 2, 1], padding='same', data_format=dfmt)(x)

        # do z dimension conv
        x = bneck_resid3d(x, filt, [1, 1, 3], dfmt, dropout, act)

        # fuse skip connections with concatenation of features
        x = tf.concat([x, skips[n]], axis=-1 if dfmt == 'channels_last' else 1)

        # horizontal blocks
        for layer in range(n_layers):
            x = bneck_resid3d(x, filt, ksize, dfmt, dropout, act)

    # output layer
    if params.final_layer == "conv":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
    elif params.final_layer == "sigmoid":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.sigmoid(x, dtype='float32')
    elif params.final_layer == "softmax":
        x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1, dtype='float32')
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
        x = Conv2D(filters=output_filt, kernel_size=[1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
    elif params.final_layer == "sigmoid":
        x = Conv2D(filters=output_filt, kernel_size=[1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.sigmoid(x, dtype='float32')
    elif params.final_layer == "softmax":
        x = Conv2D(filters=output_filt, kernel_size=[1, 1], padding='same', data_format=dfmt)(x)
        x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1, dtype='float32')
    else:
        assert ValueError("Specified final layer is not implemented: {}".format(params.final_layer))

    return Model(inputs=inputs, outputs=x)


# Wrapper function
def net_builder(params):
    # set up mixed precision computation to take advantage of Nvidia tensor cores
    # https://www.tensorflow.org/guide/mixed_precision
    if params.mixed_precision:
        print("WARNING: using tensorflow mixed precision... This could lead to numeric instability in some cases.")
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        # warn if batch size and/or nfilters is not a multpile of 8
        if not params.base_filters % 8 == 0:
            print("WARNING: parameter base_filters is not a multiple of 8, which will slow down tensor cores.")
        if not params.batch_size % 8 == 0:
            print("WARNING: parameter batch_size is not a multiple of 8, which will slow down tensor cores.")

    # determine network
    if params.model_name in globals():
        model = globals()[params.model_name](params)
    else:
        raise ValueError("Specified network does not exist in net_builder.py: " + params.model_name)

    return model
