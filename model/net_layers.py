import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv3D, Conv2D, Dropout, Conv3DTranspose, Dense
from utilities.activations import activation_layer


def dense_act_bn(x, filt, dropout=0., act='relu', policy='float32', reg=None):
    x = Dense(filt, activation=act, dtype=policy, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    if dropout > 0.:
        x = Dropout(rate=dropout)(x)
    return x


def conv3d_act_bn(x, filt, ksize, dfmt, dropout, act, policy='float32'):
    x = Conv3D(filt, ksize, padding='same', data_format=dfmt, dtype=policy)(x)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    if dropout > 0.:
        x = Dropout(rate=dropout)(x)
    return x


def bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy='float32'):

    # create shortcut
    shortcut = x

    # perform first 1x1x1 conv for bottleneck block using 1/4 filters
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3D(int(round(filt/4)), kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # perform 3x3x3 conv for bottleneck block with bn and activation using 1/4 filters (optionally strided)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3D(int(round(filt/4)), ksize, padding='same', data_format=dfmt, dtype=policy)(x)

    # perform second 1x1x1 conv with full filters (no strides)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3D(filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # optional dropout layer
    if dropout > 0.:
        x = Dropout(rate=dropout)(x)

    # fuse shortcut with tensor output, transforming filter number as needed
    if x.shape[-1] == shortcut.shape[-1]:
        x = tf.add(x, shortcut)
    else:
        shortcut = Conv3D(filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(shortcut)
        x = tf.add(x, shortcut)

    return x


def bneck_resid3d_down(x, filt, ksize, dfmt, strides, dropout, act, policy='float32'):

    # create shortcut, downsample skip with conv layer
    shortcut = x
    shortcut = Conv3D(filt, kernel_size=[1, 1, 1], strides=strides, padding='same', data_format=dfmt,
                      dtype=policy)(shortcut)

    # perform first 1x1x1 conv for bottleneck block using 1/4 filters
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3D(int(round(filt/4)), kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # perform 3x3x3 conv for bottleneck block with bn and activation using 1/4 filters (optionally strided)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3D(int(round(filt/4)), ksize, strides=strides, padding='same', data_format=dfmt, dtype=policy)(x)

    # perform second 1x1x1 conv with full filters (no strides)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3D(filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # optional dropout layer
    if dropout > 0.:
        x = Dropout(rate=dropout)(x)

    # fuse shortcut with tensor output
    x = tf.add(x, shortcut)

    return x


def bneck_resid3d_up(x, filt, ksize, dfmt, strides, dropout, act, policy='float32'):

    # create shortcut, upsample skip with conv layer
    shortcut = x
    shortcut = Conv3DTranspose(filt, kernel_size=[1, 1, 1], strides=strides, padding='same', data_format=dfmt,
                               dtype=policy)(shortcut)

    # perform first 1x1x1 conv for bottleneck block using 1/4 filters
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3D(int(round(filt/4)), kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # perform 3x3x3 conv for bottleneck block with bn and activation using 1/4 filters and upsample
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3DTranspose(int(round(filt/4)), ksize, strides=strides, padding='same', data_format=dfmt, dtype=policy)(x)

    # perform second 1x1x1 conv with full filters (no strides)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3D(filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # optional dropout layer
    if dropout > 0.:
        x = Dropout(rate=dropout)(x)

    # fuse shortcut with tensor output
    x = tf.add(x, shortcut)

    return x


def bneck_resid2d(x, filt, ksize, dfmt, dropout, act, policy='float32'):

    # create shortcut, and if strided, downsample skip with conv layer
    shortcut = x

    # perform first 1x1 conv for bottleneck block using 1/4 filters
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv2D(int(round(filt/4)), kernel_size=[1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # perform 3x3 conv for bottleneck block with bn and activation using 1/4 filters (optionally strided)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv2D(int(round(filt/4)), ksize, padding='same', data_format=dfmt, dtype=policy)(x)

    # perform second 1x1 conv with full filters (no strides)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv2D(filt, kernel_size=[1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # optional dropout layer
    if dropout > 0.:
        x = Dropout(rate=dropout)(x)

    # fuse shortcut with tensor output
    x = tf.add(x, shortcut)

    return x
