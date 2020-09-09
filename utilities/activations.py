import tensorflow as tf


# regular relu
def relu():
    return tf.keras.layers.ReLU(
        max_value=None,
        negative_slope=0,
        threshold=0
    )

# leaky relu
def leaky_relu():
    return tf.keras.layers.LeakyReLU(
        alpha=0.3
    )

# prelu
def prelu():
    return tf.keras.layers.PReLU(
        alpha_initializer='zeros',
        alpha_regularizer=None,
        alpha_constraint=None,
        shared_axes=None
    )

def activation_layer(activation_method):

    # sanity checks
    if not isinstance(activation_method, str):
        raise ValueError("Activation parameter must be a string")

    # check for specified loss method and error if not found
    if activation_method.lower() in globals():
        return globals()[activation_method]()
    else:
        try:
            return tf.keras.layers.Activation(activation_method)
        except:
            raise NotImplementedError("Specified learning rate decay method is not implemented in learning_rates.py: "
                                  + activation_method)
