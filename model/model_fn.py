from model.net_builder import net_builder
from utilities.losses import loss_picker
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
from contextlib import redirect_stdout


def model_fn(params, metrics=('accuracy',)):

    # consider adding metric choices here
    if not isinstance(metrics, list):
        metrics = list(metrics)

    # Define model and loss using loss picker function
    with params.strategy.scope():
        model = net_builder(params)
        loss = loss_picker(params)
        model.compile(optimizer=Adam(), loss=loss, metrics=metrics)

    # save text representation of graph
    model_sum = os.path.join(params.model_dir, 'model_summary.txt')
    if not os.path.isfile(model_sum):
        with open(model_sum, 'w+') as f:
            with redirect_stdout(f):
                model.summary()

    # save graphical representation of graph
    model_im = os.path.join(params.model_dir, 'model.png')
    if not os.path.isfile(model_im):
        tf.keras.utils.plot_model(
            model, to_file=model_im, show_shapes=False, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96)

    return model
