"""Train the model"""

import argparse
from glob import glob
import logging
import os
# set tensorflow logging to FATAL before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from utilities.utils import Params
from utilities.utils import set_logger
from utilities.patch_input_fn import patch_input_fn, patch_input_fn_3d
from utilities.learning_rates import learning_rate_picker
from model.model_fn import model_fn


# define functions
def train(param_file):
    # get params
    params = Params(param_file)

    # determine model dir
    if params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        params.model_dir = os.path.dirname(param_file)
    if not os.path.isdir(params.model_dir):
        raise ValueError("Specified model directory does not exist")

    # Set the logger, delete old log file if overwrite param is set to yes
    log_path = os.path.join(params.model_dir, 'train.log')
    if os.path.isfile(log_path) and params.overwrite:
        print("Overwriting existing log...")
        os.remove(log_path)
    set_logger(log_path)
    logging.info("Log file created at " + log_path)

    # Determine if 2d or 3d and create the two iterators over the two datasets
    logging.info("Generating dataset objects...")
    if params.dimension_mode == '2D':  # handle 2d inputs
        train_inputs = patch_input_fn(mode='train', params=params)
        eval_inputs = patch_input_fn(mode='eval', params=params)
    elif params.dimension_mode in ['2.5D', '3D']:  # handle 3d inputs
        train_inputs = patch_input_fn_3d(mode='train', params=params)
        eval_inputs = patch_input_fn_3d(mode='eval', params=params)
    else:
        raise ValueError("Training dimensions mode not understood: " + str(params.dimension_mode))
    logging.info("- done generating dataset objects")

    # Check for existing model and load if exists, otherwise create from scratch
    completed_epochs = 0
    checkpoint_path = os.path.join(params.model_dir, 'checkpoints')
    checkpoints = glob(checkpoint_path + '/*.hdf5')
    if checkpoints and not params.overwrite:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print("Found checkpoint file {}, attempting to resume...".format(latest_ckpt))
        # Load model checkpoints:
        model = model_fn(params)  # recreating model is neccesary if custom loss function is being used
        model.load_weights(latest_ckpt)
        print("- done loading model")
        # Finding the epoch index from which we are resuming
        completed_epochs = int(os.path.basename(latest_ckpt).split('.')[1])
    else:
        # Define the model
        logging.info("Creating the model...")
        model = model_fn(params)
        logging.info("- done creating model")
    epochs_todo = params.num_epochs - completed_epochs

    # define learning rate schedule callback for model
    learning_rate = learning_rate_picker(params.learning_rate, params.learning_rate_decay)
    learning_rate = LearningRateScheduler(learning_rate)

    # set checkpoint callback
    checkpoint_path = os.path.join(params.model_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    ckpt = os.path.join(checkpoint_path, 'weights.{epoch:02d}.hdf5')
    # save checkpoint only if validation loss is higher
    checkpoint = ModelCheckpoint(ckpt, monitor='val_loss', verbose=1, save_weights_only=False,
                                 save_best_only=True, mode='auto', save_freq='epoch')

    # tensorboard callback
    tensorboard = TensorBoard(
        log_dir=params.model_dir, histogram_freq=1, write_graph=True, write_images=True,
        update_freq='epoch', profile_batch=2, embeddings_freq=0,
        embeddings_metadata=None)

    # combine callbacks for the model
    callbacks = [learning_rate, checkpoint, tensorboard]

    # Train the model
    logging.info("Starting training for {} epochs out of a total of {} epochs".format(epochs_todo, params.num_epochs))
    model.fit(train_inputs, epochs=params.num_epochs, callbacks=callbacks, validation_data=eval_inputs, shuffle=False,
              initial_epoch=completed_epochs)


# executed  as script
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', default=None,
                        help="Path to params.json")

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert args.param_file, "Must specify a parameter file using --param_file"
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)

    # do work
    train(args.param_file)
