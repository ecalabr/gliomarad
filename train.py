"""Train the model"""

import argparse
from glob import glob
import logging
import os
# set tensorflow logging level before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.INFO)
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from utilities.utils import Params
from utilities.utils import set_logger
from utilities.patch_input_fn import patch_input_fn
from utilities.learning_rates import learning_rate_picker
from model.model_fn import model_fn


# define functions
def train(param_file):

    # load params from param file
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

    # set up checkpoint directories and determine current epoch
    checkpoint_path = os.path.join(params.model_dir, 'checkpoints')
    latest_ckpt = None
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    checkpoints = glob(checkpoint_path + '/*.hdf5')
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        epoch_num = os.path.basename(latest_ckpt).split('epoch_')[1]
        # handle saves with validation loss in filename
        if '_valloss' in epoch_num:
            epoch_num = epoch_num.split('_valloss')[0]
        completed_epochs = int(epoch_num)
    else:
        completed_epochs = 0

    # Check for existing model and load if exists, otherwise create from scratch
    if latest_ckpt and not params.overwrite:
        print("Found checkpoint file {}, attempting to resume...".format(latest_ckpt))
        # Load model checkpoints:
        model = model_fn(params)  # recreating model is neccesary if custom loss function is being used
        model.load_weights(latest_ckpt)
        print("- done loading model")
    else:
        # Define the model
        logging.info("Creating the model...")
        model = model_fn(params)
        logging.info("- done creating model")

    # generate dataset objects for model inputs
    train_inputs = patch_input_fn(params, mode='train')
    eval_inputs = patch_input_fn(params, mode='eval')

    # CALLBACKS
    # define learning rate schedule callback for model
    learning_rate = learning_rate_picker(params.learning_rate, params.learning_rate_decay)
    learning_rate = LearningRateScheduler(learning_rate)

    # set checkpoint save callback
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    if eval_inputs:
        ckpt = os.path.join(checkpoint_path, 'epoch_{epoch:02d}_valloss_{val_loss:.4f}.hdf5')
    else:
        ckpt = os.path.join(checkpoint_path, 'epoch_{epoch:02d}.hdf5')
    checkpoint = ModelCheckpoint(ckpt, monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=False,
                                 mode='auto', save_freq='epoch')

    # tensorboard callback
    tensorboard = TensorBoard(
        log_dir=params.model_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch',
        profile_batch=2, embeddings_freq=0, embeddings_metadata=None)

    # combine callbacks for the model
    train_callbacks = [learning_rate, checkpoint, tensorboard]

    # TRAINING
    # Train the model with evalutation after each epoch
    logging.info("Training the model...")
    model.fit(train_inputs,
              epochs=params.num_epochs,
              initial_epoch=completed_epochs,
              steps_per_epoch=params.samples_per_epoch//params.batch_size,
              callbacks=train_callbacks,
              validation_data=eval_inputs,
              shuffle=False,
              verbose=1)


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
