"""Train the model"""

import argparse
from glob import glob
import logging
import os
# set tensorflow logging to FATAL before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.INFO)
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from utilities.utils import Params
from utilities.utils import set_logger
from utilities.patch_input_fn import patch_input_fn
from utilities.learning_rates import learning_rate_picker
from model.model_fn import model_fn
import shutil


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

    # set up checkpoint directories and determine current epoch and val_loss
    # last checkpoint directory/epoch setup
    last_weights_path = os.path.join(params.model_dir, 'last_weights')
    latest_ckpt = None
    if not os.path.isdir(last_weights_path):
        os.mkdir(last_weights_path)
    last_checkpoints = glob(last_weights_path + '/*.hdf5')
    if last_checkpoints:
        latest_ckpt = max(last_checkpoints, key=os.path.getctime)
        completed_epochs = int(os.path.basename(latest_ckpt).split('after_epoch_')[1])
    else:
        completed_epochs = 0
    # best checkpoint directory/val_loss setup
    best_weights_path = os.path.join(params.model_dir, 'best_weights')
    if not os.path.isdir(best_weights_path):
        os.mkdir(best_weights_path)
    best_checkpoints = glob(last_weights_path + '/*.hdf5')
    if best_checkpoints:
        best_ckpt = max(best_checkpoints, key=os.path.getctime)
        best_val_loss = float(os.path.basename(best_ckpt).split('val_loss_')[1])
    else:
        best_val_loss = float('inf')

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
    epochs_todo = params.num_epochs - completed_epochs

    # define learning rate schedule callback for model
    learning_rate = learning_rate_picker(params.learning_rate, params.learning_rate_decay)
    learning_rate = LearningRateScheduler(learning_rate)

    # set checkpoint callback
    if not os.path.isdir(last_weights_path):
        os.mkdir(last_weights_path)
    ckpt = os.path.join(last_weights_path, 'after_epoch_{epoch:02d}.hdf5')
    # save checkpoint only if validation loss is higher (in case validation is not performed use training loss)
    checkpoint = ModelCheckpoint(ckpt, monitor='loss', verbose=1, save_weights_only=False, save_best_only=False,
                                 mode='auto', save_freq='epoch')

    # tensorboard callback
    tensorboard = TensorBoard(
        log_dir=params.model_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch',
        profile_batch=2, embeddings_freq=0, embeddings_metadata=None)

    # combine callbacks for the model
    train_callbacks = [learning_rate, checkpoint, tensorboard]

    # make inputs
    train_inputs = patch_input_fn(params, mode='train')
    eval_inputs = patch_input_fn(params, mode='eval')

    # Train the model with evalutation after each epoch
    logging.info("Starting training for {} epochs out of a total of {} epochs".format(epochs_todo, params.num_epochs))
    epochs_left = params.num_epochs - completed_epochs
    while epochs_left > 0:
        # train one epoch
        logging.info("Training the model...")
        model.fit(train_inputs, epochs=completed_epochs + 1, initial_epoch=completed_epochs, callbacks=train_callbacks,
                  shuffle=False, verbose=1)
        # eval one epoch
        logging.info("Evaluating the model...")
        results = model.evaluate(eval_inputs, verbose=1, callbacks=None)
        # move to best_weights directory if val_loss improves
        if results[0] < best_val_loss:
            logging.info("- validation loss improved from {:0.6f} to {:0.6f}".format(best_val_loss, results[0]))
            best_val_loss = results[0]
            latest_ckpt = max(glob(last_weights_path + '/*.hdf5'), key=os.path.getctime)
            best_ckpt = os.path.join(best_weights_path,
                                     os.path.basename(latest_ckpt).split('.hdf5')[0]
                                     + '_val_loss_{:0.6f}.hdf5'.format(results[0]))
            logging.info("- saving new best weights to {}".format(best_ckpt))
            shutil.copy(latest_ckpt, best_ckpt)
        else:
            logging.info("- validation loss {:0.6f} did not improve from {:0.6f}".format(results[0], best_val_loss))
        # increment epochs counters
        epochs_left = epochs_left - 1
        completed_epochs = completed_epochs + 1


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
