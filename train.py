"""Train the model"""

import argparse
from glob import glob
import logging
import os
# set tensorflow logging level before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.INFO)
import tensorflow as tf
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
        raise ValueError("Specified model directory does not exist: {}".format(params.model_dir))

    # Set the logger, delete old log file if overwrite param is set to yes
    log_path = os.path.join(params.model_dir, 'train.log')
    if os.path.isfile(log_path) and params.overwrite:
        os.remove(log_path)
    set_logger(log_path)
    logging.info("Using model directory {}".format(params.model_dir))
    logging.info("Using logging file: {}".format(log_path))
    logging.info("Using TensorFlow version {}".format(tf.__version__))

    # determine distribute strategy
    if params.dist_strat.lower() == 'mirrored':
        logging.info("Using Mirrored distribution strategy")
        params.strategy = tf.distribute.MirroredStrategy()
    else:
        params.strategy = tf.distribute.get_strategy()
    params.batch_size = params.batch_size * params.strategy.num_replicas_in_sync

    # check other important params
    if not os.path.isdir(params.data_dir):
        raise ValueError("Specified data directory does not exist: {}".format(params.data_dir))
    logging.info("Using data directory {}".format(params.data_dir))

    # set up checkpoint directories and determine current epoch
    checkpoint_path = os.path.join(params.model_dir, 'checkpoints')
    latest_ckpt = None
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    checkpoints = glob(checkpoint_path + '/*.hdf5')
    if checkpoints and not params.overwrite:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        completed_epochs = int(os.path.splitext(os.path.basename(latest_ckpt).split('epoch_')[1])[0].split('_')[0])
        logging.info("Checkpoint exists for epoch {}".format(completed_epochs))
    else:
        completed_epochs = 0

    # generate dataset objects for model inputs
    train_inputs = patch_input_fn(params, mode='train')
    eval_inputs = patch_input_fn(params, mode='eval')

    # Check for existing model and load if exists, otherwise create from scratch
    if latest_ckpt and not params.overwrite:
        logging.info("Loading checkpoint file {}".format(latest_ckpt))
        # Load model checkpoints:
        model = model_fn(params)  # recreating model is neccesary if custom loss function is being used
        model.load_weights(latest_ckpt)
    else:
        # Define the model
        logging.info("Creating the model...")
        model = model_fn(params)

    # SET CALLBACKS
    # define learning rate schedule callback for model
    learning_rate = LearningRateScheduler(learning_rate_picker(params.learning_rate, params.learning_rate_decay))

    # checkpoint save callback
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    # save validation loss in name if evaluation files are passed, else use train loss
    if params.train_fract < 1.:
        ckpt = os.path.join(checkpoint_path, 'epoch_{epoch:02d}_valloss_{val_loss:.4f}.hdf5')
    else:
        ckpt = os.path.join(checkpoint_path, 'epoch_{epoch:02d}_trainloss_{loss:.4f}.hdf5')
    checkpoint = ModelCheckpoint(
        ckpt,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=False,
        mode='auto',
        save_freq='epoch')

    # tensorboard callback
    tensorboard = TensorBoard(
        log_dir=params.model_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq=(params.samples_per_epoch // params.batch_size) // 100, # write losses/metrics 100x per epoch
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None)

    # combine callbacks for the model
    train_callbacks = [learning_rate, checkpoint, tensorboard]

    # report
    logging.info("Training for {} total epochs starting at epoch {}".format(params.num_epochs, completed_epochs + 1))

    # TRAINING
    model.fit(
        train_inputs,
        epochs=params.num_epochs,
        initial_epoch=completed_epochs,
        steps_per_epoch=params.samples_per_epoch // params.batch_size,
        callbacks=train_callbacks,
        validation_data=eval_inputs,
        shuffle=False,
        verbose=1)

    # completion logging
    logging.info("Successfully trained model for {} epochs".format(params.num_epochs - completed_epochs))

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
