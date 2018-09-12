"""Train the model"""

import argparse
import logging
import os
from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn
from model.model_fn import model_fn


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--param_file', default='/home/ecalabr/PycharmProjects/gbm_preproc/model/params.json',
                    help="Path to params.json")


if __name__ == '__main__':

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
    params = Params(args.param_file)

    # determine model dir
    if params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        params.model_dir = os.path.dirname(args.param_file)
    if not os.path.isdir(params.model_dir):
        raise ValueError("Specified model directory does not exist")

    # Check that we are not overwriting some previous experiment
    if params.overwrite != 'yes' and os.listdir(params.model_dir):
        raise ValueError("Overwrite param is not 'yes' but there are files in specified model directory!")

    # Set the logger, delete old log file if overwrite param is set to yes
    log_path = os.path.join(params.model_dir, 'train.log')
    if os.path.isfile(log_path) and params.overwrite == 'yes':
        os.remove(log_path)
    set_logger(log_path)
    logging.info("Log file created at " + log_path)

    # Create the two iterators over the two datasets
    logging.info("Generating dataset objects...")
    train_inputs = input_fn(mode='train', params=params)
    eval_inputs = input_fn(mode='eval', params=params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model_spec = model_fn(train_inputs, params, mode='train', reuse=False)
    eval_model_spec = model_fn(eval_inputs, params, mode='eval', reuse=True)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    restor_dir = os.path.join(params.model_dir, params.restore_dir)  # get full path to restore directory
    train_and_evaluate(train_model_spec, eval_model_spec, params.model_dir, params, params.restore_dir)