"""Train the model"""

import argparse
import logging
import os
from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.patch_input_fn import patch_input_fn
from model.patch_input_fn import patch_input_fn_3d
from model.model_fn import model_fn

########################## define functions ##########################
def train_one(param_file):

    # get params
    params = Params(param_file)

    # determine model dir
    if params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        params.model_dir = os.path.dirname(param_file)
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
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model_spec = model_fn(train_inputs, params, mode='train', reuse=False)
    eval_model_spec = model_fn(eval_inputs, params, mode='eval', reuse=True)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    restore_dir = os.path.join(params.model_dir, params.restore_dir)  # get full path to restore directory
    train_and_evaluate(train_model_spec, eval_model_spec, params.model_dir, params, restore_dir)

########################## executed  as script ##########################
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', default='/home/ecalabr/PycharmProjects/gbm_preproc/model/params.json',
                        help="Path to params.json")

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)

    # do work
    train_one(args.param_file)
