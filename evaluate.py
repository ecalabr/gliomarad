""" Evaluate the model """

import argparse
import logging
import os
from model.utils import Params
from model.utils import set_logger
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate

########################## define functions ##########################
def evaluate_one(param_file):
    # get params
    params = Params(param_file)

    # determine model dir
    if params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        params.model_dir = os.path.dirname(param_file)
    if not os.path.isdir(params.model_dir):
        raise ValueError("Specified model directory does not exist")

    # Set the logger, delete old log file if overwrite param is set to yes
    log_path = os.path.join(params.model_dir, 'eval.log')
    if os.path.isfile(log_path) and params.overwrite == 'yes':
        os.remove(log_path)
    set_logger(log_path)
    logging.info("Log file created at " + log_path)

    # Create the input dataset
    logging.info("Generating dataset object...")
    eval_inputs = input_fn(mode='eval', params=params)
    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    eval_model_spec = model_fn(eval_inputs, params, mode='eval', reuse=True)
    logging.info("- done.")

    # Evaluate the model
    logging.info("Starting evaluation.")
    evaluate(eval_model_spec, params.model_dir, params.restore_dir)

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
    evaluate_one(args.param_file)
