
import argparse
import logging
import os
from model.utils import Params, set_logger
from model.prediction import predict
from model.input_fn import input_fn
from model.model_fn import model_fn
import tensorflow as tf


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--param_file', default='/home/ecalabr/PycharmProjects/gbm_preproc/model/params.json',
                    help="Path to params.json")

if __name__ == '__main__':

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
    params = Params(args.param_file)

    # Set the logger, delete old log file if overwrite param is set to yes
    log_path = os.path.join(params.model_dir, 'predict.log')
    if os.path.isfile(log_path) and params.overwrite == 'yes':
        os.remove(log_path)
    set_logger(os.path.join(params.model_dir, 'predict.log'))
    logging.info("Log file created at " + log_path)

    # Create the two iterators over the two datasets
    logging.info("Generating dataset objects...")
    infer_inputs = input_fn(mode='infer', params=params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    infer_model_spec = model_fn(infer_inputs, params, mode='infer', reuse=tf.AUTO_REUSE)
    logging.info("- done.")

    # Train the model
    predict(infer_model_spec, params.model_dir, params)