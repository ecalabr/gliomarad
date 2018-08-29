"""Train the model"""

import argparse
import logging
import os
import tensorflow as tf
from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--param_file', default='/media/ecalabr/data2/model/params.json',
                    help="Path to params.json")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    #tf.set_random_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
    params = Params(args.param_file)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    #model_dir_has_best_weights = os.path.isdir(params.model_dir)
    #overwritting = model_dir_has_best_weights and args.restore_dir is None
    #assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(params.model_dir, 'train.log'))

    # Create the two iterators over the two datasets
    train_inputs = input_fn(is_training=True, params=params)
    eval_inputs = input_fn(is_training=False, params=params)
    logging.info("- done.")


    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model_spec = model_fn(train_inputs, params, mode='train')
    eval_model_spec = model_fn(eval_inputs, params, mode='eval')
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, params.model_dir, params, params.restore_dir)