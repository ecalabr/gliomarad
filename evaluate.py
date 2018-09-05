"""Evaluate the model"""

import argparse
import logging
import os
from model.utils import Params
from model.utils import set_logger
from model.input_fn import input_fn
from model.model_fn import model_fn
import tensorflow as tf
from model.evaluation import evaluate_sess


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--param_file', default='/home/ecalabr/PycharmProjects/gbm_preproc/model/params.json',
                    help="Path to params.json")

if __name__ == '__main__':

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
    params = Params(args.param_file)

    # Check that we are not overwriting some previous experiment
    # if params.overwrite != 'yes' and os.listdir(params.model_dir):
    #    raise ValueError("Overwrite param is not 'yes' but there are files in specified model directory!")

    # Set the logger, delete old log file if overwrite param is set to yes
    log_path = os.path.join(params.model_dir, 'eval.log')
    if os.path.isfile(log_path) and params.overwrite == 'yes':
        os.remove(log_path)
    set_logger(log_path)
    logging.info("Log file created at " + log_path)

    # Create the two iterators over the two datasets
    logging.info("Generating dataset objects...")
    eval_inputs = input_fn(mode='eval', params=params)
    logging.info("- done.")

    # Define the models
    logging.info("Creating the model...")
    eval_model_spec = model_fn(eval_inputs, params, mode='eval', reuse=False)
    logging.info("- done.")

    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(eval_model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        logging.info("Loading model weights...")
        save_path = os.path.join(params.model_dir, params.restore_dir)
        meta = os.path.join(save_path, 'after-epoch-10.meta')
        checkpoint = tf.train.latest_checkpoint(save_path)
        
        saver.restore(sess, save_path)
        logging.info("- done.")

        # Evaluate
        metrics = evaluate_sess(sess, eval_model_spec)
        # metrics_name = '_'.join(restore_from.split('/'))
        # save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
        # save_dict_to_json(metrics, save_path)
        print(metrics)