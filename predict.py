
import argparse
import logging
import os
from model.utils import Params, set_logger
from model.prediction import predict
from model.input_fn import infer_input_fn
from model.model_fn import model_fn


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--param_file', default='/home/ecalabr/PycharmProjects/gbm_preproc/model/params.json',
                    help="Path to params.json")
parser.add_argument('--infer_dir', default='/media/ecalabr/data2/qc_complete/12309838',
                    help="Path to directory to generate inference from")

if __name__ == '__main__':

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
    params = Params(args.param_file)
    infer_dir = args.infer_dir

    # determine model dir
    if params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        params.model_dir = os.path.dirname(args.param_file)
    if not os.path.isdir(params.model_dir):
        raise ValueError("Specified model directory does not exist")

    # Set the logger, delete old log file if overwrite param is set to yes
    log_path = os.path.join(params.model_dir, 'predict.log')
    if os.path.isfile(log_path) and params.overwrite == 'yes':
        os.remove(log_path)
    set_logger(os.path.join(params.model_dir, 'predict.log'))
    logging.info("Log file created at " + log_path)

    # Create the two iterators over the two datasets
    logging.info("Generating dataset objects...")
    # the below no longer needed after changes to input function for inference
    # params.batch_size = 1  # manually set batch size here so there is no dropped remainder
    infer_inputs = infer_input_fn(params=params, infer_dir=infer_dir)  # can optionally pass specific dirs here
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    infer_model_spec = model_fn(infer_inputs, params, mode='infer', reuse=False)  # reuse only if model already exists
    logging.info("- done.")

    # Train the model
    predict(infer_model_spec, params.model_dir, params, infer_dir)