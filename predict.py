
import argparse
import logging
import os
from model.utils import Params, set_logger
from model.prediction import predict
from model.patch_input_fn import infer_input_fn, infer_input_fn_3d
from model.model_fn import model_fn


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--param_file', default='/home/ecalabr/PycharmProjects/gbm_preproc/model/params.json',
                    help="Path to params.json")
parser.add_argument('--infer_dir', default='/media/ecalabr/data2/qc_complete/12309838',
                    help="Path to directory to generate inference from")
parser.add_argument('--best_last', default='last_weights',
                    help="Either 'best_weights' or 'last_weights' - whether to use best or last model weights for inference")

if __name__ == '__main__':

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
    params = Params(args.param_file)
    infer_dir = args.infer_dir
    best_last = args.best_last
    if best_last not in ['best_weights', 'last_weights']:
        raise ValueError("Did not understand best_last value: " + str(best_last))

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
    # handle 2D vs 3D
    if params.dimension_mode == '2D':
        infer_inputs = infer_input_fn(params=params, infer_dir=infer_dir)  # can optionally pass specific dirs here
    elif params.dimension_mode in ['2.5D', '3D']:
        infer_inputs = infer_input_fn_3d(params=params, infer_dir=infer_dir)
    else:
        raise ValueError("Training dimensions mode not understood: " + str(params.dimension_mode))
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    infer_model_spec = model_fn(infer_inputs, params, mode='infer', reuse=False)  # reuse only if model already exists
    logging.info("- done.")

    # predict using the model
    predict(infer_model_spec, params.model_dir, params, infer_dir, best_last)