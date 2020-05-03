import argparse
import logging
import os
from utilities.utils import Params, set_logger
from model.prediction import predict
from utilities.patch_input_fn import infer_input_fn, infer_input_fn_3d
from model.model_fn import model_fn
from glob import glob


# define funiction to predict one inference directory
def predict_one(params, infer_dir, best_last, out_dir):

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
    nii_out = predict(infer_model_spec, params.model_dir, params, infer_dir, best_last, out_dir)

    return nii_out


# executed  as script
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', default=None,
                        help="Path to params.json")
    parser.add_argument('--infer_dir', default=None,
                        help="Path to directory to generate inference from")
    parser.add_argument('--best_last', default='last_weights',
                        help="'best_weights' or 'last_weights' - whether to use best or last weights for inference")
    parser.add_argument('--out_dir', default=None,
                        help="Optionally specify output directory")

    # handle param argument
    args = parser.parse_args()
    assert args.param_file, "Must specify param file using --param_file"
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
    my_params = Params(args.param_file)

    # handle best_last argument
    if args.best_last not in ['best_weights', 'last_weights']:
        raise ValueError("Did not understand best_last value: " + str(args.best_last))

    # handle out_dir argument
    if args.out_dir:
        assert os.path.isdir(args.out_dir), "Specified output directory does not exist: {}".format(args.out_dir)

    # handler inference directory argument
    assert args.infer_dir, "No infer directory specified. Use --infer_dir="
    assert os.path.isdir(args.infer_dir), "No inference directory found at {}".format(args.infer_dir)
    if not glob(args.infer_dir + '/*' + my_params.data_prefix[0] + '.nii.gz'):
        raise ValueError("No image data found in inference directory: {}".format(args.infer_dir))

    # determine model dir
    if my_params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        my_params.model_dir = os.path.dirname(args.param_file)
    if not os.path.isdir(my_params.model_dir):
        raise ValueError("Specified model directory does not exist")

    # do work
    nii_output = predict_one(my_params, args.infer_dir, args.best_last, args.out_dir)
