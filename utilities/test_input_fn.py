import argparse
import logging
import os
# set tensorflow logging to FATAL before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
from utils import Params, display_tf_dataset, save_tf_dataset
from input_fn import get_input_fn
import numpy as np


# define functions
def test_input_fn(param_file, output, n=10, display=True, nii=False, skip=10):

    # get params
    params = Params(param_file)

    # determine model dir
    if params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        params.model_dir = os.path.dirname(param_file)
    if not os.path.isdir(params.model_dir):
        raise ValueError("Specified model directory does not exist")

    # load inputs with input function
    inputs = get_input_fn(params, mode='train').as_numpy_iterator()

    # determine if weighted
    weighted = False if isinstance(params.mask_weights, np.bool) and not params.mask_weights else True

    # run tensorflow session
    s = 0  # the overall slice count
    i = 0  # the output count
    e = 1  # the epoch count
    # iterate until display/saved count is equal to total count
    while i < n:
        try:
            data_slice = next(inputs)
            s += 1  # increment overall counter when slice is generated
        except tf.errors.OutOfRangeError:
            # increment epoch counter when input is exhausted and refresh generator
            e += 1
            inputs = get_input_fn(params, mode='train').as_numpy_iterator()
            data_slice = next(inputs)

        # increment counter at ever skip number of input slices
        if skip == 0 or s % skip == 0:
            i += 1
            print("Processing input slice {}, epoch {}, [{:03d} of {:03d}]".format(s, e, i, n))
            if display:
                display_tf_dataset(data_slice, params.data_format, params.train_dims, weighted=weighted)
            if nii:
                outname = os.path.join(output, "slice_{:03d}_epoch_{:02d}.nii.gz".format(s, e))
                save_tf_dataset(data_slice, params.data_format, params.train_dims, outname, weighted=weighted)
        else:
            print("Generated input slice {}, epoch {}, which is skipped".format(s, e))


# executed  as script
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_file', default=None,
                        help="Path to params.json")
    parser.add_argument('-o', '--output', default=None,
                        help="Path to output directory")
    parser.add_argument('-i', '--nii', default=False, action="store_true",
                        help="Write output to nifti volumes")
    parser.add_argument('-d', '--display', default=False, action="store_true",
                        help="Display output as a plot")
    parser.add_argument('-n', '--number', default=10,
                        help="Number of input iterations to consider")
    parser.add_argument('-s', '--skip', default=10,
                        help="Number of input iterations to skip between display/saving")

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert args.param_file, "Must specify param file with --param_file"
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)

    # handle arguments
    try:
        args.number = int(args.number)
    except:
        raise ValueError("Number argument (-n/--number) must be castable to int but is {}".format(args.number))
    try:
        args.skip = int(args.skip)
    except:
        raise ValueError("Skip argument (-s/--skip) must be castable to int but is {}".format(args.skip))
    assert os.path.isdir(args.output), "Specified output directory does not exist: {}".format(args.output)

    # do work
    test_input_fn(args.param_file, args.output, n=args.number, display=args.display, nii=args.nii, skip=args.skip)
