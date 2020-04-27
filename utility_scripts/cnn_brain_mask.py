""" hacky script for making a bunch of brain masks using a CNN """

import argparse
import os
from glob import glob
from model.utils import Params
from predict import predict_one
from utility_scripts.prob2seg import convert_prob
import tensorflow as tf


# define function to make a batch of brain masks from a list of directories
def batch_mask(infer_direcs, param_files, best_last, out_dir, overwrite=False):
    # initiate outputs
    outnames = []
    # run inference and post-processing for each infer_dir
    for direc in infer_direcs:
        # inner for loop for multiple models
        probs = []
        for param_file in param_files:
            # load params and determine model dir
            params = Params(param_file)
            if params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
                params.model_dir = os.path.dirname(param_file)
            if not os.path.isdir(params.model_dir):
                raise ValueError("Specified model directory does not exist")

            # run predict on one directory and get the output probabilities
            prob = predict_one(params, direc, best_last, out_dir)
            probs.append(prob)

            # clear graph
            tf.compat.v1.reset_default_graph()

        # convert probs to mask with cleanup
        idno = os.path.basename(direc.rsplit('/', 1)[0] if direc.endswith('/') else direc)
        nii_out_path = os.path.join(direc, idno + "_combined_brain_mask.nii.gz")
        if os.path.isfile(nii_out_path) and not overwrite:
            print("Mask file already exists at {}".format(nii_out_path))
        else:
            if probs:
                nii_out_path = convert_prob(probs, nii_out_path, clean=True)

                # report
                if os.path.isfile(nii_out_path):
                    print("Created mask file at: {}".format(nii_out_path))
                else:
                    raise ValueError("No mask output file found at: {}".format(direc))

                # add to outname list
                outnames.append(nii_out_path)

    return outnames


# executed  as script
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', default=None,
                        help="Path to params.json")
    parser.add_argument('--model_dir', default=None,
                        help="Path to model directory containing multiple param files. (optional, replaces param_file)")
    parser.add_argument('--infer_dir', default=None,
                        help="Path to directory to generate inference from")
    parser.add_argument('--best_last', default='best_weights',
                        help="'best_weights' or 'last_weights' - whether to use best/last model weights for inference")
    parser.add_argument('--out_dir', default=None,
                        help="Optionally specify temporary directory. Default is model directory.")
    parser.add_argument('--start', default=0,
                        help="Index of directories to start processing at")
    parser.add_argument('--end', default=None,
                        help="Index of directories to end processing at")
    parser.add_argument('--list', action="store_true", default=False,
                        help="List the directories to be processed in order then exit")
    parser.add_argument('--overwrite', action="store_true", default=False,
                        help="Overwrite existing brain mask")

    # handle model_dir argument
    args = parser.parse_args()
    if args.model_dir:
        my_param_files = glob(args.model_dir + '/*/params.json')
        if not my_param_files:
            raise ValueError("No parameter files found in model directory {}".format(args.model_dir))
    else:
        # handle params argument
        assert args.param_file, "Must specify param file using --param_file"
        assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
        my_param_files = [args.param_file]
    for f in my_param_files:
        print(f)

    # handle best_last argument
    if args.best_last not in ['best_weights', 'last_weights']:
        raise ValueError("Did not understand best_last value: " + str(args.best_last))

    # handle out_dir argument
    if args.out_dir:
        assert os.path.isdir(args.out_dir), "Specified output directory does not exist: {}".format(args.out_dir)

    # handle inference directory argument
    assert args.infer_dir, "No infer directory specified. Use --infer_dir"
    assert os.path.isdir(args.infer_dir), "No inference directory found at {}".format(args.infer_dir)

    # check if provided dir is a single image dir or a dir full of image dirs
    if glob(args.infer_dir + '/*.nii.gz'):
        infer_dirs = [args.infer_dir]
    elif glob(args.infer_dir + '/*/*.nii.gz'):
        infer_dirs = sorted(list(set([os.path.dirname(f)
                                      for f in glob(args.infer_dir + '/*/*.nii.gz')])))
    else:
        raise ValueError("No image data found in inference directory: {}".format(args.infer_dir))

    # handle list argument
    if args.list:
        for i, item in enumerate(infer_dirs, 0):
            print(str(i) + ': ' + item)
        exit()

    # handle start and end arguments
    if args.end:
        infer_dirs = infer_dirs[int(args.start):int(args.end)+1]
    else:
        infer_dirs = infer_dirs[int(args.start):]

    # make sure all input directories have the required input images
    my_params = Params(my_param_files[0])
    data_prefixes = [str(item) for item in my_params.data_prefix]
    compl_infer_dirs = []
    for inf_dir in infer_dirs:
        if all([glob(inf_dir + '/*' + prefix + '.nii.gz') for prefix in data_prefixes]):
            compl_infer_dirs.append(inf_dir)
        else:
            print("Skipping {} which does not have all the required images.".format(inf_dir))

    # do work
    output_names = batch_mask(compl_infer_dirs, my_param_files, args.best_last, args.out_dir, overwrite=args.overwrite)
