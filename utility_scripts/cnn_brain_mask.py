""" hacky script for making a bunch of brain masks using a CNN """

import argparse
import os
from model.utils import Params
from glob import glob

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--param_file', default=None,
                    help="Path to params.json")
parser.add_argument('--infer_dir', default=None,
                    help="Path to directory to generate inference from")
parser.add_argument('--best_last', default='last_weights',
                    help="Either 'best_weights' or 'last_weights' - whether to use best or last model weights for inference")
parser.add_argument('--out_dir', default=None,
                    help="Optionally specify output directory")

if __name__ == '__main__':
    # handle params argument
    args = parser.parse_args()
    assert args.param_file, "Must specify param file using --param_file"
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
    params = Params(args.param_file)

    # handle best_last argument
    best_last = args.best_last
    if best_last not in ['best_weights', 'last_weights']:
        raise ValueError("Did not understand best_last value: " + str(best_last))

    # handle out_dir argument
    if args.out_dir:
        assert os.path.isdir(args.out_dir), "Specified output directory does not exist: {}".format(args.out_dir)

    # handler inference directory argument
    infer_dir = args.infer_dir
    assert infer_dir, "No infer directory specified. Use --infer_dir="
    assert os.path.isdir(infer_dir), "No inference directory found at {}".format(infer_dir)
    # check if provided dir is a single image dir or a dir full of image dirs
    if glob(infer_dir + '/*' + params.data_prefix[0] + '.nii.gz'):
        infer_dirs = [infer_dir]
    elif glob(infer_dir + '/*/*' + params.data_prefix[0] + '.nii.gz'):
        infer_dirs = sorted(list(set([os.path.dirname(f)
                                      for f in glob(infer_dir + '/*/*' + params.data_prefix[0] + '.nii.gz')])))
    else:
        raise ValueError("No image data found in inference directory: {}".format(infer_dir))

    # run inference and post-processing for each infer_dir
    for direc in infer_dirs:
        # run predict to get the output probabilities
        param = " --param_file='" + args.param_file + "'"
        infer = " --infer_dir='" + direc + "'"
        best = " --best_last='best_weights'"
        out = " --out_dir='" + direc + "'"
        cmd = 'python /home/ecalabr/PycharmProjects/gbm_preproc/predict.py' + param + infer + best + out
        os.system(cmd)

        # check that predictions were made
        probs = glob(direc + "/*_predictions_*.nii.gz")
        if probs:
            if len(probs) == 1:
                probs = probs[0]
            else:
                raise ValueError("More than one probability image found in {}".format(direc))
        else:
            raise ValueError("No probability image found in {}".format(direc))

        # convert probs to mask with cleanup
        data = " --data='" + probs + "'"
        idno = os.path.basename(direc.rsplit('/',1)[0] if direc.endswith('/') else direc)
        outname = " --outname='" + idno + "_combined_brain_mask.nii.gz'"
        outpath = " --outpath='" + direc + "'"
        clean = " --clean"
        cmd = "/home/ecalabr/PycharmProjects/gbm_preproc/utility_scripts/prob2seg.py" + data + outname + outpath + clean
        os.system(cmd)

        # report
        outfile = os.path.join(direc, idno + "_combined_brain_mask.nii.gz")
        if os.path.isfile(outfile):
            print(" created mask file at {}".format(outfile))
        else:
            raise ValueError("No mask output file found at {}".format(direc))