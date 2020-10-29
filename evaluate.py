"""Evaluate a trained model"""

import argparse
import logging
import os
import nibabel as nib
import numpy as np
import json
import subprocess
from glob import glob
# set tensorflow logging to FATAL before importing things that contain tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from predict import predict
from utilities.eval_metrics import metric_picker
from utilities.patch_input_fn import get_study_dirs, train_test_split
from utilities.utils import Params, set_logger
from utilities.input_fn_util import normalize


# globals
EPSILON = 1e-10  # small number to avoid divide by zero errors


# define funiction to predict and evaluate one directory
def eval_pred(params, eval_dirs, pred_niis, out_dir, mask, metrics, verbose=False, redo=False, suffix=None):
    # get sorted lists of true niis, predicted niis, and masks
    true_niis = sorted([glob(eval_d + '/*' + params.label_prefix[0] + '.nii.gz')[0] for eval_d in eval_dirs])
    pred_niis = sorted(pred_niis)
    if mask:
        mask_niis = sorted([glob(eval_d + '/*' + mask + '.nii.gz')[0] for eval_d in eval_dirs])
    else:
        mask_niis = []

    # add crop to mask to allow for smaller brain masks to be used
    # ExtractRegionFromImageByMask
    # Extract a sub-region from image using the bounding box from a label image, with optional padding radius.
    # Usage : ExtractRegionFromImageByMask ImageDimension inputImage outputImage labelMaskImage [label=1] [padRadius=0]
    if mask:
        # report
        logging.info("Eval mask specified - cropping files to mask")
        # temporary output vars
        tmp_true = []
        tmp_pred = []
        tmp_mask = []
        # loop through all eval niis
        idx = 1
        for true_nii, pred_nii, mask_nii in zip(true_niis, pred_niis, mask_niis):
            logging.info("Cropping images to eval mask {} of {}...".format(idx, len(pred_niis)))
            # defined cropped outnames
            crop_true_nii = os.path.join(out_dir,
                                         os.path.basename(true_nii).split('.nii.gz')[0] + '_' + mask + '.nii.gz')
            crop_pred_nii = os.path.join(out_dir,
                                         os.path.basename(pred_nii).split('.nii.gz')[0] + '_' + mask + '.nii.gz')
            crop_mask_nii = os.path.join(out_dir,
                                         os.path.basename(mask_nii).split('.nii.gz')[0] + '_crop.nii.gz')
            # crop if not already done
            # true nii
            if not os.path.isfile(crop_true_nii) or redo:
                cmd = "ExtractRegionFromImageByMask 3 {} {} {} 1 0".format(true_nii, crop_true_nii, mask_nii)
                if verbose:
                    logging.debug(cmd)
                subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            else:
                logging.info(
                    "Cropped true image already exists at {} and will not be overwritten".format(crop_true_nii))
            # pred nii
            if not os.path.isfile(crop_pred_nii) or redo:
                cmd = "ExtractRegionFromImageByMask 3 {} {} {} 1 0".format(pred_nii, crop_pred_nii, mask_nii)
                if verbose:
                    logging.debug(cmd)
                subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            else:
                logging.info(
                    "Cropped predicted image already exists at {} and will not be overwritten".format(crop_pred_nii))
            # mask nii
            if not os.path.isfile(crop_mask_nii) or redo:
                cmd = "ExtractRegionFromImageByMask 3 {} {} {} 1 0".format(mask_nii, crop_mask_nii, mask_nii)
                if verbose:
                    logging.debug(cmd)
                subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            else:
                logging.info("Cropped mask already exists at {} and will not be overwritten".format(crop_mask_nii))
            # add cropped niis to output lists
            tmp_true.append(str(crop_true_nii))
            tmp_pred.append(str(crop_pred_nii))
            tmp_mask.append(str(crop_mask_nii))
            # increment index
            idx += 1
        # at end of loop, replace nii lists with new cropped lists
        true_niis = tmp_true
        pred_niis = tmp_pred
        mask_niis = tmp_mask

    # set up metrics variables for collecting results
    metrics_dict = {}
    # loop through each pair of niis for comparissons
    for ind, pred_nii in enumerate(pred_niis):
        logging.info("Evaluating directory {} of {}...".format(ind+1, len(pred_niis)))

        # if normalization is specified in params, then normalize the true nii and save result to eval directory
        if params.norm_labels:
            # load true image
            true_nii = nib.load(true_niis[ind])
            true_im = true_nii.get_fdata()
            # load mask
            if mask:
                mask_im = nib.load(mask_niis[ind]).get_fdata()
                true_im = true_im * (mask_im > 0.)
            # do normalization
            true_im = normalize(true_im, mode=params.norm_mode)
            # save results in eval directory and update true_im name
            new_true_nii = nib.Nifti1Image(true_im, true_nii.affine, true_nii.header)
            true_nii_out = os.path.join(out_dir, os.path.basename(true_niis[ind]).split('.nii.gz')[0] + 'n.nii.gz')
            nib.save(new_true_nii, true_nii_out)
            true_nii = true_nii_out
        else:
            true_nii = true_niis[ind]

        # handle optional mask argument
        if mask:
            mask_nii = mask_niis[ind]
        else:
            mask_nii = None

        # get metric values
        tmp = {pred_nii: {}}
        for metric in metrics:
            # calculate metric - if calculation fails return nan
            try:
                metric_val = metric_picker(metric, true_nii, pred_nii, mask_nii, mask=mask, verbose=verbose)
            except:
                logging.warning("Unable to calculate metric {} for predicted image {}".format(metric, pred_nii))
                metric_val = float("nan")
            tmp[pred_nii].update({metric: metric_val})

        # update metrics dict
        metrics_dict.update(tmp)

    # get metric averages
    metrics_dict.update({'Averages': {}})
    for metric in metrics:
        metric_avg = np.nanmean([metrics_dict[item][metric] for item in metrics_dict.keys() if not item == 'Averages'])
        metrics_dict['Averages'].update({metric: metric_avg})

    # save metrics
    # use prefix string to ID which mask was used for evaluation
    if mask:
        mask_str = mask
    else:
        mask_str = 'no_mask'
    # handle extra suffix option
    if suffix:
        metrics_filepath = os.path.join(out_dir, "{}_{}_eval_metrics.json".format(mask_str, suffix))
    else:
        metrics_filepath = os.path.join(out_dir, "{}_eval_metrics.json".format(mask_str))
    # write output json
    with open(metrics_filepath, 'w+', encoding='utf-8') as fi:
        json.dump(metrics_dict, fi, ensure_ascii=False, indent=4)

    return metrics_dict


# executed  as script
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_file', default=None,
                        help="Path to params.json")
    parser.add_argument('-o', '--out_dir', default=None,
                        help="Optionally specify output directory")
    parser.add_argument('-m', '--mask', default=None,
                        help="Whole brain mask for masking out predictions background")
    parser.add_argument('-e', '--eval_mask', default=None,
                        help="Optionally specify a mask nii prefix for evaluation. All values > 0 are included in mask")
    parser.add_argument('-v', '--verbose', default=False, action="store_true",
                        help="Verbose terminal output flag")
    parser.add_argument('-t', '--metrics', default=['cc', 'mi', 'ssim', 'mse', 'nrmse', 'smape', 'logac', 'medsymac'],
                        nargs='+', help="Metric or metrics to be evaluated (can specify multiple)")
    parser.add_argument('-c', '--checkpoint', default='best',
                        help="'best' or 'last' - whether to use best or last weights for inference")
    parser.add_argument('-x', '--overwrite', default=False, action="store_true",
                        help="Use this flag to overwrite existing data")
    parser.add_argument('-l', '--baseline', default=None,
                        help="Prefix for image contrast to use in place of predictions")
    parser.add_argument('-r', '--rename', default=None,
                        help="Optionally rename the base folder for study dirs (useful when trained in production env)")

    # handle param argument
    args = parser.parse_args()
    assert args.param_file, "Must specify param file using --param_file"
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
    my_params = Params(args.param_file)
    # turn of distributed strategy and mixed precision
    my_params.dist_strat = None
    my_params.mixed_precision = False

    # determine model dir
    if my_params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        my_params.model_dir = os.path.dirname(args.param_file)
    if not os.path.isdir(my_params.model_dir):
        raise ValueError("Specified model directory does not exist")

    # handle out_dir argument
    if args.out_dir:
        assert os.path.isdir(args.out_dir), "Specified output directory does not exist: {}".format(args.out_dir)
    else:
        args.out_dir = os.path.join(os.path.dirname(args.param_file), 'evaluation')
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)

    # set up logger, delete old log file if overwrite param is set to yes
    log_path = os.path.join(args.out_dir, 'evaluate.log')
    if os.path.isfile(log_path) and my_params.overwrite == 'yes':
        os.remove(log_path)
    set_logger(log_path)
    logging.info("Log file created at " + log_path)

    # get list of study valid study directories - optionally change base directory name
    study_dirs = get_study_dirs(my_params, change_basedir=args.rename)

    # separate eval dirs from list of all study dirs using train fraction (same function used by train.py)
    _, my_eval_dirs = train_test_split(study_dirs, my_params)

    # predict output niis
    niis_pred = predict(my_params, my_eval_dirs, args.out_dir, mask=args.mask, checkpoint=args.checkpoint)

    # handle baseline predictions
    if args.baseline:
        niis_pred = [glob("{}/*{}.nii.gz".format(eval_dir, args.baseline))[0] for eval_dir in my_eval_dirs if
                     os.path.isfile(glob("{}/*{}.nii.gz".format(eval_dir, args.baseline))[0])]

    # evaluate
    my_metrics_dict = eval_pred(my_params, my_eval_dirs, niis_pred, args.out_dir, args.eval_mask, args.metrics,
                                verbose=args.verbose, redo=args.overwrite, suffix=args.baseline)

    # report
    for my_metric in my_metrics_dict['Averages'].keys():
        logging.info("Average {} = {}".format(my_metric, my_metrics_dict['Averages'][my_metric]))
