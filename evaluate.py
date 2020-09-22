import argparse
import logging
import os
# set tensorflow logging to FATAL before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from utilities.utils import Params
from glob import glob
import nibabel as nib
import numpy as np
from utilities.input_fn_util import normalize
import json
from predict import predict, predictions_2_nii
from nipype.interfaces.ants import MeasureImageSimilarity


# define funiction to predict and evaluate one directory
def eval_pred(params, eval_dirs, pred_niis, out_dir, mask, mask_val):
    # get sorted lists of true niis, predicted niis, and masks
    true_niis = sorted([glob(eval_d + '/*' + params.label_prefix[0] + '.nii.gz')[0] for eval_d in eval_dirs])
    pred_niis = sorted(pred_niis)
    if mask:
        mask_niis = sorted([glob(eval_d + '/*' + mask + '.nii.gz')[0] for eval_d in eval_dirs])
    else:
        mask_niis = []

    # set up metrics variables for collecting results
    metrics_dict = {}
    # loop through each pair of niis for comparissons
    for ind, pred_nii in enumerate(pred_niis):
        print("Evaluating directory {} of {}...".format(ind+1, len(pred_niis)))

        # if normalization is specified, then normalize the true nii
        if params.norm_labels:
            # load true image
            true_nii = nib.load(true_niis[ind])
            true_im = true_nii.get_fdata()
            # load mask
            if mask:
                mask_im = nib.load(mask_niis[ind]).get_fdata()
                if mask_val == 0:
                    nonzero_inds = np.nonzero(mask_im * true_im)
                else:
                    nonzero_inds = np.nonzero((mask_im == mask_val) * true_im)
            else:
                nonzero_inds = np.nonzero(true_im)
            # do normalization
            true_im = normalize(true_im, mode=params.norm_mode)[nonzero_inds]
            # save results in eval directory and update true_im name
            new_true_nii = nib.Nifti1Image(true_im, true_nii.affine, true_nii.header)
            true_nii_out = os.path.join(out_dir, os.path.basename(true_niis[ind]).split('.nii.gz')[0] + 'n.nii.gz')
            nib.save(new_true_nii, true_nii_out)
            true_nii = new_true_nii
        else:
            true_nii = true_niis[ind]

        # perform image comparrison using CC
        sim = MeasureImageSimilarity()
        sim.inputs.dimension = 3
        sim.inputs.metric = 'CC'
        sim.inputs.fixed_image = true_nii
        sim.inputs.moving_image = pred_nii
        sim.inputs.metric_weight = 1.0
        sim.inputs.radius_or_number_of_bins = 4
        sim.inputs.sampling_strategy = 'None'  # None = dense sampling
        sim.inputs.sampling_percentage = 1.0
        if mask:
            sim.inputs.fixed_image_mask = mask_niis[ind]
            sim.inputs.moving_image_mask = mask_niis[ind]
        print(sim.cmdline)
        cc = sim.run()

        # MI
        sim.inputs.metri = 'MI'
        sim.inputs.radius_or_number_of_bins = 32
        mi = sim.run()

        # MSE
        sim.inputs.metric = 'MeanSquares'
        sim.inputs.radius_or_number_of_bins = 4
        ms = sim.run()

        # build dict
        tmp = {pred_nii: {'CC': cc, 'MI' : mi, 'MSE' : ms}}

        # update metrics dict
        metrics_dict.update(tmp)

    # save metrics
    # use prefix string to ID which mask was used for evaluation
    if mask:
        mask_str = mask + '_{}'.format(mask_val)
    else:
        mask_str = 'no_mask'
    metrics_filepath = os.path.join(out_dir, mask_str + '_eval_metrics.json')
    with open(metrics_filepath, 'w+', encoding='utf-8') as fi:
        json.dump(metrics_dict, fi, ensure_ascii=False, indent=4)

    # print averages
    cc_avg = np.mean([metrics_dict[item]['CC'] for item in metrics_dict.keys()])
    mi_avg = np.mean([metrics_dict[item]['MI'] for item in metrics_dict.keys()])
    ms_avg = np.mean([metrics_dict[item]['MSE'] for item in metrics_dict.keys()])
    metrics = [cc_avg, mi_avg, ms_avg]
    print("Mean error: CC = {}, MI = {}, MSE = {}".format(cc_avg, mi_avg, ms_avg))

    return metrics


# executed  as script
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_file', default=None,
                        help="Path to params.json")
    parser.add_argument('-o', '--out_dir', default=None,
                        help="Optionally specify output directory")
    parser.add_argument('-m', '--mask', default=None,
                        help="Optionally specify a mask nii prefix for evaluation")
    parser.add_argument('-v', '--mask_val', default=0,
                        help="Optionally specify a specific value for the mask nii for evaluation calculations. " +
                             "Default 0 uses all values >0. Predictions will be masked for mask>0 regardless.")
    parser.add_argument('-x', '--overwrite', default=False, action="store_true",
                        help="Use this flag to overwrite existing data")

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

    # get list of evaluation directories from model dir
    study_dirs_filepath = os.path.join(my_params.model_dir, 'study_dirs_list.json')
    if os.path.isfile(study_dirs_filepath):  # load study dirs file if it exists
        with open(study_dirs_filepath) as f:
            study_dirs = json.load(f)
    else:
        raise ValueError("Study directory file does not exist at {}".format(study_dirs_filepath))
    my_eval_dirs = study_dirs[int(round(my_params.train_fract * len(study_dirs))):]

    # do work
    # predit each output if it doesn't already exist
    niis_pred = []
    model_name = os.path.basename(my_params.model_dir)
    for i, eval_dir in enumerate(my_eval_dirs):
        if eval_dir[-1] == '/':
            eval_dir = eval_dir[0:-1]  # remove possible trailing slash
        name_prefix = os.path.basename(eval_dir)
        pred_out = os.path.join(args.out_dir, name_prefix + '_predictions_' + model_name + '.nii.gz')
        print("Predicting directory {} of {}...".format(int(i+1), len(my_eval_dirs)))
        if not os.path.isfile(pred_out) or args.overwrite:
            pred = predict(my_params, eval_dir)
            _ = predictions_2_nii(pred, eval_dir, args.out_dir, my_params, mask=args.mask)
        else:
            print("Prediction {} already exists and will not be overwritten".format(pred_out))
        niis_pred.append(pred_out)

    # evaluate
    my_metrics = eval_pred(my_params, my_eval_dirs, niis_pred, args.out_dir, args.mask, args.mask_val)
    print(np.mean(my_metrics))
