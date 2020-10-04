import argparse
import logging
import os
from utilities.utils import Params, set_logger
from utilities.patch_input_fn import patch_input_fn
from model.model_fn import model_fn
from glob import glob
import nibabel as nib
import numpy as np
from utilities.input_fn_util import reconstruct_infer_patches, reconstruct_infer_patches_3d


# set tensorflow logging to FATAL before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


# take raw predictions and convert to nifti file
def predictions_2_nii(predictions, infer_dir, out_dir, params, mask=None):
    # load one of the original images to restore original shape and to use for masking
    nii1 = nib.load(glob(infer_dir + '/*' + params.data_prefix[0] + '*.nii.gz')[0])
    affine = nii1.affine
    shape = np.array(nii1.shape)
    name_prefix = os.path.basename(infer_dir[0:-1] if infer_dir.endswith('/') else infer_dir)

    # handle 2 and 2.5 dimensional inference
    if params.dimension_mode in ['2D', '2.5D']:
        # handle 2D with patches using 2D patch reconstructor
        if params.dimension_mode == '2D' and any([overlap > 1 for overlap in params.infer_patch_overlap]):
            predictions = reconstruct_infer_patches(predictions, infer_dir, params)
        # handle 2.5D - take only middle z slice from each slab
        if params.dimension_mode == '2.5D':
            # handle channels last [b, x, y, z, c]
            if params.data_format == 'channels_last':
                predictions = predictions[:, :, :, int(round(predictions.shape[3] / 2.)), :]
            # handle channels first [b, c, x, y, z]
            elif params.data_format == 'channels_first':
                predictions = predictions[:, :, :, :, int(round(predictions.shape[3] / 2))]
            else:
                raise ValueError("Did not understand data format: " + str(params.data_format))

        # at this point all 2 and 2.5 d data is in this format: [b, x, y, c] or [b, c, x, y]
        if not len(predictions.shape) == 4:
            raise ValueError("Predictions is wrong shape in prediction.py")

        # convert to [x, y, b, c], note that b is the z dim since 1 batch was used per slice for inferring
        permute = [2, 3, 0, 1] if params.data_format == 'channels_first' else [1, 2, 0, 3]
        predictions = np.transpose(predictions, axes=permute)

        # convert back to axial leaving channels in final position
        if params.data_plane == 'ax':
            pass  # already in axial
        elif params.data_plane == 'cor':
            predictions = np.transpose(predictions, axes=(0, 2, 1, 3))
        elif params.data_plane == 'sag':
            predictions = np.transpose(predictions, axes=(2, 0, 1, 3))
        else:
            raise ValueError("Did not understand specified plane: " + str(params.data_plane))

        # crop back to original shape (same as input data) - this reverses tensorflow extract patches padding
        pred_shape = np.array(predictions.shape[:3])
        pads = pred_shape - shape
        if any(item < 0 for item in pads):
            assert ValueError("Some dimensions of output are smaller than inputs:)"
                              + " output shape = {} input shape = {}".format(pred_shape, shape))
        predictions = predictions[
                      int(np.floor(pads[0] / 2.)):pred_shape[0] - int(np.ceil(pads[0] / 2.)),
                      int(np.floor(pads[1] / 2.)):pred_shape[1] - int(np.ceil(pads[1] / 2.)),
                      int(np.floor(pads[2] / 2.)):pred_shape[2] - int(np.ceil(pads[2] / 2.)),
                      :]  # do not pad channels dim

    # handle 3D inference
    elif params.dimension_mode == '3D':
        predictions = reconstruct_infer_patches_3d(predictions, infer_dir, params)
    else:
        raise ValueError("Dimension mode must be in [2D, 2.5D, 3D] but is: " + str(params.dimension_mode))

    # mask predictions based on provided mask
    if mask:
        mask_nii = glob(infer_dir + '/*' + mask + '.nii.gz')[0]
        mask_img = nib.load(mask_nii).get_fdata() > 0
        predictions = np.squeeze(predictions) * mask_img

    # convert to nifti format and save
    model_name = os.path.basename(params.model_dir)
    nii_out = os.path.join(out_dir, name_prefix + '_predictions_' + model_name + '.nii.gz')
    img = nib.Nifti1Image(predictions, affine)
    logging.info("Saving predictions to: " + nii_out)
    nib.save(img, nii_out)
    if not os.path.isfile(nii_out):
        raise ValueError("Output nii could not be created at {}".format(nii_out))

    return nii_out


# predict a batch of input directories
def predict(params, pred_dirs, out_dir, mask=None, best_last='last'):

    # load latest checkpoint
    checkpoint_path = os.path.join(params.model_dir, 'checkpoints')
    checkpoints = glob(checkpoint_path + '/*.hdf5')
    if checkpoints:
        # load best or last checkpoint
        # determine last by timestamp
        if best_last == 'last':
            ckpt = max(checkpoints, key=os.path.getctime)
        # determine best by minimum loss value in filename
        elif best_last == 'best':
            vals = [float(item[0:-5].split('_')[-1]) for item in checkpoints]
            ckpt = checkpoints[np.argmin(vals)]
        else:
            raise ValueError("Did not understand best_last value: {}".format(args.best_last))
        # net_builder input layer uses train_dims, so set these to infer dims to allow different size inference
        params.train_dims = params.infer_dims
        # batch size for inference is hard-coded to 1
        params.batch_size = 1
        # recreate the model using infer dims as input dims
        logging.info("Creating the model...")
        model = model_fn(params)
        # load weights from last checkpoint
        logging.info("Loading {} checkpoint from {}...".format(best_last, ckpt))
        model.load_weights(ckpt)
    else:
        raise ValueError("No model checkpoints found at {}".format(checkpoint_path))

    # infer directories in a loop
    niis_out = []
    for pred_dir in pred_dirs:
        # define expected output file name to check if output prediction already exists
        model_name = os.path.basename(params.model_dir)
        name_prefix = os.path.basename(pred_dir)
        pred_out = os.path.join(out_dir, name_prefix + '_predictions_' + model_name + '.nii.gz')
        # if output doesn't already exist, then predict and make nii
        if not os.path.isfile(pred_out):
            # Create the inference dataset structure
            infer_inputs = patch_input_fn(params=params, mode='infer', infer_dir=pred_dir)
            # predict
            predictions = model.predict(infer_inputs)
            # save nii
            pred_out = predictions_2_nii(predictions, pred_dir, out_dir, params, mask=mask)
        # update list of output niis
        niis_out.append(pred_out)

    return niis_out


# executed  as script
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_file', default=None,
                        help="Path to params.json")
    parser.add_argument('-d', '--data_dir', default=None,
                        help="Path to directory to generate inference from")
    parser.add_argument('-b', '--best_last', default='last',
                        help="'best' or 'last' - whether to use best or last weights for inference")
    parser.add_argument('-m', '--mask', default=None,
                        help="Optionally specify a filename prefix for a mask to mask the predictions")
    parser.add_argument('-o', '--out_dir', default=None,
                        help="Optionally specify output directory")

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

    # handle data_dir argument
    # get all subdirectories in data_dir
    study_dirs = [item for item in glob(args.data_dir + '/*/') if os.path.isdir(item)]
    # make sure all necessary files are present in each folder
    study_dirs = [study for study in study_dirs if all(
        [glob('{}/*{}.nii.gz'.format(study, item)) and os.path.isfile(glob('{}/*{}.nii.gz'.format(study, item))[0])
         for item in args.data_prefix + args.label_prefix])]
    # study dirs sorted in alphabetical order for reproducible results
    study_dirs.sort()

    # handle best_last argument
    if args.best_last not in ['best', 'last']:
        raise ValueError("Did not understand best_last value: {}".format(args.best_last))

    # handle mask argument
    if args.mask:
        mask_niis = [glob(study_dir + '/*' + args.mask + '.nii.gz')[0] for study_dir in study_dirs]
        if not all(os.path.isfile(item) for item in mask_niis):
            raise ValueError("Specified mask prefix is not present for all studies in data_dir: {}".format(args.mask))

    # handle out_dir argument
    if args.out_dir:
        assert os.path.isdir(args.out_dir), "Specified output directory does not exist: {}".format(args.out_dir)
    else:
        args.out_dir = os.path.join(os.path.dirname(args.param_file), 'prediction')
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)

    # set up logger, delete old log file if overwrite param is set to yes
    log_path = os.path.join(args.out_dir, 'predict.log')
    if os.path.isfile(log_path) and my_params.overwrite == 'yes':
        os.remove(log_path)
    set_logger(log_path)
    logging.info("Log file created at " + log_path)

    # make predictions
    pred = predict(my_params, study_dirs, args.out_dir, mask=args.mask, best_last=args.best_last)
