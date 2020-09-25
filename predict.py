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

# define funiction to predict one inference directory
def predict(params, infer_dir):

    # Set the logger, delete old log file if overwrite param is set to yes
    log_path = os.path.join(params.model_dir, 'predict.log')
    if os.path.isfile(log_path) and params.overwrite == 'yes':
        os.remove(log_path)
    set_logger(log_path)
    logging.info("- Log file created at " + log_path)

    # Create the inference dataset structure
    infer_inputs = patch_input_fn(params=params, mode='infer', infer_dir=infer_dir)

    # load latest checkpoint
    checkpoint_path = os.path.join(params.model_dir, 'checkpoints')
    checkpoints = glob(checkpoint_path + '/*.hdf5')
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print("- Loading checkpoint from {}...".format(latest_ckpt))
        # net_builder input layer uses train_dims, so set these to infer dims to allow different size inference
        params.train_dims = params.infer_dims
        # batch size for inference is hard-coded to 1
        params.batch_size = 1
        # recreate the model using infer dims as input dims
        model = model_fn(params)
        # load weights from last checkpoint
        model.load_weights(latest_ckpt)
        print("- Done loading checkpoint")
    else:
        raise ValueError("No checkpoints found at {}".format(checkpoint_path))

    # infer - assuming that model returns probabilities
    predictions = model.predict(infer_inputs)

    return predictions


def predictions_2_nii(predictions, infer_dir, out_dir, params, mask=None):
    # assumes that model yeilds probabilities
    # load one of the original images to restore original shape and to use for masking
    nii1 = nib.load(glob(infer_dir + '/*' + params.data_prefix[0] + '*.nii.gz')[0])
    affine = nii1.affine
    shape = np.array(nii1.shape)
    name_prefix = os.path.basename(infer_dir)

    # handle 2 and 2.5 dimensional inference
    if params.dimension_mode in ['2D', '2.5D']:
        # handle 2D with patches using 2D patch reconstructor
        if params.dimension_mode == '2D' and any([overlap > 1 for overlap in params.infer_patch_overlap]):
            predictions = reconstruct_infer_patches(predictions, infer_dir, params)
        # handle 2.5D - take only middle z slice from each slab
        if params.dimension_mode == '2.5D':
            # handle channels last [b, x, y, z, c]
            if params.data_format == 'channels_last':
                predictions = predictions[:, :, :, int(round(predictions.shape[3]/2.)), :]
            # handle channels first [b, c, x, y, z]
            elif params.data_format == 'channels_first':
                predictions = predictions[:, :, :, :, int(round(predictions.shape[3]/2))]
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
                      int(np.floor(pads[0]/2.)):pred_shape[0]-int(np.ceil(pads[0]/2.)),
                      int(np.floor(pads[1]/2.)):pred_shape[1]-int(np.ceil(pads[1]/2.)),
                      int(np.floor(pads[2]/2.)):pred_shape[2]-int(np.ceil(pads[2]/2.)),
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
    logging.info("- Saving predictions to: " + nii_out)
    nib.save(img, nii_out)
    if not os.path.isfile(nii_out):
        raise ValueError("Output nii could not be created at {}".format(nii_out))

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

    # turn of distributed strategy and mixed precision
    my_params.dist_strat = None
    my_params.mixed_precision = False

    # handle best_last argument
    if args.best_last not in ['best_weights', 'last_weights']:
        raise ValueError("Did not understand best_last value: " + str(args.best_last))

    # handle out_dir argument
    if args.out_dir:
        assert os.path.isdir(args.out_dir), "Specified output directory does not exist: {}".format(args.out_dir)
    else:
        args.out_dir = os.path.join(os.path.dirname(args.param_file), 'predictions')
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)

    # handle inference directory argument
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
    pred = predict(my_params, args.infer_dir)
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    nii = predictions_2_nii(pred, args.infer_dir, args.out_dir, my_params)
