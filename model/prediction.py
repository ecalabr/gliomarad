"""Tensorflow utility functions for prediction"""

import os
import tensorflow as tf
import numpy as np
import nibabel as nib
import logging
import time
from glob import glob
from patch_input_fn import reconstruct_infer_patches_3d
from patch_input_fn import reconstruct_infer_patches


def predict_sess(sess, model_spec):
    """Do some predictions.
    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
    """

    # initialize iterator
    sess.run(model_spec['iterator_init_op'])

    # compute predictions over the dataset
    predictions = []
    n=0
    while True:
        try:
            prediction = sess.run(model_spec['predictions'])
            n = n+1
            if type(predictions) != np.ndarray:
                start = time.time()
                predictions = prediction
                logging.info("Processing chunk " + str(n) + " took " + str(time.time()-start) + " seconds")
            else:
                start = time.time()
                predictions = np.concatenate([predictions, prediction])  # concatenates on axis=0 by default
                logging.info("Processing chunk " + str(n) + " took " + str(time.time() - start) + " seconds")
        except tf.errors.OutOfRangeError:
            break

    return predictions


def predict(model_spec, model_dir, params, infer_dir, best_last):
    """Evaluate the model
    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
        infer_dir: (str) the path to the directory for inference
        best_last: (str) in ['best_weights', 'last_weights'] - whether to use best or last model weights for inference
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
    """

    # make sure we are not overwriting something important
    pred_dir = os.path.join(model_dir, 'predictions')
    if not os.path.isdir(pred_dir):
        os.mkdir(pred_dir)
    if params.overwrite != 'yes' and os.listdir(pred_dir):
        raise ValueError("Overwrite param is not 'yes' but there are files in the predictions directory!")

    # Initialize tf.Saver
    saver = tf.train.Saver()

    # load model and run predictions
    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload best weights from the weights subdirectory
        save_path = os.path.join(model_dir, best_last)
        logging.info("Restoring parameters from {}".format(save_path))
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # predict
        predictions = predict_sess(sess, model_spec)

    # load one of the original images to restore original shape and to use for masking
    if infer_dir[-1] == '/': infer_dir = infer_dir[0:-1]  # remove possible trailing slash
    nii = nib.load(glob(infer_dir + '/*' + params.data_prefix[0] + '*.nii.gz')[0])
    affine = nii.affine
    shape = np.array(nii.shape)
    name_prefix = os.path.basename(infer_dir)

    # handle 2 and 2.5 dimensional inference
    if params.dimension_mode in ['2D', '2.5D']:
        # handle 2D with patches using 2D patch reconstructor
        if params.dimension_mode == '2D' and any([overlap > 1 for overlap in params.infer_patch_overlap]):
            predictions = reconstruct_infer_patches(predictions, infer_dir, params)
        # handle 2.5D - take only middle slice from each slab
        if params.dimension_mode == '2.5D':
            # handle channels last [b, x, y, z, c]
            if params.data_format == 'channels_last':
                predictions = predictions[:, :, :, predictions.shape[3]/2 + 1, :]
            # handle channels first [b, c, x, y, z]
            elif params.data_format == 'channels_first':
                predictions = predictions[:, :, :, :, predictions.shape[3]/2 + 1]
            else:
                raise ValueError("Did not understand data format: " + str(params.data_format))

        # handle multiple predictions
        if params.data_format == 'channels_first':
            if predictions.shape[1] > 1:
                predictions = np.expand_dims(predictions[:, 0, :, :], axis=1)
        if params.data_format == 'channels_last':
            if predictions.shape[-1] > 1:
                predictions = np.expand_dims(predictions[:, :, :, 0], axis=-1)

        # convert to batch dim last and squeeze channels dim
        permute = [2, 3, 0, 1] if params.data_format == 'channels_first' else [1, 2, 0, 3]
        predictions = np.squeeze(np.transpose(predictions, axes=permute))

        # convert back to axial
        if params.data_plane == 'ax':
            pass
        elif params.data_plane == 'cor':
            predictions = np.transpose(predictions, axes=(0, 2, 1))
        elif params.data_plane == 'sag':
            predictions = np.transpose(predictions, axes=(2, 0, 1))
        else:
            raise ValueError("Did not understand specified plane: " + str(params.data_plane))

        # crop back to original shape (same as input data) - this reverses tensorflow extract patches padding
        pred_shape = np.array(predictions.shape)
        pads = pred_shape - shape
        predictions = predictions[int(np.floor(pads[0]/2.)):pred_shape[0]-int(np.ceil(pads[0]/2.)),
                      int(np.floor(pads[1]/2.)):pred_shape[1]-int(np.ceil(pads[1]/2.)),
                      int(np.floor(pads[2]/2.)):pred_shape[2]-int(np.ceil(pads[2]/2.))]

    # handle 3D inference
    elif params.dimension_mode == '3D':
        predictions = reconstruct_infer_patches_3d(predictions, infer_dir, params)
    else:
        raise ValueError("Dimension mode must be in [2D, 2.5D, 3D] but is: " + str(params.dimension_mode))

    # mask predictions based on original input data
    mask = nii.get_data() > 0
    predictions = predictions * mask

    # convert to nifti format and save
    nii_out = os.path.join(pred_dir, name_prefix + '_predictions_' + best_last + '.nii.gz')
    img = nib.Nifti1Image(predictions, affine)
    logging.info("Saving predictions to: " + nii_out)
    nib.save(img, nii_out)
