"""Tensorflow utility functions for prediction"""

import os
import tensorflow as tf
import numpy as np
import nibabel as nib
import logging
import time
from glob import glob


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
    n=1
    while True:
        try:
            prediction = sess.run(model_spec['predictions'])
            n = n+1
            if type(predictions) != np.ndarray:
                start = time.time()
                predictions = prediction
                logging.info("Processing slice " + str(n) + " took " + str(time.time()-start) + " seconds")
            else:
                start = time.time()
                predictions = np.concatenate([predictions, prediction])  # concatenates on axis=0 by default
                logging.info("Processing slice " + str(n) + " took " + str(time.time() - start) + " seconds")
        except tf.errors.OutOfRangeError:
            break

    return predictions


def predict(model_spec, model_dir, params, infer_dir):
    """Evaluate the model
    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
        infer_dir: (str) the path to the directory for inference
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

        # Reload last weights from the weights subdirectory
        save_path = os.path.join(model_dir, 'last_weights')
        logging.info("Restoring parameters from {}".format(save_path))
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # predict
        predictions = predict_sess(sess, model_spec)

    # load one of the original images to restore original shape
    if infer_dir[-1] == '/': infer_dir = infer_dir[0:-1]  # remove possible trailing slash
    nii = nib.load(glob(infer_dir + '/*' + params.data_prefix[0] + '*.nii.gz')[0])
    affine = nii.affine
    shape = nii.shape
    name_prefix = os.path.basename(infer_dir)

    # make predictions the correct shape (same as input data)
    permute = [2, 3, 0, 1] if params.data_format == 'channels_first' else [1, 2, 0, 3]
    predictions = np.squeeze(np.transpose(predictions, axes=permute))
    pads = ((0, 0), (0, 0), (0, shape[2]-predictions.shape[-1]))
    predictions = np.pad(predictions[0:shape[0], 0:shape[1], :], pads, 'constant')

    # mask predictions based on input data
    mask = nii.get_data() > 0
    predictions = predictions * mask

    # convert to nifti format and save
    nii_out = os.path.join(pred_dir, name_prefix + '_predictions.nii.gz')
    img = nib.Nifti1Image(predictions, affine)
    logging.info("Saving predictions to: " + nii_out)
    nib.save(img, nii_out)
