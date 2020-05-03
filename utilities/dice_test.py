import tensorflow as tf
import nibabel as nib
import os
import numpy as np

data_format = 'channels_last'


def dice_loss(labels, predictions):
    # convert labels to int and softmax pred
    y = tf.cast(labels, tf.float32)
    y_pred = tf.nn.softmax(predictions, axis=-1 if data_format == 'channels_last' else 1)
    # do dice calculation
    mean = 0.5  # the weight is the proportion of 1s to 0s in the label set
    w_1 = 1. / mean ** 2.
    w_0 = 1. / (1. - mean) ** 2.
    y_true_f_1 = tf.reshape(y, [-1])
    y_pred_f_1 = tf.reshape(y_pred[..., 1] if data_format == 'channels_last' else y_pred[:, 1, ...], [-1])
    y_true_f_0 = tf.reshape(1 - y, [-1])
    y_pred_f_0 = tf.reshape(y_pred[..., 0] if data_format == 'channels_last' else y_pred[:, 0, ...], [-1])
    int_0 = tf.reduce_sum(y_true_f_0 * y_pred_f_0)
    int_1 = tf.reduce_sum(y_true_f_1 * y_pred_f_1)
    dice = 2. * (w_0 * int_0 + w_1 * int_1) / (
            (w_0 * (tf.reduce_sum(y_true_f_0) + tf.reduce_sum(y_pred_f_0))) +
            (w_1 * (tf.reduce_sum(y_true_f_1) + tf.reduce_sum(y_pred_f_1))))
    loss_function = 1. - dice
    return loss_function

# load a set of labels
lab = nib.load('/media/ecalabr/scratch/test/00000116/00000116_combined_brain_mask.nii.gz')
lab_data = lab.get_fdata()

lab_slice = np.expand_dims(lab_data[:, :, round(lab_data.shape[2]/2)], axis=-1)
lab_batch = np.stack((lab_slice, lab_slice, lab_slice, lab_slice, lab_slice, lab_slice, lab_slice, lab_slice))
pred_batch = np.concatenate((1000*np.abs(1-lab_batch), 1000*lab_batch), axis=-1)
lab_tf = tf.constant(lab_batch, dtype=tf.float32)
pred_tf = tf.constant(pred_batch, dtype=tf.float32)
loss = dice_loss(lab_tf, pred_tf)
print(loss)
