import os
import math
from glob import glob
import numpy as np
import nibabel as nib
import scipy.stats as stats
import scipy.ndimage as ndi
import tensorflow as tf
import random
import logging
import json
import csv


##############################################
# TENSORFLOW MAP FUNCTIONS
##############################################


def tf_patches(data, labels, patch_size, chan_dim, data_format, overlap=1):
    """
    Extract 2D patches from a data array with overlap if desired
    :param data: (numpy array) the data tensorflow tensor
    :param labels: (numpy array) the labels tensorflow tensor
    :param patch_size: (list or tupe of ints) the patch dimensions
    :param chan_dim:  (int) the number of data channels
    :param data_format: (str) either channels_last or channels_first - the tensorflow data format
    :param overlap: (int) the divisor for patch strides - determines the patch overlap in x, y (default no overlap)
    :return: returns tensorflow tensor patches
    """

    # sanity checks
    if not len(patch_size) == 2:
        raise ValueError("Patch size must be shape 2 to use 2D patch function but is: " + str(patch_size))

    # handle overlap int vs list/tuple
    if not isinstance(overlap, (np.ndarray, int, list, tuple)):
        raise ValueError("Overlap must be a list, tuple, array, or int.")
    if isinstance(overlap, int):
        overlap = [overlap] * 2

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 2, 3, 1])
        labels = tf.transpose(a=labels, perm=[0, 2, 3, 1])

    # make patches
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap[0], patch_size[1] / overlap[1], 1]
    strides = [int(round(item)) for item in strides]
    rates = [1, 1, 1, 1]
    data = tf.image.extract_patches(data, sizes=ksizes, strides=strides, rates=rates, padding='SAME')
    data = tf.reshape(data, [-1] + patch_size + [chan_dim])
    labels = tf.image.extract_patches(labels, sizes=ksizes, strides=strides, rates=rates, padding='SAME')
    labels = tf.reshape(labels, [-1] + patch_size + [1])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 3, 1, 2])
        labels = tf.transpose(a=labels, perm=[0, 3, 1, 2])

    return data, labels


def tf_patches_infer(data, patch_size, chan_dim, data_format, overlap=1):
    """
    Extract 2D patches from a data array with overlap if desired - no labels. used for inference
    :param data: (numpy array) the data tensorflow tensor
    :param patch_size: (list or tupe of ints) the patch dimensions
    :param chan_dim:  (int) the number of data channels
    :param data_format: (str) either channels_last or channels_first - the tensorflow data format
    :param overlap: (int or list/tuple of ints) the divisor for patch strides - determines the patch overlap in x, y
    :return: returns tensorflow tensor patches
    """

    # sanity checks
    if not len(patch_size) == 2:
        raise ValueError("Patch size must be shape 2 to use 2D patch function but is: " + str(patch_size))

    # handle overlap int vs list/tuple
    if not isinstance(overlap, (np.ndarray, int, list, tuple)):
        raise ValueError("Overlap must be a list, tuple, array, or int.")
    if isinstance(overlap, int):
        overlap = [overlap] * 2

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 2, 3, 1])

    # make patches
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap[0], patch_size[1] / overlap[1], 1]
    strides = [int(round(item)) for item in strides]
    rates = [1, 1, 1, 1]
    data = tf.image.extract_patches(data, sizes=ksizes, strides=strides, rates=rates, padding='SAME')
    data = tf.reshape(data, [-1] + patch_size + [chan_dim])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 3, 1, 2])

    return data


def tf_patches_3d(data, labels, patch_size, data_format, data_chan, label_chan=1, overlap=1, weighted=False):
    """
    Extract 3D patches from a data array with overlap if desired
    :param data: (numpy array) the data tensorflow tensor
    :param labels: (numpy array) the labels tensorflow tensor
    :param patch_size: (list or tupe of ints) the patch dimensions
    :param data_format: (str) either channels_last or channels_first - the tensorflow data format
    :param data_chan: (int) the number of channels in the feature data
    :param label_chan: (int) the number of channels in the label data
    :param overlap: (int or list/tuple of ints) the divisor for patch strides - determines the patch overlap in x, y
    :param weighted: (bool) weather or not the labels data includes a weight tensor as the last element of labels
    :return: returns tensorflow tensor patches
    """

    # sanity checks
    if not len(patch_size) == 3:
        raise ValueError("Patch size must be shape 3 to use 3D patch function but is: " + str(patch_size))

    # handle overlap int vs list/tuple
    if not isinstance(overlap, (np.ndarray, int, list, tuple)):
        raise ValueError("Overlap must be a list, tuple, array, or int.")
    if isinstance(overlap, int):
        overlap = [overlap] * 3

    # handle channels first by temporarily converting to channels last
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 2, 3, 4, 1])
        labels = tf.transpose(a=labels, perm=[0, 2, 3, 4, 1])

    # for sliding window 3d slabs
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap[0], patch_size[1] / overlap[1], patch_size[2] / overlap[2], 1]
    strides = [int(round(item)) for item in strides]

    # make patches
    data = tf.extract_volume_patches(data, ksizes=ksizes, strides=strides, padding='VALID')
    data = tf.reshape(data, [-1] + patch_size + [data_chan])
    labels = tf.extract_volume_patches(labels, ksizes=ksizes, strides=strides, padding='VALID')
    # if weighted is true, then add 1 to label_chan for the weights tensor that should be concatenated here
    labels = tf.reshape(labels, [-1] + patch_size + [label_chan + 1 if weighted else label_chan])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 4, 1, 2, 3])
        labels = tf.transpose(a=labels, perm=[0, 4, 1, 2, 3])

    return data, labels


def tf_patches_3d_infer(data, patch_size, chan_dim, data_format, overlap=1):
    """
    Extract 3D patches from a data array with overlap if desired - no labels, used for inference
    :param data: (numpy array) the data tensorflow tensor
    :param patch_size: (list or tupe of ints) the patch dimensions
    :param chan_dim:  (int) the number of data channels
    :param data_format: (str) either channels_last or channels_first - the tensorflow data format
    :param overlap: (int) the divisor for patch strides - determines the patch overlap in x, y (default no overlap)
    :return: returns tensorflow tensor patches
    """

    # sanity checks
    if not len(patch_size) == 3:
        raise ValueError("Patch size must be shape 3 to use 3D patch function but is: " + str(patch_size))

    # handle overlap int vs list/tuple
    if not isinstance(overlap, (np.ndarray, int, list, tuple)):
        raise ValueError("Overlap must be a list, tuple, array, or int.")
    if isinstance(overlap, int):
        overlap = [overlap] * 3

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 2, 3, 4, 1])

    # for sliding window 3d slabs
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap[0], patch_size[1] / overlap[1], patch_size[2] / overlap[2], 1]
    strides = [int(round(item)) for item in strides]

    # make patches
    data = tf.extract_volume_patches(data, ksizes=ksizes, strides=strides, padding='SAME')
    data = tf.reshape(data, [-1] + patch_size + [chan_dim])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 4, 1, 2, 3])

    return data


def filter_zero_patches(labels, data_format, mode, thresh=0.05):
    """
    Filters out patches that contain mostly zeros in the label data. Works for 3D and 2D patches.
    :param labels: (tf.tensor) containing labels data (uses only first channel currently)
    :param data_format: (str) either 'channels_first' or 'channels_last' - the tensorflow data format
    :param mode: (str) either '2D' '2.5D' or '3D' - the mode for training
    :param thresh: (float) the threshold percentage for keeping patches. Default is 5%.
    :return: Returns tf.bool False if less than threshold, else returns tf.bool True
    """
    if float(thresh) == 0.:
        return tf.constant(True, dtype=tf.bool)

    if data_format == 'channels_last':
        # handle channels last - use entire slice if 2D, use entire slab if 3D or 2.5D
        if mode == '2.5D':  # [x, y, z, c]
            labels = labels[:, :, int(round(labels.get_shape()[2] / 2.)), 0]
        elif mode == '2D':  # [x, y, c]
            labels = labels[:, :, 0]
        elif mode == '3D':  # [x, y, z, c]
            labels = labels[:, :, :, 0]
        else:
            raise ValueError("Mode must be 2D, 2.5D, or 3D but is: " + str(mode))
    else:
        # handle channels first - use entire slice if 2D, use entire slab if 3D or 2.5D
        if mode == '2.5D':  # [c, x, y, z]
            labels = labels[0, :, :, int(round(labels.get_shape()[2] / 2.))]
        elif mode == '2D':  # [c, x, y]
            labels = labels[0, :, :]
        elif mode == '3D':  # [c, x, y, z]
            labels = labels[0, :, :, :]
        else:
            raise ValueError("Labels shape must be 2D or 3D but is: " + str((labels.get_shape())))

    # make threshold a tf tensor for comparisson
    thr = tf.constant(thresh, dtype=tf.float32)

    # get nonzero fraction
    nf = tf.math.count_nonzero(labels, dtype=tf.float32) / tf.cast(tf.size(input=labels, out_type=tf.int32), tf.float32)

    return tf.less(thr, nf)


# DATA UTILITIES
def load_single_study(study_dir, file_prefixes, data_format, plane=None, norm=False, norm_mode='zero_mean'):
    """
    Image data I/O function for use in tensorflow Dataset map function. Takes a study directory and file prefixes and
    returns a 4D numpy array containing the image data. Performs optional slice trimming in z and normalization.
    :param study_dir: (str) the full path to the study directory
    :param file_prefixes: (str, list(str)) the file prefixes for the images to be loaded
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param norm: (bool) whether or not to perform per dataset normalization
    :param norm_mode: (str) The method for normalization, used by normalize function.
    :return: output - a 4D numpy array containing the image data
    """

    # sanity checks
    if not os.path.isdir(study_dir):
        raise ValueError("Specified study_dir does not exist")
    if data_format not in ['channels_last', 'channels_first']:
        raise ValueError("data_format invalid")
    images = [glob(study_dir + '/*' + contrast + '.nii.gz')[0] for contrast in file_prefixes]
    if not images:
        raise ValueError("No matching image files found for file prefixes: " + str(images))

    # get data dims
    nii = nib.load(images[0])
    data_dims = nii.shape

    # preallocate
    data = np.empty(data_dims + (len(images),), dtype=np.float32)

    # load images and concatenate into a 4d numpy array
    for ind, image in enumerate(images):
        # first nii is already loaded
        if ind > 0:
            nii = nib.load(images[ind])
        # handle normalization
        if norm:
            data[..., ind] = normalize(nii.get_fdata(), norm_mode)
        else:
            data[..., ind] = nii.get_fdata()

    # permute to desired plane in format [x, y, z, channels] for tensorflow
    if plane == 'ax':
        pass
    elif plane == 'cor':
        data = np.transpose(data, axes=(0, 2, 1, 3))
    elif plane == 'sag':
        data = np.transpose(data, axes=(1, 2, 0, 3))
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_format == 'channels_first':
        data = np.transpose(data, axes=(3, 0, 1, 2))

    return data


def expand_region(input_dims, region_bbox, delta):
    """
    Symmetrically expands a given 3D region bounding box by delta in each dim without exceeding original image dims
    If delta is a single int then each dim is expanded by this amount. If a list or tuple, of ints then dims are
    expanded to match the size of each int in the list respectively.
    :param input_dims: (list or tuple of ints) the original image dimensions
    :param region_bbox: (list or tuple of ints) the region bounding box to expand
    :param delta: (int or list/tuple of ints) the amount to expand each dimension by.
    :return: (list or tuple of ints) the expanded bounding box
    """

    # if delta is actually a list, then refer to expand_region_dims, make sure its same length as input_dims
    if isinstance(delta, (list, tuple, np.ndarray)):
        if not len(delta) == len(input_dims):
            raise ValueError("When a list is passed, parameter mask_dilate must have one val for each dim in mask")
        return expand_region_dims(input_dims, region_bbox, delta)

    # if delta is zero return
    if delta == 0:
        return region_bbox

    # determine how much to add on each side of the bounding box
    deltas = np.array([-int(np.floor(delta / 2.)), int(np.ceil(delta / 2.))] * 3)

    # use deltas to get a new bounding box
    tmp_bbox = np.array(region_bbox) + deltas

    # make sure there are not values outside of the original image
    new_bbox = []
    for i, item in enumerate(tmp_bbox):
        if i % 2 == 0:  # for even indices, make sure there are no negatives
            if item < 0:
                item = 0
        else:  # for odd indices, make sure they do not exceed original dims
            if item > input_dims[int(round((i - 1) / 2))]:
                item = input_dims[int(round((i - 1) / 2))]
        new_bbox.append(item)

    return new_bbox


def expand_region_dims(input_dims, region_bbox, out_dims):
    """
    Symmetrically expands a given 3D region bounding box to the specified output size
    :param input_dims: (list or tuple of ints) the original image dimensions
    :param region_bbox: (list or tuple of ints) the region bounding box to expand
    :param out_dims: (list or tuple of ints) the desired output dimensions.
    :return: (list or tuple of ints) the expanded bounding box
    """

    # determine region dimensions
    region_dims = [region_bbox[1] - region_bbox[0], region_bbox[3] - region_bbox[2], region_bbox[5] - region_bbox[4]]
    # region_dims = [region_bbox[x] - region_bbox[x - 1] for x in range(len(region_bbox))[1::2]]

    # determine the delta in each dimension - exclude negatives
    deltas = [x - y for x, y in zip(out_dims, region_dims)]
    deltas = [0 if d < 0 else d for d in deltas]

    # determine how much to add on each side of the bounding box
    pre_inds = np.array([-int(np.floor(d / 2.)) for d in deltas])
    post_inds = np.array([int(np.ceil(d / 2.)) for d in deltas])
    deltas = np.empty((pre_inds.size + post_inds.size,), dtype=pre_inds.dtype)
    deltas[0::2] = pre_inds
    deltas[1::2] = post_inds

    # use deltas to get a new bounding box
    tmp_bbox = np.array(region_bbox) + deltas

    # make sure there are not values outside of the original image
    new_bbox = []
    for i, item in enumerate(tmp_bbox):
        if i % 2 == 0:  # for even indices, make sure there are no negatives
            if item < 0:
                item = 0
        else:  # for odd indices, make sure they do not exceed original dims
            if item > input_dims[int(round((i - 1) / 2))]:
                item = input_dims[int(round((i - 1) / 2))]
        new_bbox.append(item)

    return new_bbox


def create_affine(theta=None, phi=None, psi=None):
    """
    Creates a 3D rotation affine matrix given three rotation angles.
    :param theta: (float) The theta angle in radians. If None, a random angle is chosen.
    :param phi: (float) The phi angle in radians. If None, a random angle is chosen.
    :param psi: (float) The psi angle in radians. If None, a random angle is chosen.
    :return: (np.ndarray) a 3x3 affine rotation matrix.
    """

    # return identitiy if all angles are zero
    if all(val == 0. for val in [theta, phi, psi]):
        return np.eye(3)

    # define angles
    if theta is None:
        theta = np.random.random() * (np.pi / 2.)
    if phi is None:
        phi = np.random.random() * (np.pi / 2.)
    if psi is None:
        psi = np.random.random() * (np.pi / 2.)

    # define affine array
    affine = np.asarray([
        [np.cos(theta) * np.cos(psi),
         -np.cos(phi) * np.sin(psi) + np.sin(phi) * np.sin(theta) * np.cos(psi),
         np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi)],

        [np.cos(theta) * np.sin(psi),
         np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi),
         -np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi)],

        [-np.sin(theta),
         np.sin(phi) * np.cos(theta),
         np.cos(phi) * np.cos(theta)]
    ])

    return affine


def affine_transform(image, affine, offset=None, order=1):
    """
    Apply a 3D rotation affine transform to input_img with specified offset and spline interpolation order.
    :param image: (np.ndarray) The input image.
    :param affine: (np.ndarray) The 3D affine array of shape [3, 3]
    :param offset: (np.ndarray) The offset to apply to the image after rotation, should be shape [3,]
    :param order: (int) The spline interpolation order. Must be 0-5
    :return: The input image after applying the affine rotation and offset.
    """

    # sanity checks
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image should be np.ndarray but is: " + str(type(image)))
    # define affine if it doesn't exist
    if affine is None:
        affine = create_affine()
    if not isinstance(affine, np.ndarray):
        raise TypeError("Affine should be np.ndarray but is: " + str(type(affine)))
    if not affine.shape == (3, 3):
        raise ValueError("Affine should have shape (3, 3)")
    # define offset if it doesn't exist
    if offset is None:
        center = np.array(image.shape)
        offset = center - np.dot(affine, center)
    if not isinstance(offset, np.ndarray):
        raise TypeError("Offset should be np.ndarray but is: " + str(type(offset)))
    if not offset.shape == (3,):
        raise ValueError("Offset should have shape (3,)")

    # Apply affine
    # handle 4d
    if len(image.shape) > 3:
        # make 4d identity matrix and replace xyz component with 3d affine
        affine4d = np.eye(4)
        affine4d[:3, :3] = affine
        offset4d = np.append(offset, 0.)
        image = ndi.interpolation.affine_transform(image, affine4d, offset=offset4d, order=order, output=np.float32)
    # handle 3d
    else:
        image = ndi.interpolation.affine_transform(image, affine, offset=offset, order=order, output=np.float32)

    return image


def affine_transform_roi(image, roi, labels=None, affine=None, dilate=None, order=1, scale_cube_dim=None):
    """
    Function for affine transforming and extracting an roi from a set of input images and labels
    :param image: (np.ndarray) a 3d or 4d input image
    :param roi: (np.ndarray) a 3d mask or segmentation roi
    :param labels: (np.ndarray) a set of 3d labels (optional)
    :param affine: (np.ndarray) an affine transform. if None, a random transform is created
    :param dilate: (np.ndarray) an int or list of 3 ints. If int, each dim is symmetrically expanded by that amount. If
    an array of 3 ints, then each dimension is expanded to the value of the corresponding int.
    :param order: (int) the spline order used for label interpolation
    :param scale_cube_dim: (None or int) If None, has no effect. If an int, this will extract the ROI as a cube with
    width equal to largest roi dimension and will then scale result to a cube with width equal to scale_cube_dim
    :return: (np.ndarray) The transformed and cropped images, rois and optionally labels
    """

    # sanity checks
    if affine is None:
        affine = create_affine()

    # get input tight bbox shape - convert to cubic ROI if specified with scale_cube_dim
    roi_bbox = nonzero_slice_inds3d(roi)

    # dilate if necessary
    if dilate is not None:
        roi_bbox = expand_region(roi.shape, roi_bbox, dilate)

    # scale calculations if using scale_cube_dim
    if scale_cube_dim is not None:
        maxdim = np.max([roi_bbox[1] - roi_bbox[0], roi_bbox[3] - roi_bbox[2], roi_bbox[5] - roi_bbox[4]])
        scale = maxdim / scale_cube_dim
        # to add scale factor to affine, compute dot product of rotation affine and scale affine
        affine = np.dot(affine, np.eye(3) * scale)

    # determine input and output shape
    in_shape = [roi_bbox[1] - roi_bbox[0], roi_bbox[3] - roi_bbox[2], roi_bbox[5] - roi_bbox[4]]
    if scale_cube_dim is not None:
        # set output shape to cube size
        out_shape = [scale_cube_dim, scale_cube_dim, scale_cube_dim]
    else:
        # else determine output shape
        out_x = in_shape[0] * np.abs(affine[0, 0]) + in_shape[1] * np.abs(affine[0, 1]) + in_shape[2] * np.abs(
            affine[0, 2])
        out_y = in_shape[0] * np.abs(affine[1, 0]) + in_shape[1] * np.abs(affine[1, 1]) + in_shape[2] * np.abs(
            affine[1, 2])
        out_z = in_shape[0] * np.abs(affine[2, 0]) + in_shape[1] * np.abs(affine[2, 1]) + in_shape[2] * np.abs(
            affine[2, 2])
        out_shape = [int(np.ceil(out_x)), int(np.ceil(out_y)), int(np.ceil(out_z))]

    # determine affine transform offset
    inv_af = affine.T
    c_in = np.array(in_shape) * 0.5 + np.array(roi_bbox[::2])
    c_out = np.array(out_shape) * 0.5
    offset = c_in - c_out.dot(inv_af)

    # Apply affine to roi
    roi = ndi.interpolation.affine_transform(roi, affine, offset=offset, order=0, output=np.float32,
                                             output_shape=out_shape)
    # Apply affine to labels if given
    if labels is not None:
        labels = ndi.interpolation.affine_transform(labels, affine, offset=offset, order=order, output=np.float32,
                                                    output_shape=out_shape)
    # Apply affine to image, accouting for possible 4d image
    if len(image.shape) > 3:
        # make 4d identity matrix and replace xyz component with 3d affine
        affine4d = np.eye(4)
        affine4d[:3, :3] = affine
        offset4d = np.append(offset, 0.)
        out_shape4d = np.append(out_shape, image.shape[-1])
        image = ndi.interpolation.affine_transform(image, affine4d, offset=offset4d, order=1, output=np.float32,
                                                   output_shape=out_shape4d)
    else:
        image = ndi.interpolation.affine_transform(image, affine, offset=offset, order=1, output=np.float32,
                                                   output_shape=out_shape)

    # if using scale_cube_dims return as is
    if scale_cube_dim is not None:
        if labels is not None:
            return image, roi, labels
        else:
            return image, roi

    # crop to rotated input ROI, adjusting for dilate --- what does this do?
    nzi = nonzero_slice_inds3d(roi)
    if dilate is not None:
        nzi = expand_region(roi.shape, nzi, dilate)
    roi = roi[nzi[0]:nzi[1], nzi[2]:nzi[3], nzi[4]:nzi[5]]
    image = image[nzi[0]:nzi[1], nzi[2]:nzi[3], nzi[4]:nzi[5]]
    if labels is not None:
        labels = labels[nzi[0]:nzi[1], nzi[2]:nzi[3], nzi[4]:nzi[5]]
        return image, roi, labels
    else:
        return image, roi


def normalize(input_img, mode='zero_mean'):
    """
    Performs image normalization to zero mean, unit variance or to interval [0, 1].
    :param input_img: (np.ndarray) The input numpy array.
    :param mode: (str) The normalization mode: 'unit' for scaling to [0,1] or 'zero_mean' for zero mean, unit variance.
    :return: The input array normalized to zero mean, unit variance or [0, 1].
    """

    # sanity checks
    if not isinstance(input_img, np.ndarray):
        raise TypeError("Input image should be np.ndarray but is: " + str(type(input_img)))

    # define epsilon for divide by zero errors
    epsilon = 1e-10

    # handle unit mode
    def unit(img):
        # perform normalization to [0, 1]
        img *= 1.0 / (np.max(img) + epsilon)
        return img

    # handle mean zscore
    def zscore(img):
        # perform z score normalization to 0 mean, unit std
        nonzero_bool = img != 0.
        mean = np.mean(img[nonzero_bool], axis=None)
        std = np.std(img[nonzero_bool], axis=None) + epsilon
        img = np.where(nonzero_bool, ((img - mean) / std), 0.)
        return img

    # handle mean stdev
    def mean_stdev(img):
        # constants
        new_mean = 1000.
        new_std = 200.
        # perform normalization to specified mean, stdev
        nonzero_bool = img != 0.
        mean = np.mean(img[nonzero_bool], axis=None)
        std = np.std(img[nonzero_bool], axis=None) + epsilon
        img = np.where(nonzero_bool, ((img - mean) / (std / new_std)) + new_mean, 0.)
        return img

    # handle median interquartile range
    def med_iqr(img, new_med=0., new_stdev=1.):
        # perform normalization to median, normalized interquartile range
        # uses factor of 0.7413 to normalize interquartile range to standard deviation
        nonzero_bool = img != 0.
        med = np.median(img[nonzero_bool], axis=None)
        niqr = stats.iqr(img[nonzero_bool], axis=None) * 0.7413 + epsilon
        img = np.where(nonzero_bool, ((img - med) / (niqr / new_stdev)) + new_med, 0.)
        return img

    # handle not implemented
    if mode in locals():
        input_img = locals()[mode](input_img)
    else:
        # get list of available normalization modes
        norm_modes = [k for k in locals().keys() if k not in ["input_img", "mode", "epsilon"]]
        raise NotImplementedError(
            "Specified normalization mode: '{}' is not one of the available modes: {}".format(mode, norm_modes))

    return input_img


def nonzero_slice_inds3d(input_numpy, cube=False):
    """
    Takes numpy array and returns slice indices of first and last nonzero pixels in 3d
    :param input_numpy: (np.ndarray) a numpy array containing image data.
    :param cube: (bool) if true, returns a cubic bounding box centered on roi with cube width equal to largest ROI
    dimension
    :return: inds - a list of 2 indices per dimension corresponding to the first and last nonzero slices in the array
    """

    # sanity checks
    if type(input_numpy) is not np.ndarray:
        raise ValueError("Input must be numpy array")

    # finds inds of first and last nonzero pixel in x
    vector = np.max(np.max(input_numpy, axis=2), axis=1)
    nz = np.nonzero(vector)[0]
    # if everything is zero, use whole input
    if nz.size == 0:
        nz = [0, input_numpy.shape[0]]
    xinds = [nz[0], nz[-1]]

    # finds inds of first and last nonzero pixel in y
    vector = np.max(np.max(input_numpy, axis=0), axis=1)
    nz = np.nonzero(vector)[0]
    # if everything is zero, use whole input
    if nz.size == 0:
        nz = [0, input_numpy.shape[0]]
    yinds = [nz[0], nz[-1]]

    # finds inds of first and last nonzero pixel in z
    vector = np.max(np.max(input_numpy, axis=0), axis=0)
    nz = np.nonzero(vector)[0]
    # if everything is zero, use whole input
    if nz.size == 0:
        nz = [0, input_numpy.shape[0]]
    zinds = [nz[0], nz[-1]]

    # handle cube option
    if cube:
        print("PRE CUBE DIMS ARE {}".format([xinds[1] - xinds[0], yinds[1] - yinds[0], zinds[1] - zinds[0]]))
        maxwidth = np.max([xinds[1] - xinds[0], yinds[1] - yinds[0], zinds[1] - zinds[0]])
        # adjust other widths to match largest
        if xinds[1] - xinds[0] < maxwidth:
            xinds[0] = xinds[0] - np.floor((xinds[1] - xinds[0]) - maxwidth)
            xinds[1] = xinds[1] + np.ceil((xinds[1] - xinds[0]) - maxwidth)
        if yinds[1] - yinds[0] < maxwidth:
            yinds[0] = yinds[0] - np.floor((yinds[1] - yinds[0]) - maxwidth)
            yinds[1] = yinds[1] + np.ceil((yinds[1] - yinds[0]) - maxwidth)
        if zinds[1] - zinds[0] < maxwidth:
            zinds[0] = zinds[0] - np.floor((zinds[1] - zinds[0]) - maxwidth)
            zinds[1] = zinds[1] + np.ceil((zinds[1] - zinds[0]) - maxwidth)
        print("POST CUBE DIMS ARE {}".format([xinds[1] - xinds[0], yinds[1] - yinds[0], zinds[1] - zinds[0]]))

    # perpare return
    inds = [xinds[0], xinds[1], yinds[0], yinds[1], zinds[0], zinds[1]]

    return inds


def zero_pad_image(input_data, out_dims, axes):
    """
    Zero pads an input image to the specified dimensions.
    :param input_data: (np.ndarray) the image data to be padded
    :param out_dims: (list(int)) the desired output dimensions for each axis.
    :param axes: (list(int)) the axes for padding. Must have same length as out_dims
    :return: (np.ndarray) the zero padded image
    """

    # sanity checks
    if type(input_data) is not np.ndarray:
        raise ValueError("Input must be a numpy array")
    if not all([np.issubdtype(val, np.integer) for val in out_dims]):
        raise ValueError("Output dims must be a list or tuple of ints")
    if not all([isinstance(axes, (tuple, list))] + [isinstance(val, int) for val in axes]):
        raise ValueError("Axes must be a list or tuple of ints")
    if not len(out_dims) == len(axes):
        raise ValueError("Output dimensions must have same length as axes")
    if len(axes) != len(set(axes)):
        raise ValueError("Axes cannot contain duplicate values")

    # determine pad widths
    pads = []
    for dim in range(len(input_data.shape)):
        pad = [0, 0]
        if dim in axes:
            total_pad = out_dims[axes.index(dim)] - input_data.shape[dim]
            pad = [int(math.ceil(total_pad / 2.)), int(math.floor(total_pad / 2.))]
        pads.append(pad)

    # pad array with zeros (default)
    input_data = np.pad(input_data, pads, 'constant')

    return input_data


##############################################
# PY FUNC 2D DATA FUNCTIONS
##############################################


def byte_convert(byte_data):
    if isinstance(byte_data, bytes):
        return byte_data.decode()
    if isinstance(byte_data, dict):
        return dict(map(byte_convert, byte_data.items()))
    if isinstance(byte_data, tuple):
        return map(byte_convert, byte_data)
    if isinstance(byte_data, (np.ndarray, list)):
        return list(map(byte_convert, byte_data))

    return byte_data


def load_multicon_preserve_size(study_dir, feature_prefx, data_fmt, plane, norm=True, norm_mode='zero_mean'):
    """
    Load multicontrast image data without cropping or otherwise adjusting size. For use with inference/prediction.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param norm: (bool) Whether or not to normalize the input data after loading.
    :param norm_mode: (str) The method for normalization, used by normalize function.
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # convert bytes to strings
    study_dir = byte_convert(study_dir)
    feature_prefx = byte_convert(feature_prefx)
    plane = byte_convert(plane)
    data_fmt = byte_convert(data_fmt)
    norm_mode = byte_convert(norm_mode)

    # sanity checks
    if not os.path.isdir(study_dir):
        raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]):
        raise ValueError("Data prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']:
        raise ValueError("data_format invalid")

    # load multi-contrast data and normalize, no slice trimming for infer data
    data = load_single_study(study_dir, feature_prefx, data_format=data_fmt, plane=plane,
                             norm=norm, norm_mode=norm_mode)

    # transpose slices to batch dimension format such that format is [z, x, y, c] or [z, c, x, y]
    axes = (3, 0, 1, 2) if data_fmt == 'channels_first' else (2, 0, 1, 3)
    data = np.transpose(data, axes=axes)

    return data


def load_roi_multicon_and_labels(study_dir, feature_prefx, label_prefx, mask_prefx, dilate=0, plane='ax',
                                 data_fmt='channels_last', aug=False, interp=1, norm=True, norm_lab=True,
                                 norm_mode='zero_mean'):
    """
    Patch loader generates 2D patch data for images and labels given a list of 3D input NiFTI images a mask.
    Performs optional data augmentation with affine rotation in 3D.
    Data is cropped to the nonzero bounding box for the mask file before patches are generated.
    :param study_dir: (str) The path to the study directory to get data from.
    :param feature_prefx: (iterable of str) The prefixes for the image files containing the data (features).
    :param label_prefx: (str) The prefixe for the image files containing the labels.
    :param mask_prefx: (str) The prefixe for the image files containing the data mask. None uses no masking.
    :param dilate: (int) The amount to dilate the region by in all dimensions
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param data_fmt (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param aug: (bool) Whether or not to perform data augmentation with random 3D affine rotation.
    :param interp: (int) The order of spline interpolation for label data. Must be 0-5
    :param norm: (bool) Whether or not to normalize the input data after loading.
    :param norm_lab: (bool) whether or not to normalize the label data after loading.
    :param norm_mode: (str) The method for normalization, used by normalize function.
    :return: (tf.tensor) The patch data for features and labels as a tensorflow variable.
    """

    # convert bytes to strings
    study_dir = byte_convert(study_dir)
    feature_prefx = byte_convert(feature_prefx)
    label_prefx = byte_convert(label_prefx)
    mask_prefx = byte_convert(mask_prefx)
    plane = byte_convert(plane)
    data_fmt = byte_convert(data_fmt)
    norm_mode = byte_convert(norm_mode)

    # sanity checks
    if plane not in ['ax', 'cor', 'sag']:
        raise ValueError("Did not understand specified plane: " + str(plane))
    if data_fmt not in ['channels_last', 'channels_first']:
        raise ValueError("Did not understand specified data_fmt: " + str(data_fmt))

    # define full paths
    data_files = [glob(study_dir + '/*' + contrast + '.nii.gz')[0] for contrast in feature_prefx]
    labels_file = glob(study_dir + '/*' + label_prefx[0] + '.nii.gz')[0]
    if mask_prefx:
        mask_file = glob(study_dir + '/*' + mask_prefx[0] + '.nii.gz')[0]
    else:
        mask_file = data_files[0]
    if not all([os.path.isfile(img) for img in data_files + [labels_file] + [mask_file]]):
        raise ValueError("One or more of the input data/labels/mask files does not exist")

    # load the mask and get the full data dims - handle None mask argument
    if mask_prefx:
        mask = (nib.load(mask_file).get_fdata() > 0.).astype(float)
    else:
        mask = np.ones_like(nib.load(mask_file).get_fdata(), dtype=float)
    data_dims = mask.shape

    # load data
    data = np.empty((data_dims[0], data_dims[1], data_dims[2], len(data_files)), dtype=np.float32)
    for i, im_file in enumerate(data_files):
        if norm:
            data[:, :, :, i] = normalize(nib.load(im_file).get_fdata(), mode=norm_mode)
        else:
            data[:, :, :, i] = nib.load(im_file).get_fdata()

    # load labels
    if norm_lab:
        labels = normalize(nib.load(labels_file).get_fdata(), mode=norm_mode)
    else:
        labels = nib.load(labels_file).get_fdata()

    # center the tumor in the image usine affine, with optional rotation for data augmentation
    if aug:  # if augmenting, select random rotation values for x, y, and z axes depending on specified plane
        posneg = 1 if np.random.random() < 0.5 else -1
        theta = np.random.random() * (np.pi / 6.) * posneg if plane == 'cor' else 0.  # rotation in yz plane
        phi = np.random.random() * (np.pi / 6.) * posneg if plane == 'sag' else 0.  # rotation in xz plane
        psi = np.random.random() * (np.pi / 6.) * posneg if plane == 'ax' else 0.  # rotation in xy plane
    else:  # if not augmenting, no rotation is applied, and affine is used only for offset to center the ROI
        theta = 0.
        phi = 0.
        psi = 0.

    # make affine, calculate offset using mask center of mass and affine
    affine = create_affine(theta=theta, phi=phi, psi=psi)
    com = ndi.measurements.center_of_mass(mask)
    cent = np.array(mask.shape) / 2.
    offset = com - np.dot(affine, cent)

    # apply affines to mask, data, labels
    mask = affine_transform(mask, affine=affine, offset=offset, order=0)  # nn interp for mask
    data = affine_transform(data, affine=affine, offset=offset, order=1)  # linear interp for data
    labels = affine_transform(labels, affine=affine, offset=offset, order=interp)  # user def interp for labels

    # get the tight bounding box of the mask after affine rotation
    msk_bbox = nonzero_slice_inds3d(mask)

    # dilate bbox if necessary - this also ensures that the bbox does not exceed original image dims
    msk_bbox = expand_region(data_dims, msk_bbox, dilate)

    # determine new dim sizes
    dim_sizes = [msk_bbox[1] - msk_bbox[0], msk_bbox[3] - msk_bbox[2], msk_bbox[5] - msk_bbox[4]]

    # extract the region from the data
    data_region = np.zeros(dim_sizes + [len(data_files)])
    for i in range(len(data_files)):
        data_region[:, :, :, i] = data[msk_bbox[0]:msk_bbox[1], msk_bbox[2]:msk_bbox[3], msk_bbox[4]:msk_bbox[5], i]
    data = data_region
    labels = labels[msk_bbox[0]:msk_bbox[1], msk_bbox[2]:msk_bbox[3], msk_bbox[4]:msk_bbox[5]]

    # permute to [batch, x, y, channels] for tensorflow patching
    if plane == 'ax':
        data = np.transpose(data, axes=(2, 0, 1, 3))
        labels = np.transpose(np.expand_dims(labels, axis=3), axes=(2, 0, 1, 3))
    elif plane == 'cor':
        data = np.transpose(data, axes=(1, 0, 2, 3))
        labels = np.transpose(np.expand_dims(labels, axis=3), axes=(1, 0, 2, 3))
    elif plane == 'sag':
        labels = np.expand_dims(labels, axis=3)
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_fmt == 'channels_first':
        data = np.transpose(data, axes=(0, 3, 1, 2))
        labels = np.transpose(labels, axes=(0, 3, 1, 2))

    return data.astype(np.float32), labels.astype(np.float32)


def load_roi_multicon_and_labels_3d(study_dir, feature_prefx, label_prefx, mask_prefx, dilate=0, plane='ax',
                                    data_fmt='channels_last', aug=False, interp=1, norm=True, norm_lab=True,
                                    norm_mode='zero_mean', return_mask=False, scale_cube_dim=None):
    """
    Patch loader generates 3D patch data for images and labels given a list of 3D input NiFTI images a mask.
    Performs optional data augmentation with affine rotation in 3D.
    Data is cropped to the nonzero bounding box for the mask file before patches are generated.
    :param study_dir: (str) The path to the study directory to get data from.
    :param feature_prefx: (iterable of str) The prefixes for the image files containing the data (features).
    :param label_prefx: (str) The prefix for the image files containing the labels.
    :param dilate: (int) The amount to dilate the region by in all dimensions
    :param mask_prefx: (str) The prefixe for the image files containing the data mask. None uses no masking.
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param data_fmt (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param aug: (bool) Whether or not to perform data augmentation with random 3D affine rotation.
    :param interp: (int) The order of spline interpolation for label data. Must be 0-5
    :param norm: (bool) Whether or not to normalize the input data after loading.
    :param norm_lab: (bool) whether or not to normalize the label data after loading.
    :param norm_mode: (str) The method for normalization, used by normalize function.
    :param return_mask: (bool or list) whether or not to return the mask as one of the outputs. If list, mask values are
    mapped to the values in the list. For example, passing [1, 2, 4] would map mask value 0->1, 1->2, 2->4.
    :param scale_cube_dim: (None or int) If None, has no effect. If an int, this will extract the ROI as a cube with
    width equal to largest roi dimension and will then scale result to a cube with width equal to scale_cube_dim
    :return: (tf.tensor) The patch data for features and labels as a tensorflow variable.
    """

    # convert bytes to strings
    study_dir = byte_convert(study_dir)
    feature_prefx = byte_convert(feature_prefx)
    label_prefx = byte_convert(label_prefx)
    mask_prefx = byte_convert(mask_prefx)
    plane = byte_convert(plane)
    data_fmt = byte_convert(data_fmt)
    norm_mode = byte_convert(norm_mode)

    # handle return mask argument, which should be bool or a list (np.ndarray) of ints to map weight values to
    if isinstance(return_mask, np.bool_) and not return_mask:
        # if return_mask is false set to None for ease of identifying below
        return_mask = None

    # sanity checks
    if plane not in ['ax', 'cor', 'sag']:
        raise ValueError("Did not understand specified plane: " + str(plane))
    if data_fmt not in ['channels_last', 'channels_first']:
        raise ValueError("Did not understand specified data_fmt: " + str(plane))

    # define full paths
    data_files = [glob(study_dir + '/*' + contrast + '.nii.gz')[0] for contrast in feature_prefx]
    labels_file = glob(study_dir + '/*' + label_prefx[0] + '.nii.gz')[0]
    if mask_prefx:
        mask_file = glob(study_dir + '/*' + mask_prefx[0] + '.nii.gz')[0]
    else:
        mask_file = data_files[0]
    if not all([os.path.isfile(img) for img in data_files + [labels_file] + [mask_file]]):
        raise ValueError("One or more of the input data/labels/mask files does not exist")

    # load the mask and get the full data dims - handle None mask argument
    if mask_prefx:
        mask = nib.load(mask_file).get_fdata()
    else:
        mask = np.ones_like(nib.load(mask_file).get_fdata(), dtype=np.float32)
    data_dims = mask.shape

    # load data and normalize
    data = np.empty((data_dims[0], data_dims[1], data_dims[2], len(data_files)), dtype=np.float32)
    for i, im_file in enumerate(data_files):
        if norm:
            data[:, :, :, i] = normalize(nib.load(im_file).get_fdata(), mode=norm_mode)
        else:
            data[:, :, :, i] = nib.load(im_file).get_fdata()

    # load labels with NORMALIZATION - add option for normalized vs non-normalized labels here
    if norm_lab:
        labels = normalize(nib.load(labels_file).get_fdata(), mode=norm_mode)
    else:
        labels = nib.load(labels_file).get_fdata()

    # center the ROI in the image usine affine, with optional rotation for data augmentation
    if aug:  # if augmenting, select random rotation values for x, y, and z axes depending on plane
        posneg = 1 if np.random.random() < 0.5 else -1
        theta = np.random.random() * (np.pi / 6.) * posneg if plane == 'cor' else 0.  # rotation in yz plane
        phi = np.random.random() * (np.pi / 6.) * posneg if plane == 'sag' else 0.  # rotation in xz plane
        psi = np.random.random() * (np.pi / 6.) * posneg if plane == 'ax' else 0.  # rotation in xy plane
    else:  # if not augmenting, no rotation is applied, and affine is used only for offset to center the mask ROI
        theta = 0.
        phi = 0.
        psi = 0.

    # make affine, calculate offset using mask center of mass of binirized mask, get nonzero bbox of mask
    affine = create_affine(theta=theta, phi=phi, psi=psi)

    # apply affines to mask, data, labels
    data, mask, labels = affine_transform_roi(data, mask, labels, affine, dilate, interp, scale_cube_dim)

    # add batch and channel dims as necessary to get to [batch, x, y, z, channel]
    data = np.expand_dims(data, axis=0)  # add a batch dimension of 1
    labels = np.expand_dims(labels, axis=(0, 4))  # add a batch and channel dimension of 1
    if return_mask is not None:
        mask = np.expand_dims(mask, axis=(0, 4))  # add a batch and channel dimension of 1

    # handle different planes
    if plane == 'ax':
        pass
    elif plane == 'cor':
        data = np.transpose(data, axes=[0, 1, 3, 2, 4])
        labels = np.transpose(labels, axes=[0, 1, 3, 2, 4])
        if return_mask is not None:
            mask = np.transpose(mask, axes=[0, 1, 3, 2, 4])
    elif plane == 'sag':
        data = np.transpose(data, axes=[0, 2, 3, 1, 4])
        labels = np.transpose(labels, axes=[0, 2, 3, 1, 4])
        if return_mask is not None:
            mask = np.transpose(mask, axes=[0, 1, 3, 2, 4])
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_fmt == 'channels_first':
        data = np.transpose(data, axes=[0, 4, 1, 2, 3])
        labels = np.transpose(labels, axes=[0, 4, 1, 2, 3])
        if return_mask is not None:
            mask = np.transpose(mask, axes=[0, 4, 1, 2, 3])

    # handle return_mask argument as list of value mappings
    if isinstance(return_mask, np.ndarray):
        # map each value of mask to the corresponding loss factor
        mask = return_mask[mask.astype(np.int32)]

    # handle return_mask flag
    if return_mask is not None:
        # concatenate weights to last (channels) dimension of labels to sneak it into model.fit for custom weighted loss
        return data.astype(np.float32), np.concatenate((labels.astype(np.float32), mask.astype(np.float32)), axis=-1)
    else:
        return data.astype(np.float32), labels.astype(np.float32)


def load_csv_and_roi_multicon_3d(study_dir, feature_prefx, label_csv, mask_prefx, label_col=1, dilate=0, plane='ax',
                                 data_fmt='channels_last', aug=False, interp=1, norm=True, norm_mode='zero_mean',
                                 scale_cube_dim=None):
    """
    Patch loader generates 3D patch data for images and labels given a list of 3D input NiFTI images a mask.
    Performs optional data augmentation with affine rotation in 3D.
    Data is cropped to the nonzero bounding box for the mask file before patches are generated.
    :param study_dir: (str) The path to the study directory to get data from.
    :param feature_prefx: (iterable of str) The prefixes for the image files containing the data (features).
    :param label_csv: (str) The full path to the labels CSV. First colum must correspond to data directory names.
    :param mask_prefx: (str) The prefix of the image file containing the ROI to extraxt image data with.
    :param label_col: (int) The index (from 0) of the csv column to use as labels
    :param dilate: (int) The amount to dilate the region by in all dimensions
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param data_fmt (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param aug: (bool) Whether or not to perform data augmentation with random 3D affine rotation.
    :param interp: (int) The order of spline interpolation for label data. Must be 0-5
    :param norm: (bool) Whether or not to normalize the input data after loading.
    :param norm_mode: (str) The method for normalization, used by normalize function.
    mapped to the values in the list. For example, passing [1, 2, 4] would map mask value 0->1, 1->2, 2->4.
    :param scale_cube_dim: (None or int) If None, has no effect. If an int, this will extract the ROI as a cube with
    width equal to largest roi dimension and will then scale result to a cube with width equal to scale_cube_dim
    :return: (tf.tensor) The image data and labels as a tensorflow tensor.
    """

    # convert bytes to strings
    study_dir = byte_convert(study_dir)
    feature_prefx = byte_convert(feature_prefx)
    label_csv = byte_convert(label_csv)
    mask_prefx = byte_convert(mask_prefx)
    plane = byte_convert(plane)
    data_fmt = byte_convert(data_fmt)
    norm_mode = byte_convert(norm_mode)

    # sanity checks
    if plane not in ['ax', 'cor', 'sag']:
        raise ValueError("Did not understand specified plane: " + str(plane))
    if data_fmt not in ['channels_last', 'channels_first']:
        raise ValueError("Did not understand specified data_fmt: " + str(plane))

    # define full paths and check that image files exist
    data_files = [glob(study_dir + '/*' + contrast + '.nii.gz')[0] for contrast in feature_prefx]
    if mask_prefx:
        mask_file = glob(study_dir + '/*' + mask_prefx[0] + '.nii.gz')[0]
    else:
        mask_file = data_files[0]
    if not all([os.path.isfile(img) for img in data_files + [mask_file]]):
        raise ValueError("One or more of the input data/labels/mask files does not exist")

    # check that label csv exists
    if not os.path.isfile(label_csv[0]):
        raise FileNotFoundError("Specified label CSV file does not exist: {}".format(label_csv))

    # load labels and get value for this specific image file
    with open(label_csv[0], 'r') as f:
        csv_data = np.array(list(csv.reader(f)))
    direc_id = os.path.basename(study_dir.rstrip('/'))
    # handle numbered study direc name with leading zeroes
    if direc_id.isdigit():
        direc_id = str(int(direc_id))
    # extract relevant datum from csv using direc ID - cast data to float32 and squeeze out extra dims
    label_row = np.nonzero(csv_data[:,0] == direc_id)
    label = np.squeeze(np.float32(csv_data[label_row, label_col]), axis=0)

    # get extra data from csv to include as inputs
    csv_features = np.squeeze(csv_data[label_row, label_col + 1:]).astype(np.float32)

    # load the mask and get the full data dims - handle None mask argument where whole image is used
    if mask_prefx:
        mask = nib.load(mask_file).get_fdata()
    else:
        mask = np.ones_like(nib.load(mask_file).get_fdata(), dtype=np.float32)
    data_dims = mask.shape

    # load data and normalize
    data = np.empty((data_dims[0], data_dims[1], data_dims[2], len(data_files)), dtype=np.float32)
    for i, im_file in enumerate(data_files):
        if norm:
            data[:, :, :, i] = normalize(nib.load(im_file).get_fdata(), mode=norm_mode)
        else:
            data[:, :, :, i] = nib.load(im_file).get_fdata()

    # center the ROI in the image usine affine, with optional rotation for data augmentation
    if aug:  # if augmenting, select random rotation values for x, y, and z axes depending on plane
        posneg = 1 if np.random.random() < 0.5 else -1
        theta = np.random.random() * (np.pi / 6.) * posneg if plane == 'cor' else 0.  # rotation in yz plane
        phi = np.random.random() * (np.pi / 6.) * posneg if plane == 'sag' else 0.  # rotation in xz plane
        psi = np.random.random() * (np.pi / 6.) * posneg if plane == 'ax' else 0.  # rotation in xy plane
    else:  # if not augmenting, no rotation is applied, and affine is used only for offset to center the mask ROI
        theta = 0.
        phi = 0.
        psi = 0.

    # make affine, calculate offset using mask center of mass of binirized mask, get nonzero bbox of mask
    affine = create_affine(theta=theta, phi=phi, psi=psi)

    # apply affines to mask, data, labels = None since labels are not image data
    data, mask = affine_transform_roi(data, mask, None, affine, dilate, interp, scale_cube_dim)

    # handle different planes
    if plane == 'ax':
        pass
    elif plane == 'cor':
        data = np.transpose(data, axes=[0, 1, 3, 2, 4])
    elif plane == 'sag':
        data = np.transpose(data, axes=[0, 2, 3, 1, 4])
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_fmt == 'channels_first':
        data = np.transpose(data, axes=[0, 4, 1, 2, 3])

    return data.astype(np.float32), csv_features.astype(np.float32), label.astype(np.float32)


def load_multicon_preserve_size_3d(study_dir, feat_prefx, data_fmt, plane='ax', norm=True, norm_mode='zero_mean'):
    """
    Load multicontrast image data without cropping or otherwise adjusting size. For use with inference/prediction.
    :param study_dir: (str) A directory containing the desired image data.
    :param feat_prefx: (list) a list of filenames - the data files to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param norm: (bool) Whether or not to normalize the input data after loading
    :param norm_mode: (str) The method for normalization, used by normalize function.
    :return: np ndarray containing the image data and regression target in the specified tf data format
    """

    # convert bytes to strings
    study_dir = byte_convert(study_dir)
    feat_prefx = byte_convert(feat_prefx)
    plane = byte_convert(plane)
    data_fmt = byte_convert(data_fmt)
    norm_mode = byte_convert(norm_mode)

    # sanity checks
    if not os.path.isdir(study_dir):
        raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feat_prefx]):
        raise ValueError("Data prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']:
        raise ValueError("data_format invalid")

    # load multi-contrast data and normalize, no slice trimming for infer data
    data = load_single_study(study_dir, feat_prefx, data_format=data_fmt, plane=plane, norm=norm, norm_mode=norm_mode)

    # generate batch size==1 format such that format is [1, x, y, z, c] or [1, c, x, y, z]
    data = np.expand_dims(data, axis=0)

    return data


def reconstruct_infer_patches(predictions, infer_dir, params):
    """
    Function for reconstructing the input 2D patches from the output predictions after 2D image patch prediction
    :param predictions: (tf.tensor) - the output of 3D patch prediction
    :param infer_dir: (str) - the directory containing the inferance dataset
    :param params: (obj) - the parameter object derived from the param file
    :return: (np.ndarray) - returns the reconstructed image generated by reversing ExtractVolumePatches
    """

    # define params - converting all to python native variables as they may be imported as numpy
    patch_size = params.infer_dims
    overlap = params.infer_patch_overlap
    data_prefix = [str(item) for item in params.data_prefix]
    data_format = params.data_format
    data_plane = params.data_plane

    # for sliding window 2d patches - must be same as in _tf_patches_infer above
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap[0], patch_size[1] / overlap[1], 1]
    rates = [1, 1, 1, 1]

    # define necessary functions
    def extract_patches(x):
        return tf.image.extract_patches(x, sizes=ksizes, strides=strides, rates=rates, padding='SAME')

    def extract_patches_inverse(x, y):
        with tf.GradientTape(persistent=True) as tape:
            _x = tf.zeros_like(x)
            tape.watch(_x)
            _y = extract_patches(_x)
            grad = tape.gradient(_y, _x)
            # Divide by grad, to "average" together the overlapping patches
            # otherwise they would simply sum up
            return tape.gradient(_y, _x, output_gradients=y) / grad

    # load original data as a dummy and convert channel dim size to match output [batch, x, y, z, channel]
    data = load_multicon_preserve_size(infer_dir, data_prefix, data_format, data_plane)
    data = np.zeros((data.shape[0:4] + (params.output_filters,)), dtype=np.float32)
    # data = data[:, :, :, [0]] if params.data_format == 'channels_last' else data[:, [0], :, :]

    # get shape of patches as they would have been generated during inference
    dummy_shape = tf.shape(input=extract_patches(data))

    # convert channels dimension to actual output_filters
    if params.data_format == 'channels_last':
        dummy_shape = dummy_shape[:-1] + [params.output_filters]
    else:
        dummy_shape = [dummy_shape[0], params.output_filters] + dummy_shape[2:]

    # reshape predictions to original patch shape
    predictions = tf.reshape(predictions, dummy_shape)

    # reconstruct
    reconstructed = extract_patches_inverse(data, predictions)
    output = np.squeeze(reconstructed.numpy())

    return output


def reconstruct_infer_patches_3d(predictions, infer_dir, params):
    """
    Function for reconstructing the input 3D volume from the output predictions after 3D image patch prediction
    :param predictions: (tf.tensor) - the output of 3D patch prediction
    :param infer_dir: (str) - the directory containing the inferance dataset
    :param params: (obj) - the parameter object derived from the param file
    :return: (np.ndarray) - returns the reconstructed image generated by reversing ExtractVolumePatches
    """

    # define params - converting all to python native variables as they may be imported as numpy
    patch_size = params.infer_dims
    overlap = params.infer_patch_overlap
    data_prefix = [str(item) for item in params.data_prefix]
    data_format = params.data_format
    data_plane = params.data_plane
    norm = params.norm_data
    norm_mode = params.norm_mode

    # for sliding window 3d slabs - must be same as in _tf_patches_3d_infer above
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap[0], patch_size[1] / overlap[1], patch_size[2] / overlap[2], 1]

    # define necessary functions
    def extract_patches(x):
        return tf.extract_volume_patches(x, ksizes=ksizes, strides=strides, padding='SAME')

    def extract_patches_inverse(x, y):
        with tf.GradientTape(persistent=True) as tape:
            _x = tf.zeros_like(x)
            tape.watch(_x)
            _y = extract_patches(_x)
            grad = tape.gradient(_y, _x)
            # Divide by grad, to "average" together the overlapping patches
            # otherwise they would simply sum up
            return tape.gradient(_y, _x, output_gradients=y) / grad

    # load original data as a dummy and convert channel dim size to match output [batch, x, y, z, channel]
    data = load_multicon_preserve_size_3d(infer_dir, data_prefix, data_format, data_plane, norm, norm_mode)
    data = np.zeros((data.shape[0:4] + (params.output_filters,)), dtype=np.float32)
    # data = data[:, :, :, :, [0]] if params.data_format == 'channels_last' else data[:, [0], :, :, :]

    # get shape of patches as they would have been generated during inference
    dummy_patches = extract_patches(data)

    # reshape predictions to original patch shape
    predictions = tf.reshape(predictions, tf.shape(input=dummy_patches))

    # reconstruct
    reconstructed = extract_patches_inverse(data, predictions)
    output = np.squeeze(reconstructed.numpy())

    return output


# utility function to get all study subdirectories in a given parent data directory
# returns shuffled directory list using user defined randomization seed
# saves a copy of output to study_dirs_list.json in study directory
def get_study_dirs(params, change_basedir=None, mode="unknown"):
    # report
    logging.info("Getting study directories for mode: {}".format(mode))

    # Study dirs json filename setup
    study_dirs_filepath = os.path.join(params.model_dir, 'study_dirs_list.json')

    # load study dirs file if it already exists for consistent training
    if os.path.isfile(study_dirs_filepath):
        logging.info("Loading existing study directories file: {}".format(study_dirs_filepath))
        with open(study_dirs_filepath) as f:
            study_dirs = json.load(f)

        # handle change_basedir argument
        if change_basedir:
            # get rename list of directories
            study_dirs = [os.path.join(change_basedir, os.path.basename(os.path.dirname(item))) for item in study_dirs]
            if not all([os.path.isdir(d) for d in study_dirs]):
                logging.error("Using change basedir argument in get_study_dirs but not all study directories exist")
                # get list of missing files
                missing = []
                for item in study_dirs:
                    if not os.path.isdir(item):
                        missing.append(item)
                raise FileNotFoundError("Missing the following data directories: {}".format('\n'.join(missing)))

        # make sure that study directories loaded from file actually exist and warn/error if some/all do not
        valid_study_dirs = []
        # check if label_prefix is a csv file
        if os.path.isfile(params.label_prefix[0]) and params.label_prefix[0].endswith('.csv'):
            logging.info("Label prefix is a csv file - assuming all directories are present in CSV file")
            search_prefix = params.data_prefix
        else:
            search_prefix = params.data_prefix + params.label_prefix
        for study in study_dirs:
            # get list of all expected files via glob
            files = [glob("{}/*{}.nii.gz".format(study, item)) for item in search_prefix]
            # check that a file was found and that file exists in each case
            if all(files) and all([os.path.isfile(f[0]) for f in files]):
                valid_study_dirs.append(study)
        # case, no valid study dirs
        if not valid_study_dirs:
            logging.error("study_dirs_list.json exists in the model directory but does not contain valid directories")
            raise ValueError("No valid study directories in study_dirs_list.json")
        # case, less valid study dirs than found in study dirs file
        elif len(valid_study_dirs) < len(study_dirs):
            logging.warning("Some study directories listed in study_dirs_list.json are missing or incomplete")
        # case, all study dirs in study dirs file are valid
        else:
            logging.info("Complete!")
        study_dirs = valid_study_dirs

    # if study dirs file does not exist, then determine study directories and create study_dirs_list.json
    else:
        logging.info("Determining train/test split based on params and available study directories in data directory")
        # get all valid subdirectories in data_dir
        study_dirs = [item for item in glob(params.data_dir + '/*/') if os.path.isdir(item)]
        # make sure all necessary files are present in each folder
        study_dirs = [study for study in study_dirs if all(
            [glob('{}/*{}.nii.gz'.format(study, item)) and os.path.isfile(glob('{}/*{}.nii.gz'.format(study, item))[0])
             for item in params.data_prefix + params.label_prefix])]

        # study dirs sorted in alphabetical order for reproducible results
        study_dirs.sort()

        # randomly shuffle input directories for training using a user defined randomization seed
        random.Random(params.random_state).shuffle(study_dirs)

        # save directory list to json file so it can be loaded in future
        with open(study_dirs_filepath, 'w+', encoding='utf-8') as f:
            json.dump(study_dirs, f, ensure_ascii=False, indent=4)  # save study dir list for consistency

    return study_dirs


# split list of all valid study directories into a train and test batch based on train fraction
def train_test_split(study_dirs, params):
    # first train fraction is train dirs, last 1-train fract is test dirs
    # assumes study dirs is already shuffled and/or stratified as wanted
    train_dirs = study_dirs[0:int(round(params.train_fract * len(study_dirs)))]
    eval_dirs = study_dirs[int(round(params.train_fract * len(study_dirs))):]

    return train_dirs, eval_dirs
