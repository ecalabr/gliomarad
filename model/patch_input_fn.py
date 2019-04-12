import os
import math
from glob import glob
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import tensorflow as tf
from random import shuffle


##############################################
# DATA UTILITIES
##############################################


def _load_single_study(study_dir, file_prefixes, data_format, slice_trim=None, norm=False, plane=None):
    """
    Image data I/O function for use in tensorflow Dataset map function. Takes a study directory and file prefixes and
    returns a 4D numpy array containing the image data. Performs optional slice trimming in z and normalization.
    :param study_dir: (str) the full path to the study directory
    :param file_prefixes: (str, list(str)) the file prefixes for the images to be loaded
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param slice_trim: (list, tuple) contains 2 ints, the first and last slice to use for trimming. None = auto trim.
                        [0, -1] does no trimming
    :param norm: (bool) whether or not to perform per dataset normalization
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :return: output - a 4D numpy array containing the image data
    """

    # sanity checks
    if not os.path.isdir(study_dir): raise ValueError("Specified study_dir does not exist")
    if data_format not in ['channels_last', 'channels_first']: raise ValueError("data_format invalid")
    if slice_trim is not None and not isinstance(slice_trim, (list, tuple)): raise ValueError(
        "slice_trim must be list/tuple")
    images = [glob(study_dir + '/*' + contrast + '*.nii.gz')[0] for contrast in file_prefixes]
    if not images: raise ValueError("No matching image files found for file prefixes: " + str(images))

    # load images and concatenate into a 4d numpy array
    output = []
    nz_inds = [0, None]
    for ind, image in enumerate(images):
        if ind == 0:  # find dimensions after trimming zero slices and preallocate 4d array
            first_image = nib.load(images[0]).get_fdata()
            if slice_trim:
                nz_inds = slice_trim
            else:
                nz_inds = _nonzero_slice_inds3d(first_image)  # get z nonzero inds only using 3d function
                nz_inds = nz_inds[4:]
            first_image = first_image[:, :, nz_inds[0]:nz_inds[1]]
            # do normalization
            if norm:
                first_image = _normalize(first_image)
            output_shape = list(first_image.shape)[0:3] + [len(images)]
            output = np.zeros(output_shape, np.float32)
            output[:, :, :, 0] = first_image
        else:
            img = nib.load(images[ind]).get_fdata()[:, :, nz_inds[0]:nz_inds[1]]
            # do normalization
            if norm:
                img = _normalize(img)
            output[:, :, :, ind] = img

    # permute to desired plane in format [x, y, z, channels] for tensorflow
    if plane == 'ax':
        pass
    elif plane == 'cor':
        output = np.transpose(output, axes=(0, 2, 1, 3))
    elif plane == 'sag':
        output = np.transpose(output, axes=(1, 2, 0, 3))
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_format == 'channels_first':
        output = np.transpose(output, axes=(3, 0, 1, 2))

    return output, nz_inds


def _expand_region(input_dims, region_bbox, delta):
    """
    Symmetrically expands a given 3D region bounding box by delta in each dimension without exceeding original image dims
    :param input_dims: (list or tuple of ints) the original image dimensions
    :param region_bbox: (list or tuple of ints) the region bounding box to expand
    :param delta: (int) the amount to expand each dimension by.
    :return: (list or tuple of ints) the expanded bounding box
    """

    # determine how much to add on each side of the bounding box
    deltas = np.array([-int(np.floor(delta/2.)), int(np.ceil(delta/2.))] * 3)

    # use deltas to get a new bounding box
    tmp_bbox = np.array(region_bbox) - deltas

    # make sure there are not values outside of the original image
    new_bbox = []
    for i, item in enumerate(tmp_bbox):
        if i % 2 == 0: # for even indices, make sure there are no negatives
            if item < 0:
                item = 0
        else: # for odd indices, make sure they do not exceed original dims
            if item > input_dims[(i-1)/2]:
                item = input_dims[(i-1)/2]
        new_bbox.append(item)

    return new_bbox


def _create_affine(theta=None, phi=None, psi=None):
    """
    Creates a 3D rotation affine matrix given three rotation angles.
    :param theta: (float) The theta angle in radians. If None, a random angle is chosen.
    :param phi: (float) The phi angle in radians. If None, a random angle is chosen.
    :param psi: (float) The psi angle in radians. If None, a random angle is chosen.
    :return: (np.ndarray) a 3x3 affine rotation matrix.
    """

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


def _affine_transform(input_img, affine, offset=None, order=1):
    """
    Apply a 3D rotation affine transform to input_img with specified offset and spline interpolation order.
    :param input_img: (np.ndarray) The input image.
    :param affine: (np.ndarray) The 3D affine array of shape [3, 3]
    :param offset: (np.ndarray) The offset to apply to the image after rotation, should be shape [3,]
    :param order: (int) The spline interpolation order. Must be 0-5
    :return: The input image after applying the affine rotation and offset.
    """

    # sanity checks
    if not isinstance(input_img, np.ndarray):
        raise TypeError("Input image should be np.ndarray but is: " + str(type(input_img)))
    # define affine if it doesn't exist
    if affine is None:
        affine = _create_affine()
    if not isinstance(affine, np.ndarray):
        raise TypeError("Affine should be np.ndarray but is: " + str(type(affine)))
    if not affine.shape == (3, 3):
        raise ValueError("Affine should have shape (3, 3)")
    # define offset if it doesn't exist
    if offset is None:
        center = np.array(input_img.shape)
        offset = center - np.dot(affine, center)
    if not isinstance(offset, np.ndarray):
        raise TypeError("Offset should be np.ndarray but is: " + str(type(offset)))
    if not offset.shape == (3,):
        raise ValueError("Offset should have shape (3,)")

    # Apply affine
    # handle 4d
    if len(input_img.shape) > 3:
        output_img = np.zeros(input_img.shape)
        for i in range(input_img.shape[-1]):
            output_img[:, :, :, i] = ndi.interpolation.affine_transform(input_img[:, :, :, i], affine,
                                                                        offset=offset, order=order)
    else:
        output_img = ndi.interpolation.affine_transform(input_img, affine, offset=offset, order=order)

    return output_img


def _normalize(input_img, mode='zero_mean'):
    """
    Performs image normalization to zero mean, unit variance or to interval [0, 1].
    :param input_img: (np.ndarray) The input numpy array.
    :param mode: (str) The normalization mode: 'unit' for scaling to [0,1] or 'zero_mean' for zero mean, unit variance.
    :return: The input array normalized to zero mean, unit variance or [0, 1].
    """

    # sanity checks
    if not isinstance(input_img, np.ndarray):
        raise TypeError("Input image should be np.ndarray but is: " + str(type(input_img)))
    if mode not in ['unit', 'zero_mean']: raise ValueError("Mode must be 'unit' or 'zero_mean' but is: " + str(mode))

    # handle zero mean mode
    if mode == 'zero_mean':
        # perform normalization to zero mean unit variance
        nzi = np.nonzero(input_img)
        mean = np.mean(input_img[nzi], None)
        std = np.std(input_img[nzi], None)
        input_img = np.where(input_img != 0., (input_img - mean) / std, 0.)

    # handle unit mode
    if mode == 'unit':
        # perform normalization to [0, 1]
        input_img *= 1.0 / input_img.max()

    return input_img


def _nonzero_slice_inds3d(input_numpy):
    """
    Takes numpy array and returns slice indices of first and last nonzero pixels in 3d
    :param input_numpy: (np.ndarray) a numpy array containing image data.
    :return: inds - a list of 2 indices per dimension corresponding to the first and last nonzero slices in the array
    """

    # sanity checks
    if type(input_numpy) is not np.ndarray: raise ValueError("Input must be numpy array")

    # finds inds of first and last nonzero pixel in x
    vector = np.max(np.max(input_numpy, axis=2), axis=1)
    nz = np.nonzero(vector)[0]
    xinds = [nz[0], nz[-1]]

    # finds inds of first and last nonzero pixel in y
    vector = np.max(np.max(input_numpy, axis=0), axis=1)
    nz = np.nonzero(vector)[0]
    yinds = [nz[0], nz[-1]]

    # finds inds of first and last nonzero pixel in z
    vector = np.max(np.max(input_numpy, axis=0), axis=0)
    nz = np.nonzero(vector)[0]
    zinds = [nz[0], nz[-1]]

    # perpare return
    inds = [xinds[0], xinds[1], yinds[0], yinds[1], zinds[0], zinds[1]]

    return inds


def _zero_pad_image(input_data, out_dims, axes):
    """
    Zero pads an input image to the specified dimensions.
    :param input_data: (np.ndarray) the image data to be padded
    :param out_dims: (list(int)) the desired output dimensions for each axis.
    :param axes: (list(int)) the axes for padding. Must have same length as out_dims
    :return: (np.ndarray) the zero padded image
    """

    # sanity checks
    if type(input_data) is not np.ndarray: raise ValueError("Input must be a numpy array")
    if not all([np.issubdtype(val, np.integer) for val in out_dims]): raise ValueError(
        "Output dims must be a list or tuple of ints")
    if not all([isinstance(axes, (tuple, list))] + [isinstance(val, int) for val in axes]): raise ValueError(
        "Axes must be a list or tuple of ints")
    if not len(out_dims) == len(axes): raise ValueError("Output dimensions must have same length as axes")
    if len(axes) != len(set(axes)): raise ValueError("Axes cannot contain duplicate values")

    # determine pad widths
    pads = []
    for dim in range(len(input_data.shape)):
        pad = [0, 0]
        if dim in axes:
            total_pad = out_dims[axes.index(dim)] - input_data.shape[dim]
            pad = [int(math.ceil(total_pad/2.)), int(math.floor(total_pad/2.))]
        pads.append(pad)

    # pad array with zeros (default)
    input_data = np.pad(input_data, pads, 'constant')

    return input_data


##############################################
# PY FUNC DATA FUNCTIONS
##############################################


def _load_multicon_and_labels(study_dir, feature_prefx, label_prefx, data_fmt, out_dims, plane='ax'):
    """
    Load multicontrast image data and target data/labels.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param label_prefx: (list) a list containing one string, the labels to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param out_dims: (list(int)) the desired output data dimensions, data will be zero padded to dims
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_dir): raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]): raise ValueError("Data prefixes must be strings")
    if not all([isinstance(a, str) for a in label_prefx]): raise ValueError("Labels prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']: raise ValueError("data_format invalid")
    if not all([np.issubdtype(a, np.integer) for a in out_dims]): raise ValueError(
        "data_dims must be a list/tuple of ints")

    # load multi-contrast data
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, norm=True, plane=plane)  # normalize input imgs

    # load labels data
    labels, nzi = _load_single_study(study_dir, label_prefx, data_format=data_fmt, slice_trim=nzi, plane=plane)

    # do data padding to desired dims
    axes = [1, 2] if data_fmt == 'channels_last' else [2, 3]
    data = _zero_pad_image(data, out_dims, axes)
    labels = _zero_pad_image(labels, out_dims, axes)

    return data, labels


def _load_multicon_preserve_size(study_dir, feature_prefx, data_fmt, out_dims, plane):
    """
    Load multicontrast image data without cropping or otherwise adjusting size. For use with inference/prediction.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param out_dims: (list(int)) the desired output data dimensions, data will be zero padded to dims
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_dir): raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]): raise ValueError("Data prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']: raise ValueError("data_format invalid")
    if not all([np.issubdtype(a, np.integer) for a in out_dims]): raise ValueError(
        "data_dims must be a list/tuple of ints")

    # load multi-contrast data and normalize, no slice trimming for infer data
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, slice_trim=[0, None],
                                   norm=True, plane=plane)

    return data


def _load_roi_multicon_and_labels(study_dir, feature_prefx, label_prefx, mask_prefx, dilate=0, plane='ax',
                                  data_fmt='channels_last', aug='no', interp=1):
    """
    Patch loader generates 2D patch data for images and labels given a list of 3D input NiFTI images a mask.
    Performs optional data augmentation with affine rotation in 3D.
    Data is cropped to the nonzero bounding box for the mask file before patches are generated.
    :param study_dir: (str) The path to the study directory to get data from.
    :param feature_prefx: (iterable of str) The prefixes for the image files containing the data (features).
    :param label_prefx: (str) The prefixe for the image files containing the labels.
    :param mask_prefx: (str) The prefixe for the image files containing the data mask.
    :param dilate: (int) The amount to dilate the region by in all dimensions
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param data_fmt (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param aug: (str) 'yes' or 'no' - Whether or not to perform data augmentation with random 3D affine rotation.
    :param interp: (int) The order of spline interpolation for label data. Must be 0-5
    :return: (tf.tensor) The patch data for features and labels as a tensorflow variable.
    """

    # sanity checks
    if not plane in ['ax', 'cor', 'sag']:
        raise ValueError("Did not understand specified plane: " + str(plane))
    if not data_fmt in ['channels_last', 'channels_first']:
        raise ValueError("Did not understand specified data_fmt: " + str(plane))

    # define full paths
    data_files = [glob(study_dir + '/*' + contrast + '*.nii.gz')[0] for contrast in feature_prefx]
    labels_file = glob(study_dir + '/*' + label_prefx[0] + '*.nii.gz')[0]
    mask_file = glob(study_dir + '/*' + mask_prefx[0] + '*.nii.gz')[0]
    if not all([os.path.isfile(img) for img in data_files + [labels_file] + [mask_file]]):
        raise ValueError("One or more of the input data/labels/mask files does not exist")

    # load the mask and get the full data dims
    mask = nib.load(mask_file).get_fdata()
    data_dims = mask.shape

    # load data
    data = np.zeros((data_dims[0], data_dims[1], data_dims[2], len(data_files)))
    for i, im_file in enumerate(data_files):
        data[:, :, :, i] = _normalize(nib.load(im_file).get_fdata())

    # load labels
    labels = nib.load(labels_file).get_fdata()

    # center the tumor in the image usine affine, with optional rotation for data augmentation
    if aug == 'yes':  # if augmenting, select random rotation values for x, y, and z axes
        theta = 0.  # np.random.random() * (np.pi / 2.)  # rotation in yz plane
        phi = 0.  # np.random.random() * (np.pi / 2.)  # rotation in xz plane
        psi = np.random.random() * (np.pi / 2.)  # rotation in xy plane
    elif aug == 'no':  # if not augmenting, no rotation is applied, and affine is used only for offset to center the ROI
        theta = 0.
        phi = 0.
        psi = 0.
    else:
        raise ValueError("Augment data param argument must be yes or no but is: " + str(aug))

    # make affine, calculate offset using mask center of mass and affine
    affine = _create_affine(theta=theta, phi=phi, psi=psi)
    com = ndi.measurements.center_of_mass(mask)
    cent = np.array(mask.shape) / 2.
    offset = com - np.dot(affine, cent)

    # apply affines to mask, data, labels
    mask = _affine_transform(mask, affine=affine, offset=offset, order=0)  # nn interp for mask
    data = _affine_transform(data, affine=affine, offset=offset, order=1)  # linear interp for data
    labels = _affine_transform(labels, affine=affine, offset=offset, order=interp)  # user def interp for labels

    # get the tight bounding box of the mask after affine rotation
    msk_bbox = _nonzero_slice_inds3d(mask)

    # dilate bbox if necessary
    if dilate:
        msk_bbox = _expand_region(data_dims, msk_bbox, dilate)

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
        labels = np.expand_dims(labels, axis=3)
        labels = np.transpose(labels, axes=(2, 0, 1, 3))
    elif plane == 'cor':
        data = np.transpose(data, axes=(1, 0, 2, 3))
        labels = np.expand_dims(labels, axis=3)
        labels = np.transpose(labels, axes=(1, 0, 2, 3))
    elif plane == 'sag':
        labels = np.expand_dims(labels, axis=3)
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_fmt == 'channels_first':
        data = np.transpose(data, axes=(0, 3, 1, 2))
        labels = np.transpose(labels, axes=(0, 3, 1, 2))

    return data.astype(np.float32), labels.astype(np.float32)


def _load_roi_multicon_and_labels_3d(study_dir, feature_prefx, label_prefx, mask_prefx, dilate=0, plane='ax',
                                  data_fmt='channels_last', aug='no', interp=1):
    """
    Patch loader generates 3D patch data for images and labels given a list of 3D input NiFTI images a mask.
    Performs optional data augmentation with affine rotation in 3D.
    Data is cropped to the nonzero bounding box for the mask file before patches are generated.
    :param study_dir: (str) The path to the study directory to get data from.
    :param feature_prefx: (iterable of str) The prefixes for the image files containing the data (features).
    :param label_prefx: (str) The prefix for the image files containing the labels.
    :param dilate: (int) The amount to dilate the region by in all dimensions
    :param mask_prefx: (str) The prefixe for the image files containing the data mask.
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param data_fmt (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param aug: (str) Either yes or no - Whether or not to perform data augmentation with random 3D affine rotation.
    :param interp: (int) The order of spline interpolation for label data. Must be 0-5
    :return: (tf.tensor) The patch data for features and labels as a tensorflow variable.
    """

    # sanity checks
    if not plane in ['ax', 'cor', 'sag']:
        raise ValueError("Did not understand specified plane: " + str(plane))
    if not data_fmt in ['channels_last', 'channels_first']:
        raise ValueError("Did not understand specified data_fmt: " + str(plane))

    # define full paths
    data_files = [glob(study_dir + '/*' + contrast + '*.nii.gz')[0] for contrast in feature_prefx]
    labels_file = glob(study_dir + '/*' + label_prefx[0] + '*.nii.gz')[0]
    mask_file = glob(study_dir + '/*' + mask_prefx[0] + '*.nii.gz')[0]
    if not all([os.path.isfile(img) for img in data_files + [labels_file] + [mask_file]]):
        raise ValueError("One or more of the input data/labels/mask files does not exist")

    # load the mask and get the full data dims
    mask = nib.load(mask_file).get_fdata()
    data_dims = mask.shape

    # load data and normalize
    data = np.zeros((data_dims[0], data_dims[1], data_dims[2], len(data_files)))
    for i, im_file in enumerate(data_files):
        data[:, :, :, i] = _normalize(nib.load(im_file).get_fdata())

    # load labels
    labels = nib.load(labels_file).get_fdata()

    # center the ROI in the image usine affine, with optional rotation for data augmentation
    if aug == 'yes':  # if augmenting, select random rotation values for x, y, and z axes
        theta = 0.  # np.random.random() * (np.pi / 2.)  # rotation in yz plane
        phi = 0.  # np.random.random() * (np.pi / 2.)  # rotation in xz plane
        psi = np.random.random() * (np.pi / 2.)  # rotation in xy plane
    elif aug == 'no':  # if not augmenting, no rotation is applied, and affine is used only for offset to center the ROI
        theta = 0.
        phi = 0.
        psi = 0.
    else:
        raise ValueError("Augment data param argument must be yes or no but is: " + str(aug))

    # make affine, calculate offset using mask center of mass and affine
    affine = _create_affine(theta=theta, phi=phi, psi=psi)
    com = ndi.measurements.center_of_mass(mask)
    cent = np.array(mask.shape) / 2.
    offset = com - np.dot(affine, cent)

    # apply affines to mask, data, labels
    mask = _affine_transform(mask, affine=affine, offset=offset, order=0)  # nn interp for mask
    data = _affine_transform(data, affine=affine, offset=offset, order=1)  # linear interp for data
    labels = _affine_transform(labels, affine=affine, offset=offset, order=interp)  # user def interp for labels

    # get the tight bounding box of the mask after affine rotation
    msk_bbox = _nonzero_slice_inds3d(mask)

    # dilate bbox if necessary
    if dilate:
        msk_bbox = _expand_region(data_dims,msk_bbox, dilate)

    # determine new dim sizes
    dim_sizes = [msk_bbox[1] - msk_bbox[0], msk_bbox[3] - msk_bbox[2], msk_bbox[5] - msk_bbox[4]]

    # extract the region from the data
    data_region = np.zeros(dim_sizes + [len(data_files)])
    for i in range(len(data_files)):
        data_region[:, :, :, i] = data[msk_bbox[0]:msk_bbox[1], msk_bbox[2]:msk_bbox[3], msk_bbox[4]:msk_bbox[5], i]
    data = data_region
    labels = labels[msk_bbox[0]:msk_bbox[1], msk_bbox[2]:msk_bbox[3], msk_bbox[4]:msk_bbox[5]]

    # add batch and channel dims as necessary to get to [batch, x, y, z, channel]
    data = np.expand_dims(data, axis=0)  # add a batch dimension of 1
    labels = np.expand_dims(np.expand_dims(labels, axis=3), axis=0)  # add a batch and channel dimension of 1

    # handle different planes
    if plane == 'ax':
        pass
    elif plane == 'cor':
        data = np.transpose(data, axes=[0, 1, 3, 2, 4])
        labels = np.transpose(labels, axes=[0, 1, 3, 2, 4])
    elif plane == 'sag':
        data = np.transpose(data, axes=[0, 2, 3, 1, 4])
        labels = np.transpose(labels, axes=[0, 2, 3, 1, 4])
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_fmt == 'channels_first':
        data = np.transpose(data, axes=[0, 4, 1, 2, 3])
        labels = np.transpose(labels, axes=[0, 4, 1, 2, 3])

    return data.astype(np.float32), labels.astype(np.float32)


def _load_multicon_and_labels_3d(study_dir, feature_prefx, label_prefx, data_fmt, plane='ax'):
    """
    Load multicontrast image data and target data/labels without using a target roi. Used primarily for evaluation.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param label_prefx: (list) a list containing one string, the labels to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_dir): raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]): raise ValueError("Data prefixes must be strings")
    if not all([isinstance(a, str) for a in label_prefx]): raise ValueError("Labels prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']: raise ValueError("data_format invalid")

    # load multi-contrast data with slice trimming in z
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, norm=True, plane=plane)  # normalize input imgs

    # load labels data with slice trimming in z
    labels, nzi = _load_single_study(study_dir, label_prefx, data_format=data_fmt, slice_trim=nzi, plane=plane)

    # add batch dims as necessary to get to [batch, x, y, z, channel]
    data = np.expand_dims(data, axis=0)  # add a batch dimension of 1
    labels = np.expand_dims(labels, axis=0)  # add a batch dimension of 1

    # handle different planes
    if plane == 'ax':
        pass
    elif plane == 'cor':
        data = np.transpose(data, axes=[0, 1, 3, 2, 4])
        labels = np.transpose(labels, axes=[0, 1, 3, 2, 4])
    elif plane == 'sag':
        data = np.transpose(data, axes=[0, 2, 3, 1, 4])
        labels = np.transpose(labels, axes=[0, 2, 3, 1, 4])
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_fmt == 'channels_first':
        data = np.transpose(data, axes=[0, 4, 1, 2, 3])
        labels = np.transpose(labels, axes=[0, 4, 1, 2, 3])

    return data.astype(np.float32), labels.astype(np.float32)


def _load_multicon_preserve_size_3d(study_dir, feature_prefx, data_fmt, out_dims, plane):
    """
    Load multicontrast image data without cropping or otherwise adjusting size. For use with inference/prediction.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param out_dims: (list(int)) the desired output data dimensions, data will be zero padded to dims
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_dir): raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]): raise ValueError("Data prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']: raise ValueError("data_format invalid")
    if not all([np.issubdtype(a, np.integer) for a in out_dims]): raise ValueError(
        "data_dims must be a list/tuple of ints")

    # load multi-contrast data and normalize, no slice trimming for infer data
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, slice_trim=[0, None],
                                   norm=True, plane=plane)

    # generate batch size==1 format such that format is [1, x, y, z, c] or [1, c, x, y, z]
    data = np.expand_dims(data, axis=0)

    return data


##############################################
# TENSORFLOW MAP FUNCTIONS
##############################################


def _tf_patches(data, labels, patch_size, chan_dim, data_format, overlap=1):
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

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(data, perm=[0, 2, 3, 1])
        labels = tf.transpose(labels, perm=[0, 2, 3, 1])

    # make patches
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap , patch_size[1] / overlap, 1]
    rates = [1, 1, 1, 1]
    data = tf.extract_image_patches(data, ksizes=ksizes, strides=strides, rates=rates, padding='SAME')
    data = tf.reshape(data, [-1] + patch_size + [chan_dim])
    labels = tf.extract_image_patches(labels, ksizes=ksizes, strides=strides, rates=rates, padding='SAME')
    labels = tf.reshape(labels, [-1] + patch_size + [1])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(data, perm=[0, 3, 1, 2])
        labels = tf.transpose(labels, perm=[0, 3, 1, 2])

    return data, labels


def _tf_patches_3d(data, labels, patch_size, chan_dim, data_format, overlap=1):
    """
    Extract 3D patches from a data array with overlap if desired
    :param data: (numpy array) the data tensorflow tensor
    :param labels: (numpy array) the labels tensorflow tensor
    :param patch_size: (list or tupe of ints) the patch dimensions
    :param chan_dim:  (int) the number of data channels
    :param data_format: (str) either channels_last or channels_first - the tensorflow data format
    :param overlap: (int) the divisor for patch strides - determines the patch overlap in x, y (default no overlap)
    :return: returns tensorflow tensor patches
    """

    # sanity checks
    if not len(patch_size) == 3:
        raise ValueError("Patch size must be shape 3 to use 3D patch function but is: " + str(patch_size))

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(data, perm=[0, 2, 3, 4, 1])
        labels = tf.transpose(labels, perm=[0, 2, 3, 4, 1])

    # for sliding window 3d slabs, stride should be 1 in z dim, for x and y move 1/3 of the window
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap , patch_size[1] / overlap, 1, 1]

    # make patches
    data = tf.extract_volume_patches(data, ksizes=ksizes, strides=strides, padding='SAME')
    data = tf.reshape(data, [-1] + patch_size + [chan_dim])
    labels = tf.extract_volume_patches(labels, ksizes=ksizes, strides=strides, padding='SAME')
    labels = tf.reshape(labels, [-1] + patch_size + [1])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(data, perm=[0, 4, 1, 2, 3])
        labels = tf.transpose(labels, perm=[0, 4, 1, 2, 3])

    return data, labels


def _tf_patches_3d_infer(data, patch_size, chan_dim, data_format, overlap=1):
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

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(data, perm=[0, 2, 3, 4, 1])

    # for sliding window 3d slabs, stride should be 1 in z dim, for x and y move 1/3 of the window
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap , patch_size[1] / overlap, 1, 1]

    # make patches
    data = tf.extract_volume_patches(data, ksizes=ksizes, strides=strides, padding='SAME')
    data = tf.reshape(data, [-1] + patch_size + [chan_dim])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(data, perm=[0, 4, 1, 2, 3])

    return data


def _filter_zero_patches(data, data_format, mode, thresh=0.05):
    """
    Filters out patches that contain mostly zeros in the label data. Works for 3D and 2D patches.
    :param data: (list of tensors) must have {'labels'} key containing labels data
    :param data_format: (str) either 'channels_first' or 'channels_last' - the tensorflow data format
    :param mode: (str) either '2D' '2.5D' or '3D' - the mode for training
    :param thresh: (float) the threshold percentage for keeping patches. Default is 5%.
    :return: Returns tf.bool False if less than threshold, else returns tf.bool True
    """

    if data_format == 'channels_last':
        # handle channels last - if 2.5D get middle slice of labels, use entire slice if 2D, use entire slab if 3D
        if mode == '2.5D': # [x, y, z, c]
            mid_sl = data['labels'][:, :, data['labels'].get_shape()[2]/2 + 1, 0]
        elif mode == '2D': # [x, y, c]
            mid_sl = data['labels'][:, :, 0]
        elif mode == ' 3D':
            mid_sl = data['labels'][:, :, :, 0]
        else:
            raise ValueError("Mode must be 2D, 2.5D, or 3D but is: " + str(mode))
    else:
        # handle channels first - if 2.5D get middle slice of labels, use entire slice if 2D, use entire slab if 3D
        if mode == '2.5D':  # [c, x, y, z]
            mid_sl = data['labels'][0, :, :, data['labels'].get_shape()[2] / 2 + 1]
        elif mode == '2D':  # [c, x, y]
            mid_sl = data['labels'][0, :, :]
        elif mode == '3D':
            mid_sl = data['labels'][0, :, :, :]
        else:
            raise ValueError("Labels shape must be 2D or 3D but is: " + str((data['labels'].get_shape())))

    # eliminate if label slice is 95% empty
    thr = tf.constant(thresh, dtype=tf.float32)

    return tf.less(thr, tf.count_nonzero(mid_sl, dtype=tf.float32) / tf.size(mid_sl, out_type=tf.float32))


##############################################
# COMPLETE 2D INPUT FUNCTIONS
##############################################


def patch_input_fn(mode, params):
    """
    Input function for UCSF GBM dataset
    :param mode: (str) the model for running the model: 'train', 'eval', 'infer'
    :param params: (class) the params class generated from a JSON file
    :return: outputs, a dict containing the features, labels, and initializer operation
    """

    # Study dirs and prefixes setup
    study_dirs_filepath = os.path.join(params.model_dir, 'study_dirs_list.npy')
    if os.path.isfile(study_dirs_filepath):  # load study dirs file if it already exists for consistent training
        study_dirs = list(np.load(study_dirs_filepath))
    else:
        study_dirs = glob(params.data_dir + '/*/')
        study_dirs.sort()  # ensure study dirs is in alphabetical order and sorted in alphabetical order
        shuffle(study_dirs)  # randomly shuffle input directories for training
        np.save(study_dirs_filepath, study_dirs)  # save study dir list for later use to ensure consistency
    train_dirs = tf.constant(study_dirs[0:int(round(params.train_fract * len(study_dirs)))])
    eval_dirs = tf.constant(study_dirs[int(round(params.train_fract * len(study_dirs))):])

    # generate input dataset objects for the different training modes
    # train mode
    if mode == 'train':
        data_dirs = train_dirs
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.mask_prefix,
                          params.train_dims,
                          params.data_plane,
                          params.data_format,
                          params.augment_train_data,
                          params.label_interp]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.py_func(_load_roi_multicon_and_labels,
                                 [x] + py_func_params,
                                 (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)
        # map each dataset to a series of patches
        dataset = dataset.map(
            lambda x, y: _tf_patches(x, y, params.train_dims, len(params.data_prefix), params.data_format,
                                     overlap=params.train_patch_overlap),
            num_parallel_calls=params.num_threads)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
        # filter out zero patches
        dataset = dataset.filter(lambda x: _filter_zero_patches(x, params.data_format, params.dimension_mode))
        # shuffle a set number of exampes
        dataset = dataset.shuffle(buffer_size=params.shuffle_size)
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)

    # eval mode
    elif mode == 'eval':
        data_dirs = eval_dirs
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.data_format,
                          params.infer_dims]
        # map tensorflow dataset variable to data
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        dataset = dataset.map(
            lambda x: tf.py_func(_load_multicon_and_labels,
                                 [x] + py_func_params,
                                 (tf.float32, tf.float32)), num_parallel_calls=params.num_threads)
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
        dataset = dataset.prefetch(params.buffer_size)
        dataset = dataset.batch(params.batch_size, drop_remainder=True)

    # infer mode
    elif mode == 'infer':
        raise ValueError("Please use separate infer data input function.")
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    # make iterator and query the output of the iterator for input to the model
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    init_op = iterator.initializer

    # manually set shapes for inputs
    data_dims = []
    if mode == 'train':
        data_dims = list(params.train_dims)
    elif mode == 'eval':
        data_dims = list(params.infer_dims)
    get_next_features = get_next['features']
    get_next_labels = get_next['labels']
    # handle channels first
    if params.data_format == 'channels_last':
        get_next_features.set_shape([params.batch_size] + data_dims + [len(params.data_prefix)])
        get_next_labels.set_shape([params.batch_size] + data_dims + [len(params.label_prefix)])
    else:
        get_next_features.set_shape([params.batch_size] + [len(params.data_prefix)] + data_dims)
        get_next_labels.set_shape([params.batch_size] + [len(params.label_prefix)] + data_dims)

    # Build and return a dictionary containing the nodes / ops
    inputs = {'features': get_next_features, 'labels': get_next_labels, 'iterator_init_op': init_op}

    return inputs


def infer_input_fn(params, infer_dir):
    """
    Input function for UCSF GBM dataset
    :param params: (class) the params class generated from a JSON file
    :param infer_dir: (str) the directory for inference
    :return: outputs, a dict containing the features, labels, and initializer operation
    """

    # force batch size of 1 for inference
    batch_size = 1  # params.batch_size

    # prepare pyfunc
    data_dims = list(params.infer_dims)
    py_func_params = [params.data_prefix, params.data_format, data_dims, params.data_plane]

    # generate tensorflow dataset object from infer directories
    dataset = tf.data.Dataset.from_tensor_slices([infer_dir])
    # map infer dirs to data using custom python function
    dataset = dataset.map(
        lambda x: tf.py_func(_load_multicon_preserve_size,
                             [x] + py_func_params,
                             tf.float32), num_parallel_calls=params.num_threads)
    # flatten dataset to individual slices
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    # pad each slice to the desired infer dimensions
    padded_shapes = ([data_dims[0], data_dims[1], len(params.data_prefix)])
    # batch to specified batch size
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=0.)

    # make iterator and query the output of the iterator for input to the model
    iterator = dataset.make_initializable_iterator()
    get_next_features = iterator.get_next()
    init_op = iterator.initializer

    # manually set shapes for inputs
    get_next_features.set_shape([batch_size, data_dims[0], data_dims[1], len(params.data_prefix)])

    # Build and return a dictionary containing the nodes / ops
    inputs = {'features': get_next_features, 'iterator_init_op': init_op}

    return inputs


##############################################
# COMPLETE 3D INPUT FUNCTIONS
##############################################


def patch_input_fn_3d(mode, params):
    """
    3d patch based input function
    :param mode: (str) the model for running the model: 'train', 'eval', 'infer'
    :param params: (class) the params class generated from a JSON file
    :return: outputs, a dict containing the features, labels, and initializer operation
    """

    # Study dirs and prefixes setup
    study_dirs_filepath = os.path.join(params.model_dir, 'study_dirs_list.npy')
    if os.path.isfile(study_dirs_filepath):  # load study dirs file if it already exists for consistent training
        study_dirs = list(np.load(study_dirs_filepath))
    else:
        study_dirs = glob(params.data_dir + '/*/')
        study_dirs.sort()  # ensure study dirs is in alphabetical order and sorted in alphabetical order
        shuffle(study_dirs)  # randomly shuffle input directories for training
        np.save(study_dirs_filepath, study_dirs)  # save study dir list for later use to ensure consistency
    train_dirs = tf.constant(study_dirs[0:int(round(params.train_fract * len(study_dirs)))])
    eval_dirs = tf.constant(study_dirs[int(round(params.train_fract * len(study_dirs))):])

    # generate input dataset objects for the different training modes
    # train mode
    if mode == 'train':
        data_dirs = train_dirs
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.mask_prefix,
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          params.augment_train_data,
                          params.label_interp]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.py_func(_load_roi_multicon_and_labels_3d,
                                 [x] + py_func_params,
                                 (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)
        # map each dataset to a series of patches with overlap of 1/3
        dataset = dataset.map(
            lambda x, y: _tf_patches_3d(x, y, params.train_dims, len(params.data_prefix), params.data_format,
                                        overlap=params.train_patch_overlap),
            num_parallel_calls=params.num_threads)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
        # filter out zero patches
        dataset = dataset.filter(lambda x: _filter_zero_patches(x, params.data_format, params.dimension_mode))
        # shuffle a set number of exampes
        dataset = dataset.shuffle(buffer_size=params.shuffle_size)
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)

    # eval mode
    elif mode == 'eval':
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.data_format,
                          params.data_plane]
        # create tensorflow dataset variable from eval data directories
        dataset = tf.data.Dataset.from_tensor_slices(eval_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.py_func(_load_multicon_and_labels_3d,
                                 [x] + py_func_params,
                                 (tf.float32, tf.float32)), num_parallel_calls=params.num_threads)
        # map each dataset to a series of patches - no overlap between patches (default)
        dataset = dataset.map(
            lambda x, y: _tf_patches_3d(x, y, params.infer_dims, len(params.data_prefix), params.data_format),
            num_parallel_calls=params.num_threads)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)

    # infer mode
    elif mode == 'infer':
        raise ValueError("Please use separate infer data input function.")
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    # make iterator and query the output of the iterator for input to the model
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    init_op = iterator.initializer

    # manually set shapes for inputs
    data_dims = []
    if mode == 'train':
        data_dims = list(params.train_dims)
    elif mode == 'eval':
        data_dims = list(params.infer_dims)
    get_next_features = get_next['features']
    get_next_labels = get_next['labels']
    # handle channels first
    if params.data_format == 'channels_last':
        get_next_features.set_shape([params.batch_size] + data_dims + [len(params.data_prefix)])
        get_next_labels.set_shape([params.batch_size] + data_dims + [len(params.label_prefix)])
    else:
        get_next_features.set_shape([params.batch_size] + [len(params.data_prefix)] + data_dims)
        get_next_labels.set_shape([params.batch_size] + [len(params.label_prefix)] + data_dims)

    # Build and return a dictionary containing the nodes / ops
    inputs = {'features': get_next_features, 'labels': get_next_labels, 'iterator_init_op': init_op}

    return inputs


def infer_input_fn_3d(params, infer_dir):
    """
    Input function for UCSF GBM dataset
    :param params: (class) the params class generated from a JSON file
    :param infer_dir: (str) the directory for inference
    :return: outputs, a dict containing the features, labels, and initializer operation
    """

    # force batch size of 1 for inference
    batch_size = 1  # params.batch_size

    # prepare pyfunc
    data_dims = list(params.infer_dims)
    chan_size = len(params.data_prefix)
    py_func_params = [params.data_prefix, params.data_format, data_dims, params.data_plane]

    # generate tensorflow dataset object from infer directory
    dataset = tf.data.Dataset.from_tensor_slices([infer_dir])
    # map infer directory to data using a custom python function
    dataset = dataset.map(
        lambda x: tf.py_func(_load_multicon_preserve_size_3d,
                             [x] + py_func_params,
                             tf.float32), num_parallel_calls=params.num_threads)
    # extract 3D patches from the infer data - force no overlap for infer
    dataset = dataset.map(
        lambda x: _tf_patches_3d_infer(x, data_dims, chan_size, params.data_format, overlap=1),
        num_parallel_calls=params.num_threads)
    # flatten patches to individual examples
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    # batch data to batch size 1 (forced above at beginning of function)
    dataset = dataset.batch(batch_size)

    # make iterator and query the output of the iterator for input to the model
    iterator = dataset.make_initializable_iterator()
    get_next_features = iterator.get_next()
    init_op = iterator.initializer

    # manually set shapes for inputs
    if params.data_format == 'channels_last':
        get_next_features.set_shape([batch_size] + data_dims + [chan_size])
    else:
        get_next_features.set_shape([batch_size] + [chan_size] + data_dims)

    # Build and return a dictionary containing the nodes / ops
    inputs = {'features': get_next_features, 'iterator_init_op': init_op}

    return inputs