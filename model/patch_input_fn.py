import os
import math
from glob import glob
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import tensorflow as tf
from random import shuffle


# DATA UTILITIES
def _load_single_study(study_dir, file_prefixes, data_format, slice_trim=None, plane=None, norm=False,
                       norm_mode='zero_mean'):
    """
    Image data I/O function for use in tensorflow Dataset map function. Takes a study directory and file prefixes and
    returns a 4D numpy array containing the image data. Performs optional slice trimming in z and normalization.
    :param study_dir: (str) the full path to the study directory
    :param file_prefixes: (str, list(str)) the file prefixes for the images to be loaded
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param slice_trim: (list, tuple) contains 2 ints, the first and last slice to use for trimming. None = auto trim.
                        [0, -1] does no trimming
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param norm: (bool) whether or not to perform per dataset normalization
    :param norm_mode: (str) The method for normalization, used by _normalize function.
    :return: output - a 4D numpy array containing the image data
    """

    # sanity checks
    if not os.path.isdir(study_dir):
        raise ValueError("Specified study_dir does not exist")
    if data_format not in ['channels_last', 'channels_first']:
        raise ValueError("data_format invalid")
    if slice_trim is not None and not isinstance(slice_trim, (list, tuple)):
        raise ValueError("slice_trim must be list/tuple")
    images = [glob(study_dir + '/*' + contrast + '.nii.gz')[0] for contrast in file_prefixes]
    if not images:
        raise ValueError("No matching image files found for file prefixes: " + str(images))

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
                first_image = _normalize(first_image, norm_mode)
            output_shape = list(first_image.shape)[0:3] + [len(images)]
            output = np.zeros(output_shape, np.float32)
            output[:, :, :, 0] = first_image
        else:
            img = nib.load(images[ind]).get_fdata()[:, :, nz_inds[0]:nz_inds[1]]
            # do normalization
            if norm:
                img = _normalize(img, norm_mode)
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
    Symmetrically expands a given 3D region bounding box by delta in each dim without exceeding original image dims
    If delta is a single int then each dim is expanded by this amount. If a list or tuple, of ints then dims are
    expanded to match the size of each int in the list respectively.
    :param input_dims: (list or tuple of ints) the original image dimensions
    :param region_bbox: (list or tuple of ints) the region bounding box to expand
    :param delta: (int or list/tuple of ints) the amount to expand each dimension by.
    :return: (list or tuple of ints) the expanded bounding box
    """

    # if delta is actually a list, then refer to _expand_region_dims, make sure its same length as input_dims
    if isinstance(delta, (list, tuple, np.ndarray)):
        if not len(delta) == len(input_dims):
            raise ValueError("When a list is passed, parameter mask_dilate must have one val for each dim in mask")
        return _expand_region_dims(input_dims, region_bbox, delta)

    # determine how much to add on each side of the bounding box
    deltas = np.array([-int(np.floor(delta/2.)), int(np.ceil(delta/2.))] * 3)

    # use deltas to get a new bounding box
    tmp_bbox = np.array(region_bbox) + deltas

    # make sure there are not values outside of the original image
    new_bbox = []
    for i, item in enumerate(tmp_bbox):
        if i % 2 == 0:  # for even indices, make sure there are no negatives
            if item < 0:
                item = 0
        else:  # for odd indices, make sure they do not exceed original dims
            if item > input_dims[(i-1)/2]:
                item = input_dims[(i-1)/2]
        new_bbox.append(item)

    return new_bbox


def _expand_region_dims(input_dims, region_bbox, out_dims):
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
    pre_inds = np.array([-int(np.floor(d/2.)) for d in deltas])
    post_inds = np.array([int(np.ceil(d/2.)) for d in deltas])
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
            if item > input_dims[int(round((i-1)/2))]:
                item = input_dims[int(round((i-1)/2))]
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

    # define epsilon for divide by zero errors
    epsilon = 1e-10

    # handle zero mean mode
    if mode == 'zero_mean':
        # perform normalization to zero mean unit variance
        nzi = np.nonzero(input_img)
        mean = np.mean(input_img[nzi], None)
        std = np.std(input_img[nzi], None) + epsilon
        input_img = np.where(input_img != 0., ((input_img - mean) / std), 0.)  # add 10 to prevent negatives

    # handle ten mean mode
    elif mode == 'ten_mean':
        # perform normalization to zero mean unit variance
        nzi = np.nonzero(input_img)
        mean = np.mean(input_img[nzi], None)
        std = np.std(input_img[nzi], None) + epsilon
        input_img = np.where(input_img != 0., ((input_img - mean) / std) + 10., 0.)  # add 10 to prevent negatives

    # handle unit mode
    elif mode == 'unit':
        # perform normalization to [0, 1]
        input_img *= 1.0 / np.max(input_img)

    # handle not implemented
    else:
        raise NotImplementedError("Specified normalization mode is not implemented yet: " + mode)

    return input_img


def _nonzero_slice_inds3d(input_numpy):
    """
    Takes numpy array and returns slice indices of first and last nonzero pixels in 3d
    :param input_numpy: (np.ndarray) a numpy array containing image data.
    :return: inds - a list of 2 indices per dimension corresponding to the first and last nonzero slices in the array
    """

    # sanity checks
    if type(input_numpy) is not np.ndarray:
        raise ValueError("Input must be numpy array")

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
            pad = [int(math.ceil(total_pad/2.)), int(math.floor(total_pad/2.))]
        pads.append(pad)

    # pad array with zeros (default)
    input_data = np.pad(input_data, pads, 'constant')

    return input_data


##############################################
# PY FUNC DATA FUNCTIONS
##############################################


def byte_convert(byte_data):
    if isinstance(byte_data, bytes):  return byte_data.decode()
    if isinstance(byte_data, dict):   return dict(map(byte_convert, byte_data.items()))
    if isinstance(byte_data, tuple):  return map(byte_convert, byte_data)
    if isinstance(byte_data, (np.ndarray, list)):   return list(map(byte_convert, byte_data))

    return byte_data


def _load_multicon_and_labels(study_dir, feature_prefx, label_prefx, data_fmt, out_dims, plane='ax', norm=True,
                              norm_lab=True, norm_mode='zero_mean'):
    """
    Load multicontrast image data and target data/labels.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param label_prefx: (list) a list containing one string, the labels to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param out_dims: (list(int)) the desired output data dimensions, data will be zero padded to dims
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param norm: (bool) Whether or not to normalize the input data after loading.
    :param norm_lab: (bool) whether or not to normalize the label data after loading.
    :param norm_mode: (str) The method for normalization, used by _normalize function.
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # convert bytes to strings
    study_dir = byte_convert(study_dir)
    feature_prefx = byte_convert(feature_prefx)
    label_prefx = byte_convert(label_prefx)
    plane = byte_convert(plane)
    data_fmt = byte_convert(data_fmt)
    norm_mode = byte_convert(norm_mode)

    # sanity checks
    if not os.path.isdir(study_dir):
        raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]):
        raise ValueError("Data prefixes must be strings")
    if not all([isinstance(a, str) for a in label_prefx]):
        raise ValueError("Labels prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']:
        raise ValueError("data_format invalid")
    if not all([np.issubdtype(a, np.integer) for a in out_dims]):
        raise ValueError("data_dims must be a list/tuple of ints")

    # load multi-contrast data - normalize input images
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, plane=plane, norm=norm,
                                   norm_mode=norm_mode)

    # load labels data
    labels, nzi = _load_single_study(study_dir, label_prefx, data_format=data_fmt, slice_trim=nzi, plane=plane,
                                     norm=norm_lab, norm_mode=norm_mode)

    # do data padding to desired dims - format is [x, y, z, c] or [c, x, y, z]
    axes = [1, 2] if data_fmt == 'channels_first' else [0, 1]
    data = _zero_pad_image(data, out_dims, axes)
    labels = _zero_pad_image(labels, out_dims, axes)

    # note that load single study handles normalization, image plane, and data format, but does not make z batch dim
    if data_fmt == 'channels_first':
        data = np.transpose(data, axes=(3, 0, 1, 2))
        labels = np.transpose(labels, axes=(3, 0, 1, 2))
    else:
        data = np.transpose(data, axes=(2, 0, 1, 3))
        labels = np.transpose(labels, axes=(2, 0, 1, 3))

    return data, labels


def _load_multicon_preserve_size(study_dir, feature_prefx, data_fmt, plane, norm=True, norm_mode='zero_mean'):
    """
    Load multicontrast image data without cropping or otherwise adjusting size. For use with inference/prediction.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param norm: (bool) Whether or not to normalize the input data after loading.
    :param norm_mode: (str) The method for normalization, used by _normalize function.
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_dir):
        raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]):
        raise ValueError("Data prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']:
        raise ValueError("data_format invalid")

    # load multi-contrast data and normalize, no slice trimming for infer data
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, slice_trim=[0, None], plane=plane,
                                   norm=norm, norm_mode=norm_mode)

    # transpose slices to batch dimension format such that format is [z, x, y, c] or [z, c, x, y]
    axes = (3, 0, 1, 2) if data_fmt == 'channels_first' else (2, 0, 1, 3)
    data = np.transpose(data, axes=axes)

    return data


def _load_roi_multicon_and_labels(study_dir, feature_prefx, label_prefx, mask_prefx, dilate=0, plane='ax',
                                  data_fmt='channels_last', aug='no', interp=1, norm=True, norm_lab=True,
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
    :param aug: (str) 'yes' or 'no' - Whether or not to perform data augmentation with random 3D affine rotation.
    :param interp: (int) The order of spline interpolation for label data. Must be 0-5
    :param norm: (bool) Whether or not to normalize the input data after loading.
    :param norm_lab: (bool) whether or not to normalize the label data after loading.
    :param norm_mode: (str) The method for normalization, used by _normalize function.
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
    aug = byte_convert(aug)

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
    data = np.zeros((data_dims[0], data_dims[1], data_dims[2], len(data_files)))
    for i, im_file in enumerate(data_files):
        if norm:
            data[:, :, :, i] = _normalize(nib.load(im_file).get_fdata(), mode=norm_mode)
        else:
            data[:, :, :, i] = nib.load(im_file).get_fdata()

    # load labels
    if norm_lab:
        labels = _normalize(nib.load(labels_file).get_fdata(), mode=norm_mode)
    else:
        labels = nib.load(labels_file).get_fdata()

    # center the tumor in the image usine affine, with optional rotation for data augmentation
    if aug == 'yes':  # if augmenting, select random rotation values for x, y, and z axes
        theta = np.random.random() * (np.pi / 2.) if plane == 'cor' else 0.  # rotation in yz plane
        phi = np.random.random() * (np.pi / 2.) if plane == 'sag' else 0.  # rotation in xz plane
        psi = np.random.random() * (np.pi / 2.) if plane == 'ax' else 0.  # rotation in xy plane
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

    # dilate bbox if necessary - this also ensures that the bbox does not exceed original image dims
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
                                     data_fmt='channels_last', aug='no', interp=1, norm=True, norm_lab=True,
                                     norm_mode='zero_mean'):
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
    :param aug: (str) Either yes or no - Whether or not to perform data augmentation with random 3D affine rotation.
    :param interp: (int) The order of spline interpolation for label data. Must be 0-5
    :param norm: (bool) Whether or not to normalize the input data after loading.
    :param norm_lab: (bool) whether or not to normalize the label data after loading.
    :param norm_mode: (str) The method for normalization, used by _normalize function.
    :return: (tf.tensor) The patch data for features and labels as a tensorflow variable.
    """

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
        mask = (nib.load(mask_file).get_fdata() > 0.).astype(float)
    else:
        mask = np.ones_like(nib.load(mask_file).get_fdata(), dtype=float)
    data_dims = mask.shape

    # load data and normalize
    data = np.zeros((data_dims[0], data_dims[1], data_dims[2], len(data_files)))
    for i, im_file in enumerate(data_files):
        if norm:
            data[:, :, :, i] = _normalize(nib.load(im_file).get_fdata(), mode=norm_mode)
        else:
            data[:, :, :, i] = nib.load(im_file).get_fdata()

    # load labels with NORMALIZATION - add option for normalized vs non-normalized labels here
    if norm_lab:
        labels = _normalize(nib.load(labels_file).get_fdata(), mode=norm_mode)
    else:
        labels = nib.load(labels_file).get_fdata()

    # center the ROI in the image usine affine, with optional rotation for data augmentation
    if aug == 'yes':  # if augmenting, select random rotation values for x, y, and z axes
        theta = np.random.random() * (np.pi / 2.) if plane == 'cor' else 0.  # rotation in yz plane
        phi = np.random.random() * (np.pi / 2.) if plane == 'sag' else 0.  # rotation in xz plane
        psi = np.random.random() * (np.pi / 2.) if plane == 'ax' else 0.  # rotation in xy plane
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

    # dilate bbox if necessary - this also ensures that the bbox does not exceed original image dims
    msk_bbox = _expand_region(data_dims, msk_bbox, dilate)

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


def _load_multicon_and_labels_3d(study_dir, feature_prefx, label_prefx, data_fmt, plane='ax', norm=True, norm_lab=True,
                                 norm_mode='zero_mean'):
    """
    Load multicontrast image data and target data/labels without using a target roi. Used primarily for evaluation.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param label_prefx: (list) a list containing one string, the labels to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param norm: (bool) Whether or not to normalize the input data after loading.
    :param norm_lab: (bool) whether or not to normalize the label data after loading.
    :param norm_mode: (str) The method for normalization, used by _normalize function.
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_dir):
        raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]):
        raise ValueError("Data prefixes must be strings")
    if not all([isinstance(a, str) for a in label_prefx]):
        raise ValueError("Labels prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']:
        raise ValueError("data_format invalid")

    # load multi-contrast data with slice trimming in z - normalize input images
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, plane=plane, norm=norm,
                                   norm_mode=norm_mode)

    # load labels data with slice trimming in z
    labels, nzi = _load_single_study(study_dir, label_prefx, data_format=data_fmt, slice_trim=nzi, plane=plane,
                                     norm=norm_lab, norm_mode=norm_mode)

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


def _load_multicon_preserve_size_3d(study_dir, feature_prefx, data_fmt, plane='ax', norm=True, norm_mode='zero_mean'):
    """
    Load multicontrast image data without cropping or otherwise adjusting size. For use with inference/prediction.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param norm: (bool) Whether or not to normalize the input data after loading
    :param norm_mode: (str) The method for normalization, used by _normalize function.
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_dir):
        raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]):
        raise ValueError("Data prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']:
        raise ValueError("data_format invalid")

    # load multi-contrast data and normalize, no slice trimming for infer data
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, slice_trim=[0, None], plane=plane,
                                   norm=norm, norm_mode=norm_mode)

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


def _tf_patches_infer(data, patch_size, chan_dim, data_format, overlap=1):
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
    rates = [1, 1, 1, 1]
    data = tf.image.extract_patches(data, sizes=ksizes, strides=strides, rates=rates, padding='SAME')
    data = tf.reshape(data, [-1] + patch_size + [chan_dim])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 3, 1, 2])

    return data


def _tf_patches_3d(data, labels, patch_size, chan_dim, data_format, overlap=1):
    """
    Extract 3D patches from a data array with overlap if desired
    :param data: (numpy array) the data tensorflow tensor
    :param labels: (numpy array) the labels tensorflow tensor
    :param patch_size: (list or tupe of ints) the patch dimensions
    :param chan_dim:  (int) the number of data channels
    :param data_format: (str) either channels_last or channels_first - the tensorflow data format
    :param overlap: (int or list/tuple of ints) the divisor for patch strides - determines the patch overlap in x, y
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
        labels = tf.transpose(a=labels, perm=[0, 2, 3, 4, 1])

    # for sliding window 3d slabs
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap[0], patch_size[1] / overlap[1], patch_size[2] / overlap[2], 1]

    # make patches
    data = tf.extract_volume_patches(data, ksizes=ksizes, strides=strides, padding='SAME')
    data = tf.reshape(data, [-1] + patch_size + [chan_dim])
    labels = tf.extract_volume_patches(labels, ksizes=ksizes, strides=strides, padding='SAME')
    labels = tf.reshape(labels, [-1] + patch_size + [1])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 4, 1, 2, 3])
        labels = tf.transpose(a=labels, perm=[0, 4, 1, 2, 3])

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

    # make patches
    data = tf.extract_volume_patches(data, ksizes=ksizes, strides=strides, padding='SAME')
    data = tf.reshape(data, [-1] + patch_size + [chan_dim])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 4, 1, 2, 3])

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
    if thresh == 0.:
        return tf.constant(True, dtype=tf.bool)

    if data_format == 'channels_last':
        # handle channels last - use entire slice if 2D, use entire slab if 3D or 2.5D
        if mode == '2.5D':  # [x, y, z, c]
            # mid_sl = data['labels'][:, :, data['labels'].get_shape()[2]/2 + 1, 0]
            mid_sl = data['labels'][:, :, :, 0]
        elif mode == '2D':  # [x, y, c]
            mid_sl = data['labels'][:, :, 0]
        elif mode == '3D':
            mid_sl = data['labels'][:, :, :, 0]
        else:
            raise ValueError("Mode must be 2D, 2.5D, or 3D but is: " + str(mode))
    else:
        # handle channels first - use entire slice if 2D, use entire slab if 3D or 2.5D
        if mode == '2.5D':  # [c, x, y, z]
            # mid_sl = data['labels'][0, :, :, data['labels'].get_shape()[2] / 2 + 1]
            mid_sl = data['labels'][0, :, :, :]
        elif mode == '2D':  # [c, x, y]
            mid_sl = data['labels'][0, :, :]
        elif mode == '3D':
            mid_sl = data['labels'][0, :, :, :]
        else:
            raise ValueError("Labels shape must be 2D or 3D but is: " + str((data['labels'].get_shape())))

    # eliminate if label slice is 95% empty
    thr = tf.constant(thresh, dtype=tf.float32)

    return tf.less(thr, tf.math.count_nonzero(mid_sl, dtype=tf.float32) / tf.size(input=mid_sl, out_type=tf.float32))


# COMPLETE 2D INPUT FUNCTIONS
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
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          params.augment_train_data,
                          params.label_interp,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.compat.v1.py_func(_load_roi_multicon_and_labels,
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
        dataset = dataset.filter(lambda x: _filter_zero_patches(
            x, params.data_format, params.dimension_mode, params.filter_zero))
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
                          params.infer_dims,
                          params.data_plane,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode]
        # map tensorflow dataset variable to data
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        dataset = dataset.map(
            lambda x: tf.compat.v1.py_func(_load_multicon_and_labels,
                                           [x] + py_func_params,
                                           (tf.float32, tf.float32)), num_parallel_calls=params.num_threads)
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
        dataset = dataset.batch(params.batch_size, drop_remainder=True)

    # infer mode
    elif mode == 'infer':
        raise ValueError("Please use separate infer data input function.")
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    # make iterator and query the output of the iterator for input to the model
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
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
    Input function for inferring 2d patches
    :param params: (class) the params class generated from a JSON file
    :param infer_dir: (str) the directory for inference
    :return: outputs, a dict containing the features, labels, and initializer operation
    """

    # force batch size of 1 for inference
    batch_size = 1  # params.batch_size

    # prepare pyfunc
    data_dims = list(params.infer_dims)
    chan_size = len(params.data_prefix)
    py_func_params = [params.data_prefix, params.data_format, params.data_plane, params.norm_data, params.norm_mode]

    # generate tensorflow dataset object from infer directory
    dataset = tf.data.Dataset.from_tensor_slices([infer_dir])
    # map infer directory to data using a custom python function
    dataset = dataset.map(
        lambda x: tf.compat.v1.py_func(_load_multicon_preserve_size,
                                       [x] + py_func_params,
                                       tf.float32), num_parallel_calls=params.num_threads)
    # extract 3D patches from the infer data using same overlap
    dataset = dataset.map(
        lambda x: _tf_patches_infer(x, data_dims, chan_size, params.data_format, params.infer_patch_overlap),
        num_parallel_calls=params.num_threads)
    # flatten patches to individual examples
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    # batch data to batch size 1 (forced above at beginning of function)
    dataset = dataset.batch(batch_size)

    # make iterator and query the output of the iterator for input to the model
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
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
        train_dims = list(params.train_dims)
        chan_size = len(params.data_prefix)
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.mask_prefix,
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          params.augment_train_data,
                          params.label_interp,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.compat.v1.py_func(_load_roi_multicon_and_labels_3d,
                                           [x] + py_func_params,
                                           (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)
        # map each dataset to a series of patches with overlap of 1/3
        dataset = dataset.map(
            lambda x, y: _tf_patches_3d(x, y, train_dims, chan_size, params.data_format,
                                        overlap=params.train_patch_overlap),
            num_parallel_calls=params.num_threads)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
        # filter out zero patches
        dataset = dataset.filter(lambda x: _filter_zero_patches(
            x, params.data_format, params.dimension_mode, params.filter_zero))
        # shuffle a set number of exampes
        dataset = dataset.shuffle(buffer_size=params.shuffle_size)
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)

    # eval mode
    elif mode == 'eval':
        # prepare pyfunc params
        infer_dims = list(params.infer_dims)
        chan_size = len(params.data_prefix)
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.data_format,
                          params.data_plane,
                          params.norm_data,
                          params.norm_mode,
                          params.norm_mode]
        # create tensorflow dataset variable from eval data directories
        dataset = tf.data.Dataset.from_tensor_slices(eval_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.compat.v1.py_func(_load_multicon_and_labels_3d,
                                           [x] + py_func_params,
                                           (tf.float32, tf.float32)), num_parallel_calls=params.num_threads)
        # map each dataset to a series of patches - no overlap between patches (default)
        dataset = dataset.map(
            lambda x, y: _tf_patches_3d(x, y, infer_dims, chan_size, params.data_format, overlap=1),
            num_parallel_calls=params.num_threads)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
        # filter out zero patches because they wont contribute much to evaluation metrics
        dataset = dataset.filter(lambda x: _filter_zero_patches(
            x, params.data_format, params.dimension_mode, params.filter_zero))
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)

    # infer mode
    elif mode == 'infer':
        raise ValueError("Please use separate infer data input function.")
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    # make iterator and query the output of the iterator for input to the model
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
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
    Input function for inferring 3d patches
    :param params: (class) the params class generated from a JSON file
    :param infer_dir: (str) the directory for inference
    :return: outputs, a dict containing the features, labels, and initializer operation
    """

    # force batch size of 1 for inference
    batch_size = 1  # params.batch_size

    # prepare pyfunc
    data_dims = list(params.infer_dims)
    chan_size = len(params.data_prefix)
    py_func_params = [params.data_prefix, params.data_format, params.data_plane, params.norm_data, params.norm_mode]

    # generate tensorflow dataset object from infer directory
    dataset = tf.data.Dataset.from_tensor_slices([infer_dir])
    # map infer directory to data using a custom python function
    dataset = dataset.map(
        lambda x: tf.compat.v1.py_func(_load_multicon_preserve_size_3d,
                                       [x] + py_func_params,
                                       tf.float32), num_parallel_calls=params.num_threads)
    # extract 3D patches from the infer data using same overlap
    dataset = dataset.map(
        lambda x: _tf_patches_3d_infer(x, data_dims, chan_size, params.data_format, params.infer_patch_overlap),
        num_parallel_calls=params.num_threads)
    # flatten patches to individual examples
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    # batch data to batch size 1 (forced above at beginning of function)
    dataset = dataset.batch(batch_size)

    # make iterator and query the output of the iterator for input to the model
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
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
        _x = tf.zeros_like(x)
        _y = extract_patches(_x)
        grad = tf.gradients(ys=_y, xs=_x)[0]
        # Divide by grad, to "average" together the overlapping patches
        # otherwise they would simply sum up
        return tf.gradients(ys=_y, xs=_x, grad_ys=y)[0] / grad

    # load original data but convert to only one channel to match output [batch, x, y, z, channel]
    data = _load_multicon_preserve_size(infer_dir, data_prefix, data_format, data_plane)
    # data = data[:, :, :, [1]] if params.data_format == 'channels_last' else data[:, [1], :, :]

    # get shape of patches as they would have been generated during inference
    dummy_shape = tf.shape(input=extract_patches(data))

    # convert channels dimension to actual output_filters
    if params.data_format == 'channels_last':
        dummy_shape = dummy_shape[:-1] + [params.output_filters]
    else:
        dummy_shape = [dummy_shape[0], params.output_filters] + dummy_shape[2:]

    # reshape predictions to original patch shape
    predictions = tf.reshape(predictions, dummy_shape)

    # handle argmax
    predictions = tf.argmax(input=tf.nn.softmax(predictions, axis=-1), axis=-1)

    # reconstruct
    reconstructed = extract_patches_inverse(data, predictions)
    with tf.compat.v1.Session() as sess:
        output = np.squeeze(reconstructed.eval(session=sess))

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
        _x = tf.zeros_like(x)
        _y = extract_patches(_x)
        grad = tf.gradients(ys=_y, xs=_x)[0]
        # Divide by grad, to "average" together the overlapping patches
        # otherwise they would simply sum up
        return tf.gradients(ys=_y, xs=_x, grad_ys=y)[0] / grad

    # load original data but convert to only one channel to match output [batch, x, y, z, channel]
    data = _load_multicon_preserve_size_3d(infer_dir, data_prefix, data_format, data_plane, norm, norm_mode)
    data = data[:, :, :, :, [1]] if params.data_format == 'channels_last' else data[:, [1], :, :, :]

    # get shape of patches as they would have been generated during inference
    dummy_patches = extract_patches(data)

    # reshape predictions to original patch shape
    predictions = tf.reshape(predictions, tf.shape(input=dummy_patches))

    # reconstruct
    reconstructed = extract_patches_inverse(data, predictions)
    with tf.compat.v1.Session() as sess:
        output = np.squeeze(reconstructed.eval(session=sess))

    return output


# define the gradient for ExtractVolumePatches - this allows reversal of patches for inferance (only required pre 1.13)
"""
if int(tf.__version__.split('.')[1]) < 13:
    from tensorflow.python.framework import ops
    from tensorflow.python.framework import sparse_tensor
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import sparse_ops
    from tensorflow.python.ops import gen_array_ops
    from tensorflow.python.ops import math_ops

    @ops.RegisterGradient("ExtractVolumePatches")
    def _ExtractVolumePatchesGrad(op, grad):
      batch_size, planes_in, rows_in, cols_in, channels = [
          dim.value for dim in op.inputs[0].shape.dims
      ]
      input_bphwc = array_ops.shape(op.inputs[0])
      batch_size = input_bphwc[0]
      channels = input_bphwc[4]

      # Create indices matrix for input tensor.
      # Note that 0 is preserved for padding location,
      # so indices for input start from 1 to 1 + rows_in * cols_in.
      input_indices_num = 1 + planes_in * rows_in * cols_in
      input_idx = array_ops.reshape(math_ops.range(1, input_indices_num,
                                                   dtype=ops.dtypes.int64),
                                    (1, planes_in, rows_in, cols_in, 1))
      input_idx_patched = gen_array_ops.extract_volume_patches(
          input_idx,
          op.get_attr("ksizes"),
          op.get_attr("strides"),
          op.get_attr("padding"))

      # Create indices matrix for output tensor.
      _, planes_out, rows_out, cols_out, _ = [dim.value for dim in
                                              op.outputs[0].shape.dims]
      _, ksize_p, ksize_r, ksize_c, _ = op.get_attr("ksizes")
      # Indices for output start from 0.
      prc_indices_num = planes_out * rows_out * cols_out
      output_indices_num = prc_indices_num * ksize_p * ksize_r * ksize_c
      output_idx = array_ops.reshape(math_ops.range(output_indices_num,
                                                    dtype=ops.dtypes.int64),
                                     (1, planes_out, rows_out, cols_out,
                                      ksize_p * ksize_r * ksize_c))

      # Construct mapping table for indices: (input -> output).
      idx_matrix = array_ops.concat(
          [array_ops.expand_dims(input_idx_patched, axis=-1),
           array_ops.expand_dims(output_idx, axis=-1)],
          axis=-1)
      idx_map = array_ops.reshape(idx_matrix, (-1, 2))

      sp_shape = (input_indices_num, output_indices_num)
      sp_mat_full = sparse_tensor.SparseTensor(
          idx_map,
          array_ops.ones([output_indices_num], dtype=grad.dtype),
          sp_shape)
      # Remove all padding locations [0, :].
      sp_mat = sparse_ops.sparse_slice(sp_mat_full,
                                       (1, 0),
                                       (input_indices_num - 1, output_indices_num))

      grad_expanded = array_ops.transpose(
          array_ops.reshape(
              grad, (batch_size, planes_out, rows_out, cols_out, ksize_p, ksize_r,
                     ksize_c, channels)),
          (1, 2, 3, 4, 5, 6, 0, 7))
      grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

      jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

      grad_out = array_ops.reshape(jac, (planes_in, rows_in, cols_in, batch_size,
                                         channels))
      grad_out = array_ops.transpose(grad_out, (3, 0, 1, 2, 4))

      return [grad_out]
"""
