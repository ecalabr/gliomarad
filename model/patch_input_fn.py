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

    # permute to desired plane in format [batch, x, y, channels] for tensorflow
    if plane == 'ax':
        output = np.transpose(output, axes=(2, 0, 1, 3))
    elif plane == 'cor':
        output = np.transpose(output, axes=(1, 0, 2, 3))
    elif plane == 'sag':
        #output = np.transpose(output, axes=(0, 1, 2, 3))
        pass
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_format == 'channels_first':
        output = np.transpose(output, axes=(0, 3, 1, 2))

    return output, nz_inds


def _extract_region(input_img, region_bbox):
    """
    Extracts a region defined by region_bbox from the input_img with zero padding if needed.
    :param input_img: (np.ndarray) The image to extract the region from.
    :param region_bbox: (array like) The indices of the bounding box for the region to be extracted.
    :return: (np.ndarray) the extracted region from input_img
    """

    # sanity checks
    if not isinstance(input_img, np.ndarray):
        raise TypeError("Input image should be np.ndarray but is: " + str(type(input_img)))
    if not isinstance(region_bbox, list):
        try:
            bbox = list(region_bbox)
        except:
            raise TypeError("Provided bounding box cannot be converted to list.")
    else:
        bbox = region_bbox

    # determine zero pads if needed
    dims = input_img.shape
    ddims = np.repeat(dims, 2)
    pads = np.zeros(len(bbox), dtype=np.int)
    for i, ind in enumerate(bbox):
        if ind < 0:
            pads[i] = abs(ind)
            bbox[i] = 0
        elif ind > ddims[i]:
            pads[i] = 1 + ind - ddims[i]  # add one here to correct for indexing vs dim size
            bbox[i] = ddims[i] - 1  # the last possible ind is shape of that dim - 1
        else:
            pads[i] = 0

    # extract region
    region = input_img[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

    # pad to desired size
    if any([pad > 0 for pad in pads]):
        pad_tuple = ((pads[0], pads[1]), (pads[2], pads[3]), (pads[4], pads[5]))
        output = np.pad(region, pad_tuple, 'constant')
    else:
        output = region

    return output


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


def _load_roi_multicon_and_labels(study_dir, feature_prefx, label_prefx, mask_prefx, patch_size, plane='ax',
                                  data_fmt='channels_last', aug=True, interp=1):
    """
    Patch loader generates 2D patch data for images and labels given a list of 3D input NiFTI images a mask.
    Performs optional data augmentation with affine rotation in 3D.
    Data is cropped to the nonzero bounding box for the mask file before patches are generated.
    :param study_dir: (str) The path to the study directory to get data from.
    :param feature_prefx: (iterable of str) The prefixes for the image files containing the data (features).
    :param label_prefx: (str) The prefixe for the image files containing the labels.
    :param mask_prefx: (str) The prefixe for the image files containing the data mask.
    :param patch_size: (list or tuple of ints) The patch size in pixels (must be shape 2 for 2d)
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param data_fmt (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param aug: (bool) Whether or not to perform data augmentation with random 3D affine rotation.
    :param interp: (int) The order of spline interpolation for label data. Must be 0-5
    :return: (tf.tensor) The patch data for features and labels as a tensorflow variable.
    """

    # sanity checks
    if not len(patch_size) == 2:
        raise ValueError("Patch size must be shape 2 for 2d data loader but is: " + str(patch_size))
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
    if aug:  # if augmenting, select random rotation values for x, y, and z axes
        theta = 0.  # np.random.random() * (np.pi / 2.)  # rotation in yz plane
        phi = 0.  # np.random.random() * (np.pi / 2.)  # rotation in xz plane
        psi = np.random.random() * (np.pi / 2.)  # rotation in xy plane
    else:  # if not augmenting, no rotation is applied, and affine is used only for offset to center the ROI
        theta = 0.
        phi = 0.
        psi = 0.

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
    mask_bbox = _nonzero_slice_inds3d(mask)
    dim_sizes = [mask_bbox[1] - mask_bbox[0], mask_bbox[3] - mask_bbox[2], mask_bbox[5] - mask_bbox[4]]

    # find the closest multiple of patch_size that encompasses the mask rounding up and get new inds centered on mask
    add = [patchsize - (dimsize % patchsize) if dimsize % patchsize > 0 else 0 for patchsize, dimsize in zip(patch_size, dim_sizes)]
    new_bbox = [mask_bbox[0] - np.floor(add[0] / 2.), mask_bbox[1] + np.ceil(add[0] / 2.),
                mask_bbox[2] - np.floor(add[1] / 2.), mask_bbox[3] + np.ceil(add[1] / 2.),
                mask_bbox[4], mask_bbox[5]]  # no adjustment needed in z for the 2d case
    new_bbox = [int(item) for item in new_bbox]
    new_dim_sizes = [new_bbox[1] - new_bbox[0], new_bbox[3] - new_bbox[2], new_bbox[5] - new_bbox[4]]

    # extract the region with zero padding to new bbox if needed
    data_region = np.zeros(new_dim_sizes + [len(data_files)])
    for i in range(len(data_files)):
        data_region[:, :, :, i] = _extract_region(data[:, :, :, i], tuple(new_bbox))
    data = data_region
    labels = _extract_region(labels, new_bbox)

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
        #data = np.transpose(data, axes=(0, 1, 2, 3))
        labels = np.expand_dims(labels, axis=3)
        #labels = np.transpose(labels, axes=(0, 1, 2, 3))
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_fmt == 'channels_first':
        data = np.transpose(data, axes=(0, 3, 1, 2))
        labels = np.transpose(labels, axes=(0, 3, 1, 2))

    return data.astype(np.float32), labels.astype(np.float32)


def _load_roi_multicon_and_labels_3d(study_dir, feature_prefx, label_prefx, mask_prefx, patch_size, plane='ax',
                                  data_fmt='channels_last', aug=True, interp=1):
    """
    Patch loader generates 3D patch data for images and labels given a list of 3D input NiFTI images a mask.
    Performs optional data augmentation with affine rotation in 3D.
    Data is cropped to the nonzero bounding box for the mask file before patches are generated.
    :param study_dir: (str) The path to the study directory to get data from.
    :param feature_prefx: (iterable of str) The prefixes for the image files containing the data (features).
    :param label_prefx: (str) The prefixe for the image files containing the labels.
    :param mask_prefx: (str) The prefixe for the image files containing the data mask.
    :param patch_size: (list or tuple of ints) The patch size in pixels (must be shape 3 for 3d)
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param data_fmt (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param aug: (bool) Whether or not to perform data augmentation with random 3D affine rotation.
    :param interp: (int) The order of spline interpolation for label data. Must be 0-5
    :return: (tf.tensor) The patch data for features and labels as a tensorflow variable.
    """

    # sanity checks
    if not len(patch_size) == 3:
        raise ValueError("Patch size must be shape 3 for 3d data loader but is: " + str(patch_size))
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
    if aug:  # if augmenting, select random rotation values for x, y, and z axes
        theta = 0.  # np.random.random() * (np.pi / 2.)  # rotation in yz plane
        phi = 0.  # np.random.random() * (np.pi / 2.)  # rotation in xz plane
        psi = np.random.random() * (np.pi / 2.)  # rotation in xy plane
    else:  # if not augmenting, no rotation is applied, and affine is used only for offset to center the ROI
        theta = 0.
        phi = 0.
        psi = 0.

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
    mask_bbox = _nonzero_slice_inds3d(mask)
    dim_sizes = [mask_bbox[1] - mask_bbox[0], mask_bbox[3] - mask_bbox[2], mask_bbox[5] - mask_bbox[4]]

    # find the closest multiple of patch_size that encompasses the mask rounding up and get new inds centered on mask
    delta = [patchsize - (dimsize % patchsize) if dimsize % patchsize > 0 else 0 for patchsize, dimsize in zip(patch_size, dim_sizes)]
    new_bbox = [mask_bbox[0] - np.floor(delta[0] / 2.), mask_bbox[1] + np.ceil(delta[0] / 2.),
                mask_bbox[2] - np.floor(delta[1] / 2.), mask_bbox[3] + np.ceil(delta[1] / 2.),
                mask_bbox[4] - np.floor(delta[2] / 2.), mask_bbox[5] + np.ceil(delta[2] / 2.)]  # z adjustment for 3d case
    new_bbox = [int(item) for item in new_bbox]
    new_dim_sizes = [new_bbox[1] - new_bbox[0], new_bbox[3] - new_bbox[2], new_bbox[5] - new_bbox[4]]

    # extract the region with zero padding to new bbox if needed
    data_region = np.zeros(new_dim_sizes + [len(data_files)])
    for i in range(len(data_files)):
        data_region[:, :, :, i] = _extract_region(data[:, :, :, i], tuple(new_bbox))
    data = data_region
    labels = _extract_region(labels, new_bbox)

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


##############################################
# TENSORFLOW MAP FUNCTIONS
##############################################


def _tf_patches(data, labels, patch_size, chan_dim, data_format):

    # sanity checks
    if not len(patch_size) == 2:
        raise ValueError("Patch size must be shape 2 to use 2D patch function but is: " + str(patch_size))

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(data, perm=[0, 2, 3, 1])
        labels = tf.transpose(labels, perm=[0, 2, 3, 1])

    # make patches
    ksizes = [1] + patch_size + [1]
    strides = [1] + patch_size + [1]
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


def _tf_patches_3d(data, labels, patch_size, chan_dim, data_format):

    # sanity checks
    if not len(patch_size) == 3:
        raise ValueError("Patch size must be shape 3 to use 3D patch function but is: " + str(patch_size))

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(data, perm=[0, 2, 3, 4, 1])
        labels = tf.transpose(labels, perm=[0, 2, 3, 4, 1])

    # make patches
    ksizes = [1] + patch_size + [1]
    strides = [1] + patch_size + [1]
    rates = [1, 1, 1, 1, 1]
    data = tf.extract_image_patches(data, ksizes=ksizes, strides=strides, rates=rates, padding='SAME')
    data = tf.reshape(data, [-1] + patch_size + [chan_dim])
    labels = tf.extract_image_patches(labels, ksizes=ksizes, strides=strides, rates=rates, padding='SAME')
    labels = tf.reshape(labels, [-1] + patch_size + [1])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(data, perm=[0, 4, 1, 2, 3])
        labels = tf.transpose(labels, perm=[0, 4, 1, 2, 3])

    return data, labels


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
        # map tensorflow dataset variable to data
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        dataset = dataset.prefetch(buffer_size=10)
        dataset = dataset.map(
            lambda x: tf.py_func(_load_roi_multicon_and_labels,
                                 [x] + py_func_params,
                                 (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)
        dataset = dataset.prefetch(buffer_size=10)
        dataset = dataset.map(
            lambda x, y: _tf_patches(x, y, params.train_dims, len(params.data_prefix), params.data_format),
            num_parallel_calls=params.num_threads)
        dataset = dataset.prefetch(buffer_size=10)
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
        dataset = dataset.prefetch(buffer_size=params.shuffle_size)
        dataset = dataset.shuffle(buffer_size=params.shuffle_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
        dataset = dataset.prefetch(buffer_size=params.buffer_size)

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
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
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
    get_next_features.set_shape([params.batch_size, data_dims[0], data_dims[1], len(params.data_prefix)])
    get_next_labels = get_next['labels']
    get_next_labels.set_shape([params.batch_size, data_dims[0], data_dims[1], len(params.label_prefix)])

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

    # generate tensorflow dataset object
    dataset = tf.data.Dataset.from_tensor_slices([infer_dir])
    dataset = dataset.map(
        lambda x: tf.py_func(_load_multicon_preserve_size,
                             [x] + py_func_params,
                             tf.float32), num_parallel_calls=params.num_threads)
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    dataset = dataset.prefetch(params.buffer_size)
    padded_shapes = ([data_dims[0], data_dims[1], len(params.data_prefix)])
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
                          params.train_dims,
                          params.data_plane,
                          params.data_format,
                          params.augment_train_data,
                          params.label_interp]
        # map tensorflow dataset variable to data
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # dataset = dataset.prefetch(buffer_size=10)
        dataset = dataset.map(
            lambda x: tf.py_func(_load_roi_multicon_and_labels_3d,
                                 [x] + py_func_params,
                                 (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)
        dataset = dataset.prefetch(buffer_size=10) # prefetch 10 whole datasets prior to patching
        dataset = dataset.map(
            lambda x, y: _tf_patches_3d(x, y, params.train_dims, len(params.data_prefix), params.data_format),
            num_parallel_calls=params.num_threads)
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
        dataset = dataset.prefetch(buffer_size=params.shuffle_size)  # prefetch shuffle_size examples before shuffling
        dataset = dataset.shuffle(buffer_size=params.shuffle_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
        dataset = dataset.prefetch(buffer_size=params.buffer_size)

    raise NotImplemented("Not implemented from this point on")

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
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
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
    get_next_features.set_shape([params.batch_size, data_dims[0], data_dims[1], len(params.data_prefix)])
    get_next_labels = get_next['labels']
    get_next_labels.set_shape([params.batch_size, data_dims[0], data_dims[1], len(params.label_prefix)])

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
    raise NotImplemented("This is not yet implemented")

    # force batch size of 1 for inference
    batch_size = 1  # params.batch_size

    # prepare pyfunc
    data_dims = list(params.infer_dims)
    py_func_params = [params.data_prefix, params.data_format, data_dims, params.data_plane]

    # generate tensorflow dataset object
    dataset = tf.data.Dataset.from_tensor_slices([infer_dir])
    dataset = dataset.map(
        lambda x: tf.py_func(_load_multicon_preserve_size,
                             [x] + py_func_params,
                             tf.float32), num_parallel_calls=params.num_threads)
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    dataset = dataset.prefetch(params.buffer_size)
    padded_shapes = ([data_dims[0], data_dims[1], len(params.data_prefix)])
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