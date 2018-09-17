import os
import math
from glob import glob
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import scipy.ndimage.interpolation as interp
import tensorflow as tf
from random import shuffle


def patch_loader(study_dir, file_prefixes, label_prefix, mask_prefix, data_format, patch_size, augment=True):

    # define full paths
    data_files = [glob(study_dir + '/*' + contrast + '*.nii.gz')[0] for contrast in file_prefixes]
    labels_file = glob(study_dir + '/*' + label_prefix[0] + '*.nii.gz')[0]
    mask_file = glob(study_dir + '/*' + mask_prefix[0] + '*.nii.gz')[0]
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

    # do optional augmentation with 3D rotation?
    if augment:
        theta = np.random.random() * (np.pi/2.)
        mask = _affine_transform(mask, theta, phi=0., psi=0., offset=None, order=0)  # nn interp for mask
        data = _affine_transform(data, theta, phi=0., psi=0., offset=None, order=1)
        labels = _affine_transform(labels, theta, phi=0., psi=0., offset=None, order=1)

    # get the tight bounding box of the mask
    mask_bbox = _nonzero_slice_inds3d(mask)
    dim_sizes = [mask_bbox[1] - mask_bbox[0], mask_bbox[3] - mask_bbox[2], mask_bbox[5] - mask_bbox[4]]

    # find the closest multiple of patch_size that encompasses the mask rounding up and get new inds centered on mask
    add = [patch_size - (dimsize % patch_size) if dimsize % patch_size else 0 for dimsize in dim_sizes]
    new_bbox = [mask_bbox[0] - np.floor(float(add[0])), mask_bbox[1] + np.ceil(float(add[0])),
                mask_bbox[2] - np.floor(float(add[1])), mask_bbox[3] + np.ceil(float(add[1])),
                mask_bbox[4] - np.floor(float(add[2])), mask_bbox[5] + np.ceil(float(add[2]))]

    # determine any zero padding that needs to happen

    # make a function that crops and returns padded value if needed
    # if negative, keep it, if positive, subtract value from dims
    pads = [val if val < 0 else dim - val for val, dim in new_bbox, np.repeat(data_dims, 2)]
    # if negative, return absolute value as the pad vale, else return 0
    pads = [abs(val) if val < 0 else 0 for val in pads]

    # crop the data down to the new inds that are a multiple of patch size

    # divide the data into patch_size squares and stack them

    # convert to desired data format and return



def _affine_transform(input_img, theta=None, phi=None, psi=None, offset=None, order=1):

    # define angles
    if not theta:
        theta = np.random.random() * (np.pi/2.)
    if not phi:
        phi = np.random.random() * (np.pi/2.)
    if not psi:
        psi = np.random.random() * (np.pi/2.)

    # define offset
    if not offset:
        offset = [i / 2. for i in input_img.shape]

    affine = [
        [np.cos(theta) * np.cos(psi),
         -np.cos(phi) * np.sin(psi) + np.sin(phi) * np.sin(theta) * np.cos(psi),
         np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi)],

        [np.cos(theta) * np.sin(psi),
         np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi),
         -np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi)],

        [-np.sin(theta),
         np.sin(phi) * np.cos(theta),
         np.cos(phi) * np.cos(theta)]
    ]

    output_img = ndi.affine_transform(input_img, affine, offset, input_img.shape, order)

    return output_img


def _load_single_study(study_dir, file_prefixes, data_format, slice_trim=None, norm=False):
    """
    Image data I/O function for use in tensorflow Dataset map function. Takes a study directory and file prefixes and
    returns a 4D numpy array containing the image data. Performs optional slice trimming in z and normalization.
    :param study_dir: (str) the full path to the study directory
    :param file_prefixes: (str, list(str)) the file prefixes for the images to be loaded
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param slice_trim: (list, tuple) contains 2 ints, the first and last slice to use for trimming. None = auto trim.
                        [0, -1] does no trimming
    :param norm: (bool) whether or not to perform per dataset normalization
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
    nz_inds = [0, -1]
    for ind, image in enumerate(images):
        if ind == 0:  # find dimensions after trimming zero slices and preallocate 4d array
            first_image = nib.load(images[0]).get_fdata()
            if slice_trim:
                nz_inds = slice_trim
            else:
                nz_inds = _nonzero_slice_inds(first_image)
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

    # permute data to desired data format
    if data_format == 'channels_first':
        output = np.transpose(output, axes=(2, 3, 0, 1))
    else:
        output = np.transpose(output, axes=(2, 0, 1, 3))

    return output, nz_inds


def _load_single_study_mask(study_dir, file_prefixes, mask, data_format, slice_trim=None, norm=False):
    """
    Image data I/O function for use in tensorflow Dataset map function. Takes a study directory and file prefixes and
    returns a 4D numpy array containing the image data. Performs optional slice trimming in z based on mask and
     optional input image normalization.
    :param study_dir: (str) the full path to the study directory
    :param file_prefixes: (str, list(str)) the file prefixes for the images to be loaded
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param mask (np.ndarray) the mask data for masking inputs
    :param slice_trim: (list, tuple) contains 2 ints, the first and last slice to use for trimming. None = auto trim.
                        [0, -1] does no trimming
    :param norm: (bool) whether or not to perform per dataset normalization
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
    nz_inds = [0, -1]
    for ind, image in enumerate(images):
        if ind == 0:  # find dimensions after trimming zero slices and preallocate 4d array
            first_image = nib.load(images[0]).get_fdata()
            if slice_trim:
                nz_inds = slice_trim
            else:
                nz_inds = _nonzero_slice_inds(first_image)
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

    # permute data to desired data format
    if data_format == 'channels_first':
        output = np.transpose(output, axes=(2, 3, 0, 1))
        # do masking
        for i in range(output.shape[1]):
            output[:, i, :, :] = np.where(np.squeeze(mask)>0, output[:, i, :, :], 0)
    else:
        output = np.transpose(output, axes=(2, 0, 1, 3))
        # do masking
        for i in range(output.shape[3]):
            output[:, :, :, i] = np.where(np.squeeze(mask)>0, output[:, :, :, i], 0)

    return output, nz_inds


def _load_single_study_crop_mask(study_dir, file_prefixes, mask_prefix, dim_out, data_format, norm=False):
    """
    Image data I/O function for use in tensorflow Dataset map function. Takes a study directory and file prefixes and
    returns a 4D numpy array containing the image data cropped to a mask. Performs optional normalization.
    :param study_dir: (str) the full path to the study directory
    :param file_prefixes: (str, list(str)) the file prefixes for the images to be loaded
    :param mask_prefix: (np.ndarray) the mask data for masking inputs
    :param dim_out: (int) the desired output dimensions for the data (currently isotropic)
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param norm: (bool) whether or not to perform per dataset normalization
    :return: output - a 4D numpy array containing the image data
    """

    # sanity checks
    if not os.path.isdir(study_dir): raise ValueError("Specified study_dir does not exist")
    if data_format not in ['channels_last', 'channels_first']: raise ValueError("data_format invalid")
    images = [glob(study_dir + '/*' + contrast + '*.nii.gz')[0] for contrast in file_prefixes]
    if not images: raise ValueError("No matching image files found for file prefixes: " + str(images))
    if isinstance(dim_out, np.ndarray): dim_out = dim_out[0]

    # load mask data
    mask_file = glob(study_dir + '/*' + mask_prefix[0] + '*.nii.gz')[0]
    mask = nib.load(mask_file).get_fdata()
    inds = _nonzero_slice_inds3d(mask)

    # scale and pad mask to desired size
    mask = mask[inds[0]:inds[1], inds[2]:inds[3], inds[4]:inds[5]]
    maxdim = np.max(mask.shape)
    zoom = np.float(dim_out) / np.float(maxdim)
    mask = ndi.zoom(mask, (zoom, zoom, zoom), order=0)  # nn interp mask

    # preallocate 4d array, load images and crop/normalize then insert into 4d array
    output = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], len(file_prefixes)))
    for ind, image in enumerate(images):
        img = nib.load(image).get_fdata()[inds[0]:inds[1], inds[2]:inds[3], inds[4]:inds[5]]
        # do normalization
        if norm:
            img = _normalize(img)
        # do zoom
        img = ndi.zoom(img, (zoom, zoom, zoom), order=1)  # linear interp data
        output[:, :, :, ind] = img

    # do masking
    # for i in range(output.shape[3]):
    #     output[:, :, :, i] = np.where(mask > 0, output[:, :, :, i], 0)

    # zero pad data to final dims
    output = _zero_pad_image(output, out_dims=[dim_out, dim_out], axes=[0, 1])  # only zeropad in x and y

    # permute data to desired data format
    if data_format == 'channels_first':
        output = np.transpose(output, axes=(2, 3, 0, 1))
    else:
        output = np.transpose(output, axes=(2, 0, 1, 3))

    return output, inds


def _normalize(input_numpy):
    """
    Performs image normalization to zero mean, unit variance.
    :param input_numpy: The input numpy array
    :return: The input array normalized to zero mean, unit variance or [0, 1]
    """

    # perform normalization to zero mean unit variance
    # input_numpy = np.where(input_numpy!=0, input_numpy - input_numpy.mean(), 0) / input_numpy.var()

    # perform normalization to [0, 1]
    input_numpy *= 1.0 / input_numpy.max()

    return input_numpy


def _nonzero_slice_inds(input_numpy):
    """
    Takes numpy array and returns slice indices of first and last nonzero slices
    :param input_numpy: (np.ndarray) a numpy array containing image data
    :return: inds - a list of 2 indices corresponding to the first and last nonzero slices in the numpy array
    """

    # sanity checks
    if type(input_numpy) is not np.ndarray: raise ValueError("Input must be numpy array")

    # finds inds of first and last nonzero slices
    vector = np.max(np.max(input_numpy, axis=0), axis=0)
    nz = np.nonzero(vector)[0]
    inds = [nz[0], nz[-1]]

    return inds


def _nonzero_slice_inds3d(input_numpy):
    """
    Takes numpy array and returns slice indices of first and last nonzero pixels in 3d
    :param input_numpy: (np.ndarray) a numpy array containing image data
    :return: inds - a list of 2 indices corresponding to the first and last nonzero slices in the numpy array
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


def _augment_image(input_data, data_format, params=(np.random.random()*90., np.random.random()>0.5), order=1):
    """
    Takes input numpy array and a data format and performs data augmentation with random rotations and flips
    :param input_data: (np.ndarray) a 4D numpy array containing image data in the specified TF data format
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param params: (list, tuple) A list or tuple of user specified values in format [rotation=float(degrees), flip=bool]
    :param order: (int) an int from 0-5, the order for spline interpolation, default is 1 (linear)
    :return: output_data - a numpy array or tuple of numpy arrays containing the augmented data
    """

    # sanity checks
    if type(input_data) is not np.ndarray: raise ValueError("Input must be numpy array or list/tuple of numpy arrays")
    if data_format not in ['channels_last', 'channels_first']: raise ValueError("data_format invalid")
    if not isinstance(params, (list, tuple)): raise ValueError("Params must be a list or tuple if specified")
    if not isinstance(params[0], (float, int)): raise ValueError("First entry in params must be a float or int")
    if not isinstance(params[1], bool): raise ValueError("Second entry in params must be a boolean")
    if order not in range(6): raise ValueError("Spline interpolation order must be in range 0-3")

    # apply rotation
    if data_format == 'channels_last':
        axes = (1, 2)
    else:
        axes = (2, 3)
    output_data = interp.rotate(input_data, float(params[0]), axes=axes, reshape=False, order=order)

    # apply flip
    if params[1]:
        output_data = np.flip(output_data, axis=axes[0])

    return output_data


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


def _mask_and_crop(data, mask, data_format):
    """
    Takes data and a mask and returns the masked data cropped to tightly to the mask borders
    :param data: (np.ndarray) the image data
    :param mask: (np.ndarray) the mask data
    :param data_format: (str) the data format, either 'channels_first' or 'channels_last'
    :return: The masked and cropped data array
    """

    # this currently doesnt crop 4d images, just 3d images
    # this needs to be corrected before use
    if data:
        raise NotImplementedError("Function not yet implemented.")

    # mask data
    data = data * (mask>1.).astype(np.float)

    # get 3d mask
    mask = np.squeeze(mask[:, :, :])

    # handle data format
    if data_format == 'channels_first':
        mask = np.squeeze(mask[:, 0, :, :])
    else:
        mask = np.squeeze(mask[:, :, :, 0])

    # get nonzero inds of mask image in 3D
    xvector = np.max(np.max(mask, axis=0), axis=1)
    nzx = np.nonzero(xvector)[0]
    nzix = [nzx[0], nzx[-1]]
    yvector = np.max(np.max(mask, axis=0), axis=0)
    nzy = np.nonzero(yvector)[0]
    nziy = [nzy[0], nzy[-1]]

    # crop data
    if data_format == 'channels_first':
        data = data[:, :, nzix[0]:nzix[1], nziy[0]:nziy[1]]
    else:
        data = data[:, nzix[0]:nzix[1], nziy[0]:nziy[1], :]

    return data


def _load_multicon_and_labels(study_dir, feature_prefx, label_prefx, data_fmt, out_dims, augment, label_interp=1):
    """
    Load multicontrast image data and target data/labels and perform augmentation if desired.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param label_prefx: (list) a list containing one string, the labels to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param out_dims: (list(int)) the desired output data dimensions, data will be zero padded to dims
    :param augment: (str) whether or not to perform data augmentation, must be 'yes' or 'no'
    :param label_interp: (int) in range 0-5, the order of spline interp for labels during augmentation
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_dir): raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]): raise ValueError("Data prefixes must be strings")
    if not all([isinstance(a, str) for a in label_prefx]): raise ValueError("Labels prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']: raise ValueError("data_format invalid")
    if not all([np.issubdtype(a, np.integer) for a in out_dims]): raise ValueError(
        "data_dims must be a list/tuple of ints")
    if augment not in ['yes', 'no']: raise ValueError("Parameter augment must be 'yes' or 'no'")
    if label_interp not in range(6): raise ValueError("Spline interpolation order must be in range 0-3")

    # load multi-contrast data
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, norm=True)  # normalize input imgs

    # load labels data
    labels, nzi = _load_single_study(study_dir, label_prefx, data_format=data_fmt, slice_trim=nzi)

    # if augmentation is to be used generate random params for augmentation and run augment function
    if augment == 'yes':
        params = (np.random.random() * 90., np.random.random() > 0.5)
        data = _augment_image(data, params=params, data_format=data_fmt, order=1)  # force linear interp for images
        labels = _augment_image(labels, params=params, data_format=data_fmt, order=label_interp)

    # do data padding to desired dims
    axes = [1, 2] if data_fmt == 'channels_last' else [2, 3]
    data = _zero_pad_image(data, out_dims, axes)
    labels = _zero_pad_image(labels, out_dims, axes)

    return data, labels


def _load_multicon_and_labels_mask(study_dir, feature_prefx, label_prefx, mask_prefx, data_fmt, out_dims, augment,
                                   label_interp=1):
    """
    Load multicontrast image data and target data/labels with optional masking and perform augmentation if desired.
    Currently this only crops the data to the nonzero part of the mask in the z dimension
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of data filename prefixes - the data files to be loaded
    :param label_prefx: (list) a list containing one string, the prefix of the labels to be loaded
    :param mask_prefx: (list) a list containing one string: the prefix of the mask nifti file
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param out_dims: (list(int)) the desired output data dimensions, data will be zero padded to dims
    :param augment: (str) whether or not to perform data augmentation, must be 'yes' or 'no'
    :param label_interp: (int) in range 0-5, the order of spline interp for labels during augmentation
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_dir): raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]): raise ValueError("Data prefixes must be strings")
    if not all([isinstance(a, str) for a in label_prefx]): raise ValueError("Labels prefixes must be strings")
    if not all([isinstance(a, str) for a in mask_prefx]): raise ValueError("Mask prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']: raise ValueError("data_format invalid")
    if not all([np.issubdtype(a, np.integer) for a in out_dims]): raise ValueError(
        "data_dims must be a list/tuple of ints")
    if augment not in ['yes', 'no']: raise ValueError("Parameter augment must be 'yes' or 'no'")
    if label_interp not in range(6): raise ValueError("Spline interpolation order must be in range 0-3")

    # check if mask
    mask_file = glob(study_dir + '/*' + mask_prefx[0] + '*.nii.gz')[0]
    if os.path.isfile(mask_file):
        # load multi-contrast data cropping to mask and normalizing
        data, mask_nzi = _load_single_study_crop_mask(study_dir, feature_prefx, mask_prefx, out_dims, data_fmt, True)
        # load labels data
        labels, _ = _load_single_study_crop_mask(study_dir, label_prefx, mask_prefx, out_dims, data_fmt, False)
    else:
        print("Mask file does not exist! Loading data without mask")
        # load labels data
        labels, labels_nzi = _load_single_study(study_dir, label_prefx, data_format=data_fmt)
        # load multi-contrast data
        data, _ = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, slice_trim=labels_nzi, norm=True)
        data = _normalize(data)

    # if augmentation is to be used generate random params for augmentation and run augment function
    if augment == 'yes':
        params = (np.random.random() * 90., np.random.random() > 0.5)
        data = _augment_image(data, params=params, data_format=data_fmt, order=1)  # force linear interp for images
        labels = _augment_image(labels, params=params, data_format=data_fmt, order=label_interp)

    return data.astype(np.float32), labels.astype(np.float32)  # why are these not floats already?


def _load_multicon_preserve_size(study_dir, feature_prefx, data_fmt, out_dims):
    """
    Load multicontrast image data without cropping or otherwise adjusting size. For use with inference/prediction.
    :param study_dir: (str) A directory containing the desired image data.
    :param feature_prefx: (list) a list of filenames - the data files to be loaded
    :param data_fmt: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param out_dims: (list(int)) the desired output data dimensions, data will be zero padded to dims
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_dir): raise ValueError("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]): raise ValueError("Data prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']: raise ValueError("data_format invalid")
    if not all([np.issubdtype(a, np.integer) for a in out_dims]): raise ValueError(
        "data_dims must be a list/tuple of ints")

    # load multi-contrast data and normalize, no slice trimming for infer data
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt, slice_trim=[0, -1], norm=True)

    return data


def display_tf_dataset(dataset_data, data_format):
    """
    Displays tensorflow dataset output images and labels/regression images.
    :param dataset_data: (tf.tensor) output from tf dataset function containing images and labels/regression image
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :return: displays images for 3 seconds then continues
    """

    # make figure
    fig = plt.figure(figsize=(10, 4))

    # define close event and create timer
    def close_event():
        plt.close()
    timer = fig.canvas.new_timer(interval=4000)
    timer.add_callback(close_event)

    # image data
    image_data = dataset_data["features"]  # dataset_data[0]
    if len(image_data.shape) > 3:
        image_data = np.squeeze(image_data[0, :, :, :])  # handle batch data
    nplots = image_data.shape[0] + 1 if data_format == 'channels_first' else image_data.shape[2] + 1
    channels = image_data.shape[0] if data_format == 'channels_first' else image_data.shape[2]
    for z in range(channels):
        ax = fig.add_subplot(1, nplots, z + 1)
        img = np.swapaxes(np.squeeze(image_data[z, :, :]), 0, 1) if data_format == 'channels_first' else np.squeeze(
            image_data[:, :, z])
        ax.imshow(img, cmap='gray')
        ax.set_title('Data Image ' + str(z + 1))

    # label data
    label_data = dataset_data["labels"]  # dataset_data[1]
    if len(label_data.shape) > 3: label_data = np.squeeze(label_data[0, :, :, :])  # handle batch data
    ax = fig.add_subplot(1, nplots, nplots)
    img = np.swapaxes(np.squeeze(label_data), 0, 1) if data_format == 'channels_first' else np.squeeze(label_data)
    ax.imshow(img, cmap='gray')
    ax.set_title('Labels')

    # start timer and show plot
    timer.start()
    plt.show()

    return


def input_fn(mode, params):
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

    # differences between training and non-training: augment, dirs
    data_dims = list(params.train_dims)
    if mode == 'train':
        data_dirs = train_dirs
        py_func_params = [params.data_prefix, params.label_prefix, params.mask_prefix, params.data_format, data_dims,
                          params.augment_train_data,
                          params.label_interp]

    elif mode == 'eval':
        data_dirs = eval_dirs
        py_func_params = [params.data_prefix, params.label_prefix, params.mask_prefix, params.data_format, data_dims,
                          'no']  # 'no' for aug
    elif mode == 'infer':
        raise ValueError("Please use separate infer data input function.")
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    # generate tensorflow dataset object
    dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
    dataset = dataset.map(
        lambda x: tf.py_func(_load_multicon_and_labels_mask,
                             [x] + py_func_params,
                             (tf.float32, tf.float32)), num_parallel_calls=params.num_threads)
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
    dataset = dataset.prefetch(params.buffer_size)
    if mode == 'train':
        dataset = dataset.shuffle(params.shuffle_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
    elif mode == 'eval':
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
    elif mode == 'infer':
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    # make iterator and query the output of the iterator for input to the model
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    init_op = iterator.initializer

    # manually set shapes for inputs
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

    # prepare pyfunc
    data_dims = list(params.infer_dims)
    py_func_params = [params.data_prefix, params.data_format, data_dims]  # 'no' for aug

    # generate tensorflow dataset object
    dataset = tf.data.Dataset.from_tensor_slices([infer_dir])
    dataset = dataset.map(
        lambda x: tf.py_func(_load_multicon_preserve_size,
                             [x] + py_func_params,
                             tf.float32), num_parallel_calls=params.num_threads)
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    dataset = dataset.prefetch(params.buffer_size)
    padded_shapes = ([data_dims[0], data_dims[1], len(params.data_prefix)])
    dataset = dataset.padded_batch(params.batch_size, padded_shapes=padded_shapes, padding_values=0.)

    # make iterator and query the output of the iterator for input to the model
    iterator = dataset.make_initializable_iterator()
    get_next_features = iterator.get_next()
    init_op = iterator.initializer

    # manually set shapes for inputs
    get_next_features.set_shape([params.batch_size, data_dims[0], data_dims[1], len(params.data_prefix)])

    # Build and return a dictionary containing the nodes / ops
    inputs = {'features': get_next_features, 'iterator_init_op': init_op}

    return inputs
