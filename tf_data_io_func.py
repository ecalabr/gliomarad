import os
import sys
from glob import glob
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import scipy.ndimage.interpolation as interp


def _load_single_study(study_dir, file_prefixes, data_format='channels_last', slice_trim=None):
    """
    Image data I/O function for use in tensorflow Dataset map function. Takes a study directory and file prefixes and
    returns a 4D numpy array containing the image data.
    :param study_dir: A string - the full path to the study directory
    :param file_prefixes: A string or list of strings - the file prefixes for the images to be loaded
    :param data_format: the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param slice_trim: A list/tuple containing 2 ints, the first and last slice to use for trimming. None = auto trim
    :return:
        output - a 4D numpy array containing the image data
    """

    # sanity checks
    if not os.path.isdir(study_dir): sys.exit("Specified study_dir does not exist")
    if data_format not in ['channels_last', 'channels_first']: sys.exit("data_format invalid")
    if slice_trim is not None and not isinstance(slice_trim, (list, tuple)): sys.exit("slice_trim must be list/tuple")
    images = [glob(study_dir + '/*' + contrast + '*.nii.gz')[0] for contrast in file_prefixes]
    if not images: sys.exit("No matching image files found for file prefixes: " + str(images))

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
            output_shape = list(first_image.shape)[0:3] + [len(images)]
            output = np.zeros(output_shape, np.float32)
            output[:, :, :, 0] = first_image
        else:
            output[:, :, :, ind] = nib.load(images[ind]).get_fdata()[:, :, nz_inds[0]:nz_inds[1]]

    # permute data to desired data format
    if data_format == 'channels_last':
        output = np.transpose(output, axes=(2, 3, 0, 1))
    else:
        output = np.transpose(output, axes=(2, 0, 1, 3))

    return output, nz_inds


def _nonzero_slice_inds(input_numpy):
    """
    Takes numpy array and returns slice indices of first and last nonzero slices
    :param input_numpy: a numpy array containing image data
    :return:
        inds - a list of 2 indices corresponding to the first and last nonzero slices in the numpy array
    """

    # sanity checks
    if type(input_numpy) is not np.ndarray: sys.exit("Input must be numpy array")

    # finds inds of first and last nonzero slices
    vector = np.max(np.max(input_numpy, axis=0), axis=0)
    nz = np.nonzero(vector)[0]
    inds = [nz[0], nz[-1]]

    return inds


def _augment_image(input_data, data_format, params=(np.random.random()*90., np.random.random()>0.5), order=1):
    """
    Takes input numpy array and a data format and performs data augmentation with random rotations and flips
    :param input_data: a 4D numpy array containing image data in the specified TF data format
    :param data_format: the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param params: A list or tuple of user specified values in format [rotation=float(degrees), flip=bool]
    :param order: an int from 0-5, the order for spline interpolation, default is 1 (linear)
    :return:
        output_data - a numpy array or tuple of numpy arrays containing the augmented data
    """

    # sanity checks
    if type(input_data) is not np.ndarray: sys.exit("Input must be numpy array or list/tuple of numpy arrays")
    if data_format not in ['channels_last', 'channels_first']: sys.exit("data_format invalid")
    if not isinstance(params, (list, tuple)): sys.exit("Params must be a list or tuple if specified")
    if not isinstance(params[0], (float, int)): sys.exit("First entry in params must be a float or int")
    if not isinstance(params[1], bool): sys.exit("Second entry in params must be a boolean")
    if not order in range(6): sys.exit("Spline interpolation order must be in range 0-3")

    # apply rotation
    if data_format == 'channels_last':
        axes = (2, 3)
    else:
        axes = (1, 2)
    output_data = interp.rotate(input_data, float(params[0]), axes=axes, reshape=False, order=order)

    # apply flip
    if params[1]:
        output_data = np.flip(output_data, axis=axes[0])

    return output_data


def load_multicon_and_labels(study_directory, data_prefixes, labels_prefix):
    """
    Load multicontrast image data and a label image.
    :param study_directory: A directory containing the desired image data.
    :param data_prefixes: a list of filenames - the data files to be loaded
    :param labels_prefix: a list containing one string, the labels to be loaded
    :return: a tuple of np ndarrays containing the image data and labels in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_directory): sys.exit("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in data_prefixes]): sys.exit("Data prefixes must be strings")
    if not all([isinstance(a, str) for a in labels_prefix]): sys.exit("Labels prefixes must be strings")

    # load multicontrast data
    data, nzi = _load_single_study(study_directory, data_prefixes, data_format='channels_last')

    # load labels data
    labels, nzi = _load_single_study(study_directory, labels_prefix, data_format='channels_last', slice_trim=nzi)

    # augment
    params = (np.random.random() * 90., np.random.random() > 0.5)
    data = _augment_image(data, params=params, data_format='channels_last', order=1)
    labels = _augment_image(labels, params=params, data_format='channels_last', order=0)  # NN interp for labels

    return data, labels


def load_multicon_and_regression(study_directory, data_prefixes, labels_prefix):
    """
    Load multicontrast image data and a regression image target.
    :param study_directory: A directory containing the desired image data.
    :param data_prefixes: a list of filenames - the data files to be loaded
    :param labels_prefix: a list containing one string, the labels to be loaded
    :return: a tuple of np ndarrays containing the image data and regression target in the specified tf data format
    """

    # sanity checks
    if not os.path.isdir(study_directory): sys.exit("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in data_prefixes]): sys.exit("Data prefixes must be strings")
    if not all([isinstance(a, str) for a in labels_prefix]): sys.exit("Labels prefixes must be strings")

    # load multicontrast data
    data, nzi = _load_single_study(study_directory, data_prefixes, data_format='channels_last')

    # load labels data
    labels, nzi = _load_single_study(study_directory, labels_prefix, data_format='channels_last', slice_trim=nzi)

    # augment
    params = (np.random.random() * 90., np.random.random() > 0.5)
    data = _augment_image(data, params=params, data_format='channels_last', order=1)
    labels = _augment_image(labels, params=params, data_format='channels_last', order=1)  # linear interp for regression

    return data, labels


def display_tf_dataset(dataset_data):
    """
    Displays tensorflow dataset output images and labels/regression images.
    :param dataset_data: output from tensorflow dataset function containing images and labels/regression image
    :return: displays images for 3 seconds then continues
    """

    # make figure and configure close event timer
    fig = plt.figure(figsize=(10,4))
    def close_event():
        plt.close()
    timer = fig.canvas.new_timer(interval=3000)
    timer.add_callback(close_event)

    # image data
    image_data = dataset_data[0]
    nplots = image_data.shape[0]+1
    for z in range(image_data.shape[0]):
        ax = fig.add_subplot(1, nplots, z+1)
        ax.imshow(np.swapaxes(np.squeeze(image_data[z, :, :]), 0, 1), cmap='gray')
        ax.set_title('Data Image ' + str(z + 1))

    # label data
    label_data = dataset_data[1]
    ax = fig.add_subplot(1, nplots, nplots)
    ax.imshow(np.swapaxes(np.squeeze(label_data), 0, 1), cmap='gray')
    ax.set_title('Labels')

    # start timer and show plot
    timer.start()
    plt.show()

    return

