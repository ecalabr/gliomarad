import os
import sys
import math
from glob import glob
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import scipy.ndimage.interpolation as interp
import tensorflow as tf


def _load_single_study(study_dir, file_prefixes, data_format, slice_trim=None):
    """
    Image data I/O function for use in tensorflow Dataset map function. Takes a study directory and file prefixes and
    returns a 4D numpy array containing the image data.
    :param study_dir: (str) the full path to the study directory
    :param file_prefixes: (str, list(str)) the file prefixes for the images to be loaded
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param slice_trim: (list, tuple) contains 2 ints, the first and last slice to use for trimming. None = auto trim
    :return: output - a 4D numpy array containing the image data
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
    if data_format == 'channels_first':
        output = np.transpose(output, axes=(2, 3, 0, 1))
    else:
        output = np.transpose(output, axes=(2, 0, 1, 3))

    return output, nz_inds


def _nonzero_slice_inds(input_numpy):
    """
    Takes numpy array and returns slice indices of first and last nonzero slices
    :param input_numpy: (np.ndarray) a numpy array containing image data
    :return: inds - a list of 2 indices corresponding to the first and last nonzero slices in the numpy array
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
    :param input_data: (np.ndarray) a 4D numpy array containing image data in the specified TF data format
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param params: (list, tuple) A list or tuple of user specified values in format [rotation=float(degrees), flip=bool]
    :param order: (int) an int from 0-5, the order for spline interpolation, default is 1 (linear)
    :return: output_data - a numpy array or tuple of numpy arrays containing the augmented data
    """

    # sanity checks
    if type(input_data) is not np.ndarray: sys.exit("Input must be numpy array or list/tuple of numpy arrays")
    if data_format not in ['channels_last', 'channels_first']: sys.exit("data_format invalid")
    if not isinstance(params, (list, tuple)): sys.exit("Params must be a list or tuple if specified")
    if not isinstance(params[0], (float, int)): sys.exit("First entry in params must be a float or int")
    if not isinstance(params[1], bool): sys.exit("Second entry in params must be a boolean")
    if order not in range(6): sys.exit("Spline interpolation order must be in range 0-3")

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
    :param out_dims: (list(int)) the desired output dimensions.
    :param axes: (list(int)) the axes for padding. Must have same length as out_dims
    :return: (np.ndarray) the zero padded image
    """

    # sanity checks
    if type(input_data) is not np.ndarray: sys.exit("Input must be a numpy array")
    if not all([np.issubdtype(val, np.integer) for val in out_dims]): sys.exit(
        "Output dims must be a list or tuple of ints")
    if not all([isinstance(axes, (tuple, list))] + [isinstance(val, int) for val in axes]): sys.exit(
        "Axes must be a list or tuple of ints")
    if not len(out_dims) == len(axes): sys.exit("Output dimensions must have same length as axes")
    if len(axes) != len(set(axes)): sys.exit("Axes cannot contain duplicate values")

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


def load_multicon_and_labels(study_dir, feature_prefx, label_prefx, data_fmt, out_dims, augment, label_interp=1):
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
    if not os.path.isdir(study_dir): sys.exit("Specified study_directory does not exist")
    if not all([isinstance(a, str) for a in feature_prefx]): sys.exit("Data prefixes must be strings")
    if not all([isinstance(a, str) for a in label_prefx]): sys.exit("Labels prefixes must be strings")
    if data_fmt not in ['channels_last', 'channels_first']: sys.exit("data_format invalid")
    if not all([np.issubdtype(a, np.integer) for a in out_dims]): sys.exit("data_dims must be a list/tuple of ints")
    if augment not in ['yes', 'no']: sys.exit("Parameter augment must be 'yes' or 'no'")
    if label_interp not in range(6): sys.exit("Spline interpolation order must be in range 0-3")

    # load multi-contrast data
    data, nzi = _load_single_study(study_dir, feature_prefx, data_format=data_fmt)

    # load labels data
    labels, nzi = _load_single_study(study_dir, label_prefx, data_format=data_fmt, slice_trim=nzi)

    # if augmentation is to be used generate random params for augmentation and run augment function
    if augment == 'yes':
        params = (np.random.random() * 30., np.random.random() > 0.5)
        data = _augment_image(data, params=params, data_format=data_fmt, order=1)  # force linear interp for images
        labels = _augment_image(labels, params=params, data_format=data_fmt, order=label_interp)

    # do data padding to desired dims
    axes = [1, 2] if data_fmt == 'channels_last' else [2, 3]
    data = _zero_pad_image(data, out_dims, axes)
    labels = _zero_pad_image(labels, out_dims, axes)

    return data, labels


def display_tf_dataset(dataset_data, data_format):
    """
    Displays tensorflow dataset output images and labels/regression images.
    :param dataset_data: (tf.tensor) output from tf dataset function containing images and labels/regression image
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :return: displays images for 3 seconds then continues
    """

    # make figure
    fig = plt.figure(figsize=(10,4))

    # define close event and create timer
    def close_event():
        plt.close()
    timer = fig.canvas.new_timer(interval=3000)
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


def input_fn(is_training, params):
    """
    Input function for UCSF GBM dataset
    :param is_training: (bool) whether or not the model is training, if training, data augmentation is used
    :param params: (class) the params class generated from a JSON file
    :return: outputs, a dict containing the features, labels, and initializer operation
    """

    # Study dirs and prefixes setup
    study_dirs = glob(params.data_dir + '/*/')
    train_dirs = tf.constant(study_dirs[0:int(round(params.train_fract * len(study_dirs)))])
    eval_dirs = tf.constant(study_dirs[int(round(params.train_fract * len(study_dirs))):])

    # differences between training and non-training: augment, dirs
    data_dims = [params.data_height, params.data_width]
    if is_training:
        data_dirs = train_dirs
        py_func_params = [params.data_prefix, params.label_prefix, params.data_format, data_dims, params.augment_train_data,
                          params.label_interp]
    else:
        data_dirs = eval_dirs
        py_func_params = [params.data_prefix, params.label_prefix, params.data_format, data_dims, 'no']  # 'no' for aug

    # generate tensorflow dataset object
    dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
    dataset = dataset.map(
        lambda x: tf.py_func(load_multicon_and_labels,
                             [x] + py_func_params,
                             (tf.float32, tf.float32)), num_parallel_calls=params.num_threads)
    dataset = dataset.prefetch(params.buffer_size)
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices({"features": x, "labels": y}))
    dataset = dataset.shuffle(params.shuffle_size)
    # dataset = dataset.batch(params.batch_size)  # this is causing errors in conv transpose when batch size changes
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))

    # make iterator and query the output of the iterator for input to the model
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()
    init_op = iterator.initializer

    # manually set shapes for inputs
    get_next_features = get_next['features']
    get_next_features.set_shape([params.batch_size, params.data_height, params.data_width, len(params.data_prefix)])
    get_next_labels = get_next['labels']
    get_next_labels.set_shape([params.batch_size, params.data_height, params.data_width, len(params.label_prefix)])

    # Build and return a dictionary containing the nodes / ops
    # inputs = {'features': get_next['features'], 'labels':get_next['labels'], 'iterator_init_op': init_op}
    inputs = {'features': get_next_features, 'labels': get_next_labels, 'iterator_init_op': init_op}

    return inputs
