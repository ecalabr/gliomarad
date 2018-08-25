import tensorflow as tf
import os
import sys
from glob import glob
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import scipy.ndimage.interpolation as interp


def _load_single_study(study_dir, file_prefixes, data_format='channels_last'):
    """
    Image data I/O function for use in tensorflow Dataset map function. Takes a study directory and file prefixes and
    returns a 4D numpy array containing the image data.
    :param study_dir: A string - the full path to the study directory
    :param file_prefixes: A string or list of strings - the file prefixes for the images to be loaded
    :param data_format: the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :return:
        output - a 4D numpy array containing the image data
    """

    # sanity checks
    if not os.path.isdir(study_dir): sys.exit("Specified study_dir does not exist")
    if not isinstance(file_prefixes, (str, list)): sys.exit("file_prefixes must be a string or list of strings")
    if isinstance(file_prefixes, str): file_prefixes = [file_prefixes]
    if data_format not in ['channels_last', 'channels_first']: sys.exit("data_format invalid")
    if not isinstance(aug, (bool, list)): sys.exit("aug parameter must be a boolean or list")

    # convert from directory name to image names
    images = [glob(study_dir + '/*' + contrast + '*.nii.gz')[0] for contrast in file_prefixes]
    output = []
    nz_inds = [0, -1]

    # load images and concatenate into a 4d numpy array
    for ind, image in enumerate(images):
        if ind == 0:  # find dimensions after trimming zero slices and preallocate 4d array
            first_image = nib.load(images[0]).get_fdata()
            nz_inds = _nonzero_slice_inds(first_image)
            output_shape = list(first_image.shape)[0:3] + [len(images)]
            output = np.zeros(output_shape, np.float32)
            output[:, :, :, 0] = first_image[:, :, nz_inds[0]:nz_inds[1]]
        else:
            output[:, :, :, ind] = nib.load(images[ind]).get_fdata()[:, :, nz_inds[0]:nz_inds[1]]

    # permute data to desired data format
    if data_format == 'channels_last':
        output = np.transpose(output, axes=(2, 3, 0, 1))
    else:
        output = np.transpose(output, axes=(2, 0, 1, 3))

    return output


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


def _load_multicon_and_labels(study_directory):

    # sanity checks
    if not os.path.isdir(study_directory): sys.exit("Specified study_directory does not exist")

    # load multicontrast data
    data_prefixes = ['FLAIR_wm', 'T1_wm', 'T1gad_wm', 'T2_wm']
    data = _load_single_study(study_directory, data_prefixes, data_format='channels_last')

    # load labels data
    labels_prefix = ['ASL_wm']
    labels = _load_single_study(study_directory, labels_prefix, data_format='channels_last')

    return data, labels


def _augment_image(input_data, data_format, params=(np.random.random() * 90., np.random.random() > 0.5)):
    """
    Takes input numpy array and a data format and performs data augmentation with random rotations and flips
    :param input_data: a 4D numpy array or list/tuple of arrays containing image data in the specified TF data format
    :param data_format: the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param params: A list or tuple of user specified values in format [rotation=float(degrees), flip=bool]
    :return:
        output_data - a numpy array or tuple of numpy arrays containing the augmented data
    """

    # sanity checks
    if isinstance(input_data, (list, tuple)):
        if not all([type(x) is np.ndarray for x in input_data]):
            sys.exit("Input data list/tuple must be all np arrays")
        else:
            data = input_data[0]
            labels = input_data[1]
    elif type(input_data) is not np.ndarray: sys.exit("Input must be numpy array or list/tuple of numpy arrays")
    else:
        data = input_data
        labels = None
    if data_format not in ['channels_last', 'channels_first']: sys.exit("data_format invalid")
    if not isinstance(params, (list, tuple)): sys.exit("Params must be a list or tuple if specified")
    if not isinstance(params[0], (float, int)): sys.exit("First entry in params must be a float or int")
    if not isinstance(params[1], bool): sys.exit("Second entry in params must be a boolean")

    # apply rotation
    if data_format == 'channels_last':
        axes = (2, 3)
    else:
        axes = (1, 2)
    data = interp.rotate(data, float(params[0]), axes=axes, reshape=False, order=1)  # bicubic interp
    if labels:
        labels = interp.rotate(labels, float(params[0]), axes=axes, reshape=False, order=1)  # NN interp

    # apply flip
    if params[1]:
        data = np.flip(data, axis=axes[0])
        if labels:
            labels = np.flip(labels, axis=axes[0])

    # recombine arrays into tuple if two were given, otherwise just return one array
    if labels:
        data = (data, labels)

    return data


### Temporary code

BUFFER_SIZE = 12
SHUFFLE_SIZE = 6
NUM_THREADS = 16

main_data_dir = '/media/ecalabr/data2/io_test'
study_numbers = '11045377', '11053877', '11065772'

main_data_dir = '/media/ecalabr/data2/qc_complete'
study_numbers = ['10672000', '10846904', '10940662', '11038642', '11129704', '11196914', '11318417', '11419123',
                 '11490693', '11597627', '11771760', '11914431', '12022810', '12133297', '12288677', '10673199',
                 '10848682', '10944302', '11040352', '11134115', '11218876', '11322899', '11423959', '11498167',
                 '11650104', '11775500', '11922167', '12054419', '12141357', '12299574', '10753655', '10855761',
                 '10957443', '11045377', '11140487', '11247744', '11325665', '11435751', '11500491', '11665288',
                 '11791685', '11936344', '12057904', '12168079', '12303318', '10754135', '10869237', '10965515',
                 '11053877', '11149024', '11252810', '11331683', '11436869', '11519429', '11666513', '11800774',
                 '11946017', '12058076', '12181551', '12309838', '10757830', '10871864', '10972773', '11065772',
                 '11150443', '11271681', '11362040', '11440727', '11534009', '11668634', '11808193', '11973818',
                 '12062889', '12182783', '12319781', '10766003', '10887875', '10981713', '11086362', '11171727',
                 '11273473', '11367916', '11466129', '11541506', '11671585', '11849170', '11996514', '12067216',
                 '12204550', '10774926', '10908346', '11010654', '11087024', '11179085', '11273740', '11371239',
                 '11466591', '11541709', '11684227', '11851391', '11999752', '12073358', '12208812', '10828712',
                 '10908535', '11014236', '11097040', '11187345', '11280555', '11372974', '11469772', '11543758',
                 '11688244', '11856683', '12001619', '12092613', '12247111', '10832837', '10908765', '11016986',
                 '11117844', '11191958', '11284029', '11392645', '11478485', '11566163', '11720435', '11879297',
                 '12006472', '12104287', '12267857', '10834262', '10926580', '11034531', '11120271', '11193909',
                 '11316744', '11418822', '11482456', '11585757', '11734883', '11901030', '12014144', '12107086',
                 '12286876']

study_dirs = [os.path.join(main_data_dir, s) for s in study_numbers]


# need to add batching and labels input

study_dirs = tf.constant(study_dirs)
dataset = tf.data.Dataset.from_tensor_slices(study_dirs)
dataset = dataset.map(lambda x: tf.py_func(_load_multicon_and_labels, [x], tf.float32), num_parallel_calls=NUM_THREADS)
dataset = dataset.prefetch(BUFFER_SIZE)
dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataset = dataset.shuffle(SHUFFLE_SIZE)
iterator = dataset.make_one_shot_iterator()
get_next = iterator.get_next()

with tf.Session() as sess:
    sess.run(get_next)
    for i in range(2000):
        data_slice = sess.run(get_next)
        if i%250 == 0:
            print(i)
            image_data = data_slice[0]
            concat = image_data.reshape(-1, image_data.shape[2])
            imgplot = plt.imshow(np.swapaxes(concat, 0, 1))
            plt.show()
