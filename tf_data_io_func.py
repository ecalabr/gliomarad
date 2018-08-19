import tensorflow as tf
import os
from glob import glob
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import scipy.ndimage.interpolation as interp


# I/O function that takes a single study directory and a list of contrasts and loads all images to a 4D numpy array
# make this focus only on loading multi contrast input and add loading of labels
def _load_single_study(study_dir, contrasts=('FLAIR_wm', 'T1_wm', 'T1gad_wm', 'T2_wm'), data_format='channels_last',
                       aug=False):
    # convert from directory name to image names
    images = [glob(study_dir + '/*' + contrast + '*.nii.gz')[0] for contrast in contrasts]
    # get shape info and populate first image into array
    first_image = nib.load(images[0])
    first_image = first_image.get_fdata()
    nz_inds = _nonzero_slice_inds(first_image)
    first_image = first_image[:,:, nz_inds[0]:nz_inds[1]]
    # handle single image contrast
    if len(images) == 1:
        output = np.transpose(first_image, axes=(1, 2, 0))
        # permute output to specified data format, i.e. from HWNC to NCHW (ch first) or NHWC (ch last)
        if data_format == 'channels_last':
            output = np.expand_dims(output, 3)
            if aug: # handle data augmentation
                output = _augment_image(output, data_format)
        else:
            output = np.expand_dims(output, 1)
            if aug: # handle data augmentation
                output = _augment_image(output, data_format)
    # handle multiple image contrasts
    else:
        output_shape = list(first_image.shape)[0:3] + [len(images)]
        output = np.zeros(output_shape, np.float32)
        output[:, :, :, 0] = first_image
        # load the rest of the images into the array
        for ind, image in enumerate(images[1:], 1):
            img = nib.load(image)
            output[:, :, :, ind] = img.get_fdata()[:,:, nz_inds[0]:nz_inds[1]]
        # permute output to specified data format, i.e. from HWNC to NCHW (ch first) or NHWC (ch last)
        if data_format == 'channels_last':
            output = np.transpose(output, axes=(2, 3, 0, 1))
            if aug: # handle data augmentation
                output = _augment_image(output, data_format)
        else:
            output = np.transpose(output, axes=(2, 0, 1, 3))
            if aug: # handle data augmentation
                output = _augment_image(output, data_format)
    return output

def _nonzero_slice_inds(input_numpy):
    # finds inds of first and last nonzero slices
    vector = np.max(np.max(input_numpy, axis=0), axis=0)
    nz = np.nonzero(vector)
    st = nz[0][0]
    end = nz[0][-1]
    slice_trim = 2 # number of additional slices to trim on either side
    inds = [st+slice_trim, end-slice_trim]
    return inds

def _augment_image(input_data, data_format):
    degrees = np.random.random() * 90.
    if data_format == 'channels_last':
        axes = (2, 3)
    else:
        axes = (1, 2)
    output_data = interp.rotate(input_data, degrees, axes=axes, reshape=False, order=1)
    if np.random.random() > 0.5:
        output_data = np.flip(output_data, axis=axes[0])
    return output_data



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
dataset = dataset.map(lambda file_name: tf.py_func(_load_single_study, [file_name], tf.float32), num_parallel_calls=NUM_THREADS)
dataset = dataset.prefetch(BUFFER_SIZE)
dataset = dataset.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
dataset = dataset.shuffle(SHUFFLE_SIZE)
iterator = dataset.make_one_shot_iterator()
get_next = iterator.get_next()

with tf.Session() as sess:
    sess.run(get_next)
    for i in range(3000):
        data_slice = sess.run(get_next)
        if i%250 == 0:
            print(i)
            array = data_slice[0]
            concat = array.reshape(-1, array.shape[2])
            imgplot = plt.imshow(np.swapaxes(concat,0,1))
            plt.show()