from net_layers import *
from glob import glob
import tensorflow as tf
from tf_data_io_func import *



# GLOBAL VARIABLES
DATA_DIR = '/media/ecalabr/data2/qc_complete'
DATA_PREFIX = ['T1_wm', 'T2_wm', 'FLAIR_wm', 'DWI_wm', 'ASL_wm']
LABEL_PREFIX = ['T1gad_wm']
BUFFER_SIZE = 12
SHUFFLE_SIZE = 6
BATCH_SIZE = 4
NUM_THREADS = 16
TRAIN_FRAC = 0.8



# Study dirs and prefixes setup
study_dirs = glob(DATA_DIR + '/*/')
train_dirs = tf.constant(study_dirs[0:round(TRAIN_FRAC*len(study_dirs))])
eval_dirs = tf.constant(study_dirs[round(TRAIN_FRAC*len(study_dirs)):])
data_prefix = tf.constant(DATA_PREFIX)
label_prefix = tf.constant(LABEL_PREFIX)

# tf.dataset setup
dataset = tf.data.Dataset.from_tensor_slices(study_dirs)
dataset = dataset.map(lambda x: tf.py_func(load_multicon_and_regression, [x, data_prefix, label_prefix], (tf.float32, tf.float32)), num_parallel_calls=NUM_THREADS)
dataset = dataset.prefetch(BUFFER_SIZE)
dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
dataset = dataset.shuffle(SHUFFLE_SIZE)
#dataset = dataset.batch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
get_next = iterator.get_next()

# run tensorflow session
n = 0
with tf.Session() as sess:
    while True:

        # iterate through entire iterator
        try:
            data_slice = sess.run(get_next)
        except tf.errors.OutOfRangeError:
            break

        # increment counter and show images
        n = n+1
        print("Processing slice " + str(n))
        if n % 250 == 0:
            display_tf_dataset(data_slice)
