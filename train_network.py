from glob import glob
import tensorflow as tf
from tf_data_io_func import *


# GLOBAL VARIABLES
BUFFER_SIZE = 12
SHUFFLE_SIZE = 6
NUM_THREADS = 16
DATA_DIR = '/media/ecalabr/data2/qc_complete'


# dataset setup
study_dirs = glob(DATA_DIR + '/*/')
study_dirs = tf.constant(study_dirs)
dataset = tf.data.Dataset.from_tensor_slices(study_dirs)
dataset = dataset.map(lambda x: tf.py_func(load_multicon_and_regression, [x], (tf.float32, tf.float32)), num_parallel_calls=NUM_THREADS)
dataset = dataset.prefetch(BUFFER_SIZE)
dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
dataset = dataset.shuffle(SHUFFLE_SIZE)
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
