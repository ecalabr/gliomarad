from model.input_fn import *
from model.utils import *

# load params
params = Params('/home/ecalabr/PycharmProjects/gbm_preproc/model/params.json')

#dataset = input_fn(is_training=True, params=params)
#iterator = dataset.make_one_shot_iterator()
#get_next = iterator.get_next()

#
inputs = input_fn(is_training=True, params=params)

# run tensorflow session
n = 0
with tf.Session() as sess:
    sess.run(inputs["iterator_init_op"])
    while True:
        # iterate through entire iterator
        try:
            data_slice = sess.run({'features': inputs['features'], 'labels': inputs['labels']})
        except tf.errors.OutOfRangeError:
            break

        # increment counter and show images
        n = n+1
        print("Processing slice " + str(n))
        if n % 250 == 0:
            display_tf_dataset(data_slice)
