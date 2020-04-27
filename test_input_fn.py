from model.patch_input_fn import *
from model.utils import *
import argparse
import tensorflow as tf


# define functions
def test_input_fn(param_file):

    # get params
    params = Params(param_file)

    # determine model dir
    if params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        params.model_dir = os.path.dirname(param_file)
    if not os.path.isdir(params.model_dir):
        raise ValueError("Specified model directory does not exist")

    # load inputs with input function
    if params.dimension_mode == '2D':  # handle 2d inputs
        inputs = patch_input_fn(mode='train', params=params)
    elif params.dimension_mode in ['2.5D', '3D']:
        inputs = patch_input_fn_3d(mode='train', params=params)
    else:
        raise ValueError("Dimension mode not understood: " + str(params.dimension_mode))

    # run tensorflow session
    n = 0
    with tf.compat.v1.Session() as sess:
        for i in range(params.num_epochs):  # multiple epochs
            sess.run(inputs["iterator_init_op"])
            while True:
                # iterate through entire iterator
                try:
                    data_slice = sess.run({'features': inputs['features'], 'labels': inputs['labels']})
                except tf.errors.OutOfRangeError:
                    break

                # increment counter and show images
                n = n + 1
                print("Processing slice " + str(n) + " epoch " + str(i + 1))
                if n % 5 == 0:
                    display_tf_dataset(data_slice, params.data_format, params.train_dims)


# executed  as script
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', default=None,
                        help="Path to params.json")

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert args.param_file, "Must specify param file with --param_file"
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)

    # do work
    test_input_fn(args.param_file)
