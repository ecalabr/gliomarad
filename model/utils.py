"""General utility functions"""

import json
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Params:
    """
    Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    # declare attributes as None initially. All attributes defined here must be fed values from params.json.
    data_dir = None
    model_dir = None
    overwrite = None
    restore_dir = None
    data_prefix = None
    label_prefix = None
    mask_prefix = None
    mask_dilate = None # must have same number of dims as mask
    filter_zero = None # set the threshold for filtering out patches where labels is mostly zero

    dimension_mode = None  # must be 2D, 2.5D, 3D
    data_plane = None
    train_dims = None
    train_patch_overlap = None
    infer_dims = None
    infer_patch_overlap = None
    augment_train_data = None
    label_interp = None

    norm_data = None
    norm_labels = None
    norm_mode = None

    model_name = None
    base_filters = None
    output_filters = None
    layer_layout = None
    kernel_size = None
    data_format = None
    activation = None

    buffer_size = None
    shuffle_size = None
    batch_size = None
    num_threads = None
    train_fract = None
    learning_rate = None
    learning_rate_decay = None
    loss = None
    num_epochs = None
    dropout_rate = None

    save_summary_steps = None

    def __init__(self, json_path):
        self.update(json_path)
        self.check()

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def check(self):
        """Checks that all required parameters are defined in params.json file"""
        member_val = [getattr(self, attr) for attr in dir(self) if
                   not callable(getattr(self, attr)) and not attr.startswith("__")]
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        if any([member is None for member in member_val]):
            raise ValueError(
                "Missing the following parameter(s) in params.json: "
                + " ".join([attr for attr in members if getattr(self, attr) is None]))

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """
    Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def learning_rate_picker(learn_rate, learning_rate_decay, global_step):
    """
    Takes a learning rate and a string specifying a learning rate decay method and returns a learning rate.
    :param learn_rate: (float) the learning rate or list/tuple(float) the learning rate, step, and decay factor
    :param learning_rate_decay (string) the learning rate decay method
    :param global_step (tensorflow global step) the global step for the model
    :return: A learning rate function using the specified starting rate
    """

    # sanity checks
    if not isinstance(learn_rate, (float, list, tuple)):
        raise ValueError("Learning rate must be a float or list/tuple")
    if not isinstance(learning_rate_decay, (str, unicode)):
        raise ValueError("Learning rate decay parameter must be a string")

    # chooser for decay method
    if learning_rate_decay == 'constant':
        if isinstance(learn_rate, (list, tuple)):
            learn_rate = learn_rate[0]
        learning_rate_function = learn_rate

    # exponential decay
    elif learning_rate_decay == 'exponential':
        if not isinstance(learn_rate, (list, tuple)):
            raise ValueError("Exponential decay requres three values: starting learning rate, steps, and decay factor")
        start_lr = learn_rate[0]
        steps = learn_rate[1]
        decay = learn_rate[2]
        learning_rate_function = tf.train.exponential_decay(start_lr, global_step, steps, decay, staircase=True)

    # not implemented yet
    else:
        raise NotImplementedError("Specified learning rate decay method is not implemented: " + learning_rate_decay)

    return learning_rate_function


def loss_picker(loss_method, labels, predictions, data_format, weights=None):
    """
    Takes a string specifying the loss method and returns a tensorflow loss function
    :param loss_method: (str) the desired loss method
    :param labels: (tf.tensor) the labels tensor
    :param predictions: (tf.tensor) the features tensor
    :param data_format: (str) the tf data format 'channels_first' or 'channels_last'
    :param weights: (tf.tensor) an optional weight tensor for masking values
    :return: A tensorflow loss function
    """

    # sanity checks
    if not isinstance(loss_method, (str, unicode)): raise ValueError("Loss method parameter must be a string")
    if weights is None:
        weights = 1.0
    loss_methods = ['MSE', 'MAE', 'auxiliary_MAE', 'MSE25D', 'MAE25D', 'softmaxCE', 'weighted_dice', 'gen_dice']

    # chooser for decay method
    # MSE loss
    if loss_method == 'MSE':
        loss_function = tf.losses.mean_squared_error(labels, predictions, weights)

    # MAE loss
    elif loss_method == 'MAE':
        loss_function = tf.losses.absolute_difference(labels, predictions, weights)

    # auxiliary loss
    # https://www-sciencedirect-com.ucsf.idm.oclc.org/science/article/pii/S1361841518301257
    elif loss_method == 'auxiliary_MAE':

        # predefine loss_function
        loss_function = None

        # determine dimension of channels
        dim = 1 if data_format == 'channels_first' else -1

        # loop through the different predictions and sum to create auxillary loss
        for i in range(predictions.shape[dim]):
            # isolate pred
            if data_format == 'channels_first':
                pred = tf.expand_dims(predictions[:, i, :, :], axis=1)
            else:
                pred = tf.expand_dims(predictions[:, :, :, i], axis=-1)
            # generate MAE loss
            loss = tf.losses.absolute_difference(labels, pred, weights)
            if i == 0:  # for first loop, use full value of loss as this is the final predictions
                loss_function = loss
            else:  # for all subsequent loops, add the loss times 0.5, as these are auxiliary losses
                loss_function = tf.add(loss_function, loss * 0.5)

    # 2.5D MSE loss
    elif loss_method == 'MSE25D':
        # cast weights to float
        weights = tf.cast(weights, tf.float32)
        # handle channels last
        if data_format == 'channels_last':
            # get center slice for channels last [b, x, y, z, c] and double weights for this slice
            center_pred = predictions[:, :, :, predictions.shape[3] / 2 + 1, :]
            center_lab = labels[:, :, :, labels.shape[3] / 2 + 1, :]
            center_weights = tf.multiply(weights[:, :, :, weights.shape[3] / 2 + 1, :], 2.)
        # handle channels first
        elif data_format == 'channels_first':
            # get center slice for channels last [b, c, x, y, z] and double weights for this slice
            center_pred = predictions[:, :, :, :, predictions.shape[3] / 2 + 1]
            center_lab = labels[:, :, :, :, labels.shape[3] / 2 + 1]
            center_weights = tf.multiply(weights[:, :, :, :, weights.shape[3] / 2 + 1], 2.)
        else:
            raise ValueError("Data format not understood: " + str(data_format))

        # define loss as only the center slice first
        loss_function = tf.losses.mean_squared_error(
            center_pred, center_lab, center_weights, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        # add loss for the rest of the slab to the loss
        loss_function = tf.add(loss_function, tf.losses.mean_squared_error(
            predictions, labels, weights, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS))

    # 2.5D MAE loss
    elif loss_method == 'MAE25D':
        # handle channels last
        if data_format == 'channels_last':
            # get center slice for channels last [b, x, y, z, c]
            center_pred = predictions[:, :, :, predictions.shape[3] / 2 + 1, :]
            center_lab = labels[:, :, :, labels.shape[3] / 2 + 1, :]
            center_weights = weights[:, :, :, weights.shape[3] / 2 + 1, :]
        # handle channels first
        elif data_format == 'channels_first':
            # get center slice for channels first [b, c, x, y, z]
            center_pred = predictions[:, :, :, :, predictions.shape[3] / 2 + 1]
            center_lab = labels[:, :, :, :, labels.shape[3] / 2 + 1]
            center_weights = weights[:, :, :, :, weights.shape[3] / 2 + 1]
        else:
            raise ValueError("Data format not understood: " + str(data_format))

        # define loss
        loss_function = tf.losses.absolute_difference(center_lab, center_pred, center_weights)

        # add remaining slices at equal weight
        loss_function = tf.add(loss_function, tf.losses.absolute_difference(labels, predictions, weights))

    # softmax cross entropy w logits
    elif loss_method == 'softmaxCE':
        # handle channels last
        if data_format == 'channels_last':
            # channels last [b, x, y, z, c]
            axis = -1
        # handle channels first
        elif data_format == 'channels_first':
            # channels first [b, c, x, y, z]
            axis = 1
        else:
            raise ValueError("Data format not understood: " + str(data_format))
        # define loss function
        loss_function = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(labels, tf.int32),
            logits=predictions,
            weights=1.0, # not weighting using mask as mask is target
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

    # weighted DICE loss - not working as expected atm - likely need to flatten
    elif loss_method == 'weighted_dice':
        axis = -1 if data_format == 'channels_last' else 1
        smooth=0.00001
        y = tf.cast(labels, tf.float32)
        if data_format == 'channels_last':
            y_pred = tf.nn.softmax(predictions, axis=axis)[..., 1, np.newaxis]
        else:
            y_pred = tf.nn.softmax(predictions, axis=axis)[:, 1, np.newaxis, ...]
        intersection = tf.reduce_sum(y * y_pred, axis=axis)
        union = tf.reduce_sum(y, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
        weighted_dice =  tf.reduce_mean(2. * (intersection + smooth / 2) / (union + smooth))
        loss_function = - weighted_dice

    # generalized DICE loss for 2D and 2.5D networks
    elif loss_method == 'gen_dice':

        # convert labels to int and softmax pred
        y = tf.cast(labels, tf.float32)
        y_pred = tf.nn.softmax(predictions, axis=-1 if data_format == 'channels_last' else 1)

        # handle 2.5D by making a triangle weight kernel weighting center slice the most and sum=1
        if len(y_pred.shape) == 5 and data_format == 'channels_last':
            # for channels last [b, x, y, z, c]
            kerlen = y_pred.get_shape().as_list()[3]
            kernel = (float(kerlen) + 1. - np.abs(np.arange(float(kerlen)) - np.arange(float(kerlen))[::-1])) / 2.
            kernel /= kernel.sum()
            kernel = tf.reshape(tf.constant(kernel, dtype=tf.float32), shape=[1, 1, 1, kerlen, 1])
            y = tf.reduce_sum(tf.multiply(y, kernel), axis=3)
            y_pred = tf.reduce_sum(tf.multiply(y_pred, kernel), axis=3)
        # handle channels first
        elif len(y_pred.shape) == 5 and data_format == 'channels_first':
            # for channels first [b, c, x, y, z]
            kerlen = y_pred.get_shape().as_list()[4]
            kernel = (kerlen + 1 - np.abs(np.arange(kerlen) - np.arange(kerlen)[::-1])) / 2
            kernel /= kernel.sum()
            kernel = tf.reshape(tf.constant(kernel, dtype=tf.float32), shape=[1, 1, 1, 1, kerlen])
            y = tf.reduce_sum(tf.multiply(y, kernel), axis=4)
            y_pred = tf.reduce_sum(tf.multiply(y_pred, kernel), axis=4)

        mean = 0.5 # the weight is the proportion of 1s to 0s in the label set
        w_1 = 1. / mean ** 2.
        w_0 = 1. / (1. - mean) ** 2.
        y_true_f_1 = tf.reshape(y, [-1])
        y_pred_f_1 = tf.reshape(y_pred[..., 1] if data_format == 'channels_last' else y_pred[:, 1, ...], [-1])
        y_true_f_0 = tf.reshape(1 - y, [-1])
        y_pred_f_0 = tf.reshape(y_pred[..., 0] if data_format == 'channels_last' else y_pred[:, 0, ...], [-1])
        int_0 = tf.reduce_sum(y_true_f_0 * y_pred_f_0)
        int_1 = tf.reduce_sum(y_true_f_1 * y_pred_f_1)
        gen_dice = 2. * (w_0 * int_0 + w_1 * int_1) / ((w_0 * (tf.reduce_sum(y_true_f_0) + tf.reduce_sum(y_pred_f_0))) +
                                                       (w_1 * (tf.reduce_sum(y_true_f_1) + tf.reduce_sum(y_pred_f_1))))
        loss_function = - gen_dice

    # not implemented loss
    else:
        raise NotImplementedError("Specified loss method is not implemented: " + loss_method +
                                  ". Possible options are: " + ' '.join(loss_methods))

    return loss_function


def display_tf_dataset(dataset_data, data_format, data_dims):
    """
    Displays tensorflow dataset output images and labels/regression images.
    :param dataset_data: (tf.tensor) output from tf dataset function containing images and labels/regression image
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param data_dims: (list or tuple of ints) the data dimensions that come out of the input function
    :return: displays images for 3 seconds then continues
    """

    # make figure
    fig = plt.figure(figsize=(10, 4))

    # define close event and create timer
    def close_event():
        plt.close()
    timer = fig.canvas.new_timer(interval=3000)
    timer.add_callback(close_event)

    # handle 2d case
    if len(data_dims) == 2:
        # image data
        image_data = dataset_data["features"]  # dataset_data[0]
        if len(image_data.shape) > 3:
            image_data = np.squeeze(image_data[0, :, :, :])  # handle batch data
        nplots = image_data.shape[0] + 1 if data_format == 'channels_first' else image_data.shape[2] + 1
        channels = image_data.shape[0] if data_format == 'channels_first' else image_data.shape[2]
        for z in range(channels):
            ax = fig.add_subplot(1, nplots, z + 1)
            data_img = np.swapaxes(np.squeeze(image_data[z, :, :]), 0, 1) if data_format == 'channels_first' else np.squeeze(
                image_data[:, :, z])
            ax.imshow(data_img, cmap='gray')
            ax.set_title('Data Image ' + str(z + 1))

        # label data
        label_data = dataset_data["labels"]  # dataset_data[1]
        if len(label_data.shape) > 3: label_data = np.squeeze(label_data[0, :, :, :])  # handle batch data
        ax = fig.add_subplot(1, nplots, nplots)
        label_img = np.swapaxes(np.squeeze(label_data), 0, 1) if data_format == 'channels_first' else np.squeeze(label_data)
        ax.imshow(label_img, cmap='gray')
        ax.set_title('Labels')

    # handle 3d case
    if len(data_dims) == 3:

        # load image data
        image_data = dataset_data["features"]  # dataset_data[0]

        # handle channels first and batch data
        if len(image_data.shape) > 4:
            if data_format == 'channels_first':
                image_data = np.transpose(image_data, [0, 2, 3, 4, 1])
            image_data = np.squeeze(image_data[0, :, :, :, :])  # handle batch data
        else:
            if data_format == 'channels_first':
                image_data = np.transpose(image_data, [1, 2, 3, 0])

        # determine n plots and channels
        nplots = image_data.shape[-1] + 1
        channels = image_data.shape[-1]

        # loop through channels
        for z in range(channels):
            ax = fig.add_subplot(1, nplots, z + 1)
            data_img = np.squeeze(image_data[:, :, :, z])
            # concatenate along z to make 1 2d image per slab
            data_img = np.reshape(np.transpose(data_img), [data_img.shape[0] * data_img.shape[2], data_img.shape[1]])
            ax.imshow(data_img, cmap='gray')
            ax.set_title('Data Image ' + str(z + 1))

        # load label data
        label_data = dataset_data["labels"]  # dataset_data[1]

        # handle channels first and batch data
        if len(label_data.shape) > 4:
            if data_format == 'channels_first':
                label_data = np.transpose(label_data, [0, 2, 3, 4, 1])
            label_data = label_data[0, :, :, :, :]  # handle batch data
        else:
            if data_format == 'channels_first':
                label_data = np.transpose(label_data, [1, 2, 3, 0])

        # add to fig
        ax = fig.add_subplot(1, nplots, nplots)
        label_img = np.squeeze(label_data)

        # concatenate along z to make 1 2d image per slab
        label_img = np.reshape(np.transpose(label_img), [label_img.shape[0] * label_img.shape[2], label_img.shape[1]])
        ax.imshow(label_img, cmap='gray')
        ax.set_title('Labels')

    # start timer and show plot
    timer.start()
    plt.show()

    return
