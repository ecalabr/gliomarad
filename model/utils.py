"""General utility functions"""

import json
import logging
import tensorflow as tf


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
    data_height = None
    data_width = None
    augment_train_data = None
    label_interp = None

    model_name = None
    base_filters = None
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
        members = [getattr(self, attr) for attr in dir(self) if
                   not callable(getattr(self, attr)) and not attr.startswith("__")]
        if any([member is None for member in members]):
            raise ValueError("One or more parameters is not defined in params.json")

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


def learning_rate_picker(learning_rate, learning_rate_decay, global_step):
    """
    Takes a learning rate and a string specifying a learning rate decay method and returns a learning rate.
    :param learning_rate: (float) the learning rate
    :param learning_rate_decay (string) the learning rate decay method
    :param global_step (tensorflow global step) the global step for the model
    :return: A learning rate function using the specified starting rate
    """

    # sanity checks
    if not isinstance(learning_rate, float): raise ValueError("Learning rate must be a float")
    if not isinstance(learning_rate_decay, (str, unicode)): raise ValueError("Learning rate decay parameter must be a string")

    # chooser for decay method
    if learning_rate_decay == 'constant':
        learning_rate_function = learning_rate
    elif learning_rate_decay == 'exponential':
        learning_rate_function = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.9, staircase=True)
    else:
        raise NotImplementedError("Specified learning rate decay method is not implemented: " + learning_rate_decay)

    return learning_rate_function


def loss_picker(loss_method, labels, predictions, weights=None):
    """
    Takes a string specifying the loss method and returns a tensorflow loss function
    :param loss_method: (str) the desired loss method
    :param labels: (tf.tensor) the labels tensor
    :param predictions: (tf.tensor) the features tensor
    :param weights: (tf.tensor) an optional weight tensor for masking values
    :return: A tensorflow loss function
    """

    # sanity checks
    if not isinstance(loss_method, (str, unicode)): raise ValueError("Loss method parameter must be a string")
    if weights is None: weights = 1.0

    # chooser for decay method
    if loss_method == 'MSE':
        loss_function = tf.losses.mean_squared_error(labels, predictions, weights)
    elif loss_method == 'MAE':
        loss_function = tf.losses.absolute_difference(labels, predictions, weights)
    else:
        raise NotImplementedError("Specified loss method is not implemented: " + loss_method)

    return loss_function
