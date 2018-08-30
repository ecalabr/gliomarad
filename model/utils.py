"""General utility functions"""

import json
import logging
import sys


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
    restore_dir = None
    data_prefix = None
    label_prefix = None
    data_height = None
    data_width = None
    augment_train_data = None
    label_interp = None

    model_name = None
    base_filters = None
    kernel_size = None
    data_format = None

    buffer_size = None
    shuffle_size = None
    batch_size = None
    num_threads = None
    train_fract = None
    learning_rate = None
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