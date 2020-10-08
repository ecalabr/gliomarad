import random
from utilities.input_fn_util import *
import json
import logging


# COMPLETE 2D INPUT FUNCTIONS
def _patch_input_fn_2d(params, mode, train_dirs, eval_dirs, infer_dir=None):
    # generate input dataset objects for the different training modes

    # train mode - uses patches, patch filtering, batching, data augmentation, and shuffling - works on train_dirs
    if mode == 'train':
        # variable setup
        data_dirs = tf.constant(train_dirs)
        data_chan = len(params.data_prefix)
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.mask_prefix,
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          params.augment_train_data,
                          params.label_interp,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # randomly shuffle directory order
        dataset = dataset.shuffle(buffer_size=len(data_dirs))
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_roi_multicon_and_labels,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32)),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches
        dataset = dataset.map(
            lambda x, y: tf_patches(x, y, params.train_dims, data_chan, params.data_format,
                                    overlap=params.train_patch_overlap),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # filter out zero patches
        if params.filter_zero > 0.:
            dataset = dataset.filter(lambda x, y: filter_zero_patches(
                y, params.data_format, params.dimension_mode, params.filter_zero))
        # shuffle a set number of exampes
        dataset = dataset.shuffle(buffer_size=params.shuffle_size)
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)
        # prefetch with experimental autotune
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        # repeat dataset infinitely so that dataset doesn't exhaust prematurely during fit
        dataset = dataset.repeat()

    # eval mode - uses patches and batches but no patch filtering, data augmentation, or shuffling, works on eval_dirs
    elif mode == 'eval':
        # variable setup
        data_dirs = tf.constant(eval_dirs)
        data_chan = len(params.data_prefix)
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.mask_prefix,
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          False, # do not do data augmentation on eval data
                          params.label_interp,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_roi_multicon_and_labels,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads) # tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches - USE SAME DATA DIMS AS TRAINING
        dataset = dataset.map(
            lambda x, y: tf_patches(x, y, params.train_dims, data_chan, params.data_format,
                                    overlap=params.train_patch_overlap),
            num_parallel_calls=params.num_threads) # tf.data.experimental.AUTOTUNE)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)
        # prefetch with experimental autotune
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # infer mode - does not use patches (patch function uses infer_dims), batches, or shuffling - works on infer_dir
    elif mode == 'infer':
        if not infer_dir:
            assert ValueError("Must specify inference directory for inference mode")
        dirs = tf.constant(infer_dir)
        # define dims of inference
        data_dims = list(params.infer_dims)
        chan_size = len(params.data_prefix)
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix, params.data_format, params.data_plane, params.norm_data, params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_multicon_preserve_size,
                                        [x] + py_func_params,
                                        tf.float32),
                                        num_parallel_calls=params.num_threads) # tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches based on infer inputs
        dataset = dataset.map(
            lambda x: tf_patches_infer(x, data_dims, chan_size, params.data_format, params.infer_patch_overlap),
                                       num_parallel_calls=params.num_threads) # tf.data.experimental.AUTOTUNE)
        # flat map so that each tensor is a single slice
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        # generate a batch of data
        dataset = dataset.batch(batch_size=1, drop_remainder=False) # force batch size 1 to ensure all data is processed
        # automatic prefetching to improve efficiency
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # error if not train, eval, or infer
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    return dataset


def _patch_input_fn_3d(params, mode, train_dirs, eval_dirs, infer_dir=None):
    # generate input dataset objects for the different training modes

    # train mode - uses patches, patch filtering, batching, data augmentation, and shuffling - works on train_dirs
    if mode == 'train':
        # variable setup
        data_dirs = tf.constant(train_dirs)
        data_chan = len(params.data_prefix)
        weighted = False if isinstance(params.mask_weights, np.bool) and not params.mask_weights else True
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.mask_prefix,
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          params.augment_train_data,
                          params.label_interp,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode,
                          params.mask_weights]  # param makes loader return weights as last channel in labels data]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # randomly shuffle directory order
        dataset = dataset.shuffle(buffer_size=len(data_dirs))
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_roi_multicon_and_labels_3d,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32)),
                                        num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches
        dataset = dataset.map(
            lambda x, y: tf_patches_3d(x, y, params.train_dims, params.data_format, data_chan,
                                       weighted=weighted,
                                       overlap=params.train_patch_overlap),
                                       num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # filter out zero patches
        if params.filter_zero > 0.:
            dataset = dataset.filter(lambda x, y: filter_zero_patches(
                y, params.data_format, params.dimension_mode, params.filter_zero))
        # shuffle a set number of exampes
        dataset = dataset.shuffle(buffer_size=params.shuffle_size)
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)
        # prefetch with experimental autotune
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        # repeat dataset infinitely so that dataset doesn't exhaust prematurely during fit
        dataset = dataset.repeat()

    # eval mode - uses patches and batches but no patch filtering, data augmentation, or shuffling, works on eval_dirs
    elif mode == 'eval':
        # variable setup
        data_dirs = tf.constant(eval_dirs)
        data_chan = len(params.data_prefix)
        weighted = False if isinstance(params.mask_weights, np.bool) and not params.mask_weights else True
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.mask_prefix,
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          False, # no data augmentation for eval mode
                          params.label_interp,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode,
                          params.mask_weights]  # param makes loader return weights as last channel in labels data]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_roi_multicon_and_labels_3d,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches
        dataset = dataset.map(
            lambda x, y: tf_patches_3d(x, y, params.train_dims, params.data_format, data_chan,
                                       weighted=weighted,
                                       overlap=params.train_patch_overlap),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)
        # prefetch with experimental autotune
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # infer mode - does not use patches (patch function uses infer_dims), batches, or shuffling - works on infer_dir
    elif mode == 'infer':
        if not infer_dir:
            assert ValueError("Must specify inference directory for inference mode")
        dirs = tf.constant(infer_dir)
        # define dims of inference
        data_dims = list(params.infer_dims)
        chan_size = len(params.data_prefix)
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix, params.data_format, params.data_plane, params.norm_data,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_multicon_preserve_size_3d,
                                        [x] + py_func_params,
                                        tf.float32),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches based on infer inputs
        dataset = dataset.map(
            lambda x: tf_patches_3d_infer(x, data_dims, chan_size, params.data_format, params.infer_patch_overlap),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # flat map so that each tensor is a single slice
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        # generate a batch of data
        dataset = dataset.batch(batch_size=1, drop_remainder=True)
        # automatic prefetching to improve efficiency
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # error if not train, eval, or infer
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    return dataset


# utility function to get all study subdirectories in a given parent data directory
# returns shuffled directory list using user defined randomization seed
def get_study_dirs(params):
    # get all subdirectories in data_dir
    study_dirs = [item for item in glob(params.data_dir + '/*/') if os.path.isdir(item)]
    # make sure all necessary files are present in each folder
    study_dirs = [study for study in study_dirs if all(
        [glob('{}/*{}.nii.gz'.format(study, item)) and os.path.isfile(glob('{}/*{}.nii.gz'.format(study, item))[0])
         for item in params.data_prefix + params.label_prefix])]
    # study dirs sorted in alphabetical order for reproducible results
    study_dirs.sort()
    # randomly shuffle input directories for training using a user defined randomization seed
    random.Random(params.random_state).shuffle(study_dirs)

    return  study_dirs


# split list of all valid study directories into a train and test batch based on train fraction
def train_test_split(study_dirs, params):
    # first train fraction is train dirs, last 1-train fract is test dirs
    # assumes study dirs is already shuffled and/or stratified as wanted
    train_dirs = study_dirs[0:int(round(params.train_fract * len(study_dirs)))]
    eval_dirs = study_dirs[int(round(params.train_fract * len(study_dirs))):]

    return train_dirs, eval_dirs


# patch input function for 2d or 3d
def patch_input_fn(params, mode, infer_dir=None):
    # set global random seed for tensorflow
    tf.random.set_seed(params.random_state)
    # Study dirs and prefixes setup
    study_dirs_filepath = os.path.join(params.model_dir, 'study_dirs_list.json')
    # load study dirs file if it already exists for consistent training
    if os.path.isfile(study_dirs_filepath):
        if mode == 'train':
            logging.info("Loading existing study directories file for training: {}".format(study_dirs_filepath))
        with open(study_dirs_filepath) as f:
            study_dirs = json.load(f)
    # if study dirs file does not exist, then create it
    else:
        logging.info("Determining train/test split based on params and available study directories in data directory")
        # get all valid subdirectories in data_dir
        study_dirs = get_study_dirs(params)
        # save directory list to json file so it can be loaded in future
        with open(study_dirs_filepath, 'w+', encoding='utf-8') as f:
            json.dump(study_dirs, f, ensure_ascii=False, indent=4)  # save study dir list for consistency

    # split study directories into train and test sets
    train_dirs, eval_dirs = train_test_split(study_dirs, params)

    # handle infer dir argument
    if infer_dir:
        infer_dir = tf.constant([infer_dir])

    # handle 2D vs 3D
    if params.dimension_mode == '2D':  # handle 2d inputs
        return _patch_input_fn_2d(params, mode, train_dirs, eval_dirs, infer_dir)
    elif params.dimension_mode in ['2.5D', '3D']:  # handle 3d inputs
        return _patch_input_fn_3d(params, mode, train_dirs, eval_dirs, infer_dir)
    else:
        raise ValueError("Training dimensions mode not understood: " + str(params.dimension_mode))
