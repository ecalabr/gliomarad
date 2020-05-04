from random import shuffle
from utilities.input_fn_util import *


# COMPLETE 2D INPUT FUNCTIONS
def patch_input_fn(params, mode, infer_dir=None):
    # Study dirs and prefixes setup
    study_dirs_filepath = os.path.join(params.model_dir, 'study_dirs_list.npy')
    if os.path.isfile(study_dirs_filepath):  # load study dirs file if it already exists for consistent training
        study_dirs = list(np.load(study_dirs_filepath))
    else:
        study_dirs = glob(params.data_dir + '/*/')
        study_dirs.sort()  # study dirs sorted in alphabetical order
        shuffle(study_dirs)  # randomly shuffle input directories for training
        np.save(study_dirs_filepath, study_dirs)  # save study dir list for later use to ensure consistency
    train_dirs = tf.constant(study_dirs[0:int(round(params.train_fract * len(study_dirs)))])
    eval_dirs = tf.constant(study_dirs[int(round(params.train_fract * len(study_dirs))):])

    # handle infer dir argument
    if infer_dir:
        infer_dir = tf.constant([infer_dir])

    # generate input dataset objects for the different training modes
    # train mode
    if mode == 'train':
        data_dirs = train_dirs
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
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.compat.v1.py_func(load_roi_multicon_and_labels,
                                           [x] + py_func_params,
                                           (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)
        # map each dataset to a series of patches
        dataset = dataset.map(
            lambda x, y: tf_patches(x, y, params.train_dims, len(params.data_prefix), params.data_format,
                                     overlap=params.train_patch_overlap),
            num_parallel_calls=params.num_threads)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # filter out zero patches
        dataset = dataset.filter(lambda x, y: filter_zero_patches(
            y, params.data_format, params.dimension_mode, params.filter_zero))
        # shuffle a set number of exampes
        dataset = dataset.shuffle(buffer_size=params.shuffle_size)
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)

    # eval mode
    elif mode == 'eval':
        data_dirs = eval_dirs
        # if there are no eval dirs return None to model
        if not any(data_dirs):
            return None
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.data_format,
                          params.train_dims,
                          params.data_plane,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_multicon_and_labels,
                                           [x] + py_func_params,
                                           (tf.float32, tf.float32)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches based on infer inputs
        dataset = dataset.map(
            lambda x, y: tf_patches(x, y, params.train_dims, len(params.data_prefix), params.data_format,
                                     overlap=params.infer_patch_overlap),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # flat map so that each tensor is a single slice
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # generate a batch of data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)
        # automatic prefetching to improve efficiency
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # infer mode
    elif mode == 'infer':
        if not infer_dir:
            assert ValueError("Must specify inference directory")
        # define dims of inference
        data_dims = list(params.infer_dims)
        chan_size = len(params.data_prefix)
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix, params.data_format, params.data_plane, params.norm_data, params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(infer_dir)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_multicon_preserve_size,
                                           [x] + py_func_params,
                                           tf.float32), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches based on infer inputs
        dataset = dataset.map(
            lambda x: tf_patches_infer(x, data_dims, chan_size, params.data_format, params.infer_patch_overlap),
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


def patch_input_fn_3d(mode, params, infer_dir=None):
    # Study dirs and prefixes setup
    study_dirs_filepath = os.path.join(params.model_dir, 'study_dirs_list.npy')
    if os.path.isfile(study_dirs_filepath):  # load study dirs file if it already exists for consistent training
        study_dirs = list(np.load(study_dirs_filepath))
    else:
        study_dirs = glob(params.data_dir + '/*/')
        study_dirs.sort()  # study dirs sorted in alphabetical order
        shuffle(study_dirs)  # randomly shuffle input directories for training
        np.save(study_dirs_filepath, study_dirs)  # save study dir list for later use to ensure consistency
    train_dirs = tf.constant(study_dirs[0:int(round(params.train_fract * len(study_dirs)))])
    eval_dirs = tf.constant(study_dirs[int(round(params.train_fract * len(study_dirs))):])

    # handle infer dir argument
    if infer_dir:
        infer_dir = tf.constant([infer_dir])

    # generate input dataset objects for the different training modes
    # train mode
    if mode == 'train':
        data_dirs = train_dirs
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
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.compat.v1.py_func(load_roi_multicon_and_labels_3d,
                                           [x] + py_func_params,
                                           (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)
        # map each dataset to a series of patches
        dataset = dataset.map(
            lambda x, y: tf_patches_3d(x, y, params.train_dims, len(params.data_prefix), params.data_format,
                                    overlap=params.train_patch_overlap),
            num_parallel_calls=params.num_threads)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # filter out zero patches
        dataset = dataset.filter(lambda x, y: filter_zero_patches(
            y, params.data_format, params.dimension_mode, params.filter_zero))
        # shuffle a set number of exampes
        dataset = dataset.shuffle(buffer_size=params.shuffle_size)
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)

    # eval mode
    elif mode == 'eval':
        data_dirs = eval_dirs
        # if there are no eval dirs return None to model
        if not any(data_dirs):
            return None
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.data_format,
                          params.data_plane,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_multicon_and_labels_3d,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches based on infer inputs
        dataset = dataset.map(
            lambda x, y: tf_patches_3d(x, y, params.train_dims, len(params.data_prefix), params.data_format,
                                    overlap=params.infer_patch_overlap),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # flat map so that each tensor is a single slice
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # generate a batch of data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)
        # automatic prefetching to improve efficiency
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # infer mode
    elif mode == 'infer':
        if not infer_dir:
            assert ValueError("Must specify inference directory")
        # define dims of inference
        data_dims = list(params.infer_dims)
        chan_size = len(params.data_prefix)
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix, params.data_format, params.data_plane, params.norm_data,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(infer_dir)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_multicon_preserve_size_3d,
                                        [x] + py_func_params,
                                        tf.float32), num_parallel_calls=tf.data.experimental.AUTOTUNE)
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