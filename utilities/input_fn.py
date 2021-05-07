from utilities.input_fn_util import *
import tensorflow as tf


# get list of globals
start_globals = list(globals().keys())

# DEFINE INPUT FUNCTIONS
# COMPLETE 2D INPUT FUNCTIONS
def patch_input_fn_2d(params, mode, train_dirs, eval_dirs, infer_dir=None):
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
                          False,  # do not do data augmentation on eval data
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
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches - USE SAME DATA DIMS AS TRAINING
        dataset = dataset.map(
            lambda x, y: tf_patches(x, y, params.train_dims, data_chan, params.data_format,
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
                                        tf.float32),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches based on infer inputs
        dataset = dataset.map(
            lambda x: tf_patches_infer(x, data_dims, chan_size, params.data_format, params.infer_patch_overlap),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # flat map so that each tensor is a single slice
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        # generate a batch of data
        dataset = dataset.batch(batch_size=1,
                                drop_remainder=False)  # force batch size 1 to ensure all data is processed
        # automatic prefetching to improve efficiency
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # error if not train, eval, or infer
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    return dataset


def patch_input_fn_3d(params, mode, train_dirs, eval_dirs, infer_dir=None):
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
                          False,  # no data augmentation for eval mode
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


def scaled_cube_input_fn_3d(params, mode, train_dirs, eval_dirs, infer_dir=None):
    # generate input dataset objects for the different training modes

    # train mode - uses patches, patch filtering, batching, data augmentation, and shuffling - works on train_dirs
    if mode == 'train':
        # variable setup
        data_dirs = tf.constant(train_dirs)
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
                          params.mask_weights,  # param makes loader return weights as last channel in labels data
                          params.train_dims[0]  # the first value in train_dims is assumed as scaled cube shape
                          ]
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
        # filter out zero patches
        if params.filter_zero > 0.:
            dataset = dataset.filter(lambda x, y: filter_zero_patches(
                y, params.data_format, params.dimension_mode, params.filter_zero))
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
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
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.mask_prefix,
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          False,  # no data augmentation for eval mode
                          params.label_interp,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode,
                          params.mask_weights]  # param makes loader return weights as last channel in labels data]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x, y: tf.numpy_function(load_roi_multicon_and_labels_3d,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32)),
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
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix, params.data_format, params.data_plane, params.norm_data,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(infer_dir)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_multicon_preserve_size_3d,
                                        [x] + py_func_params,
                                        tf.float32),
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


def scaled_cube_input_fn_3d_csv(params, mode, train_dirs, eval_dirs, infer_dir=None):
    # generate input dataset objects for the different training modes

    # train mode - uses patches, patch filtering, batching, data augmentation, and shuffling - works on train_dirs
    if mode == 'train':
        # variable setup
        data_dirs = tf.constant(train_dirs)
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,  # in this case, label prefix is the full path to the label CSV
                          params.mask_prefix,
                          1,  # this is the colum of he data csv that contains the desired label - include in params?
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          params.augment_train_data,
                          params.label_interp,
                          params.norm_data,
                          params.norm_mode,
                          params.train_dims[0]  # the first value in train_dims is assumed as scaled cube shape
                          ]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # randomly shuffle directory order
        dataset = dataset.shuffle(buffer_size=len(data_dirs))
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_csv_and_roi_multicon_3d,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # map data
        dataset = dataset.map(lambda x, y, z: ((x, y), z))
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
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,  # in this case, label prefix is the full path to the label CSV
                          params.mask_prefix,
                          1,  # this is the colum of he data csv that contains the desired label - include in params?
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          False,  # no data augmentation for eval
                          params.label_interp,
                          params.norm_data,
                          params.norm_mode,
                          params.train_dims[0]  # the first value in train_dims is assumed as scaled cube shape
                          ]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_csv_and_roi_multicon_3d,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # map data
        dataset = dataset.map(lambda x, y, z: ((x, y), z))
        # generate batch data (no shuffling) for eval mode
        dataset = dataset.batch(params.batch_size, drop_remainder=True)
        # prefetch with experimental autotune
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # infer mode - does not use patches (patch function uses infer_dims), batches, or shuffling - works on infer_dir
    elif mode == 'infer':
        if not infer_dir:
            assert ValueError("Must specify inference directory for inference mode")
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix, params.data_format, params.data_plane, params.norm_data,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        infer_dataset = tf.constant(infer_dir)
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,  # in this case, label prefix is the full path to the label CSV
                          params.mask_prefix,
                          1,  # this is the colum of he data csv that contains the desired label - include in params?
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          False,  # no data augmentation for infer mode
                          params.label_interp,
                          params.norm_data,
                          params.norm_mode,
                          params.train_dims[0]  # the first value in train_dims is assumed as scaled cube shape
                          ]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(infer_dataset)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_csv_and_roi_multicon_3d,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # map data discarding label data for inference mode
        dataset = dataset.map(lambda x, y, z: ((x, y), z))  # z is not actually used, but structure is needed for infer
        # generate batch data (no shuffling) for infer mode, batch size of 1 for infer mode
        dataset = dataset.batch(1, drop_remainder=True)
        # prefetch with experimental autotune
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # error if not train, eval, or infer
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    return dataset


# patch input function for 2d or 3d
def get_input_fn(params, mode, infer_dir=None):

    # set global random seed for tensorflow
    tf.random.set_seed(params.random_state)

    # handle inference mode
    if mode == 'infer':
        infer_dir = tf.constant([infer_dir])
        train_dirs = []
        eval_dirs = []

    # handle train and eval modes
    else:
        # get valid study directories
        study_dirs = get_study_dirs(params, mode=mode)

        # split study directories into train and test sets
        train_dirs, eval_dirs = train_test_split(study_dirs, params)

    # handle custom data loader - should all data loaders be accessed this way? i.e. no default to patch loader?
    if not params.custom_data_loader in globals():
        methods = [k for k in globals().keys() if k not in start_globals and k != "get_input_fn"]
        raise NotImplementedError(
            ("Specified custom data loader parameter <{}> is not an available global method. " +
            "This param must be the name of a function specified in input_fn.py. " +
            "Available methods are: \n{}").format(params.custom_data_loader, "\n".join(methods)))
    else:
        return globals()[params.custom_data_loader](params, mode, train_dirs, eval_dirs, infer_dir)
