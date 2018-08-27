from net_builder import *


def model_fn(inputs, params, mode):
    """
    The tensorflow model function.
    :param inputs: (dict) contains the inputs of the graph (features, labels...)
    :param params: (class: Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    :param mode: (str) whether or not the model is training, evaluating etc ('train', 'eval')
    :return: model_spec - the model function for tensorflow training/evaluation
    """

    # separate labels and features
    labels = inputs["labels"]
    features = inputs["features"]

    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        # generate the model and compute the output predictions
        predictions = net_builder(features, params, (mode == 'train'))

    # Define loss and accuracy (we need to apply a mask to account for padding)
    losses = tf.losses.mean_squared_error(labels, predictions)
    mask = labels > 0
    losses = tf.boolean_mask(losses, mask)
    loss = tf.reduce_mean(losses)
    error = tf.reduce_mean(tf.cast(tf.metrics.mean_absolute_error(labels=labels, predictions=predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = None

    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'mean_error': tf.metrics.mean_absolute_error(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('error', error)

    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = {
        'features': features,
        'labels': labels
    }
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['error'] = error
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if mode == 'train':
        model_spec['train_op'] = train_op

    return model_spec