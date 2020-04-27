"""Tensorflow utility functions for evaluation"""

import logging
import os
import tensorflow as tf
from model.utils import save_dict_to_json


def evaluate_sess(sess, model_spec, writer=None):
    """Train the model on `num_steps` batches.
    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.compat.v1.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # compute metrics over the dataset
    n = 1
    while True:
        try:
            sess.run(update_metrics)
            metrics_val = sess.run({k: v[0] for k, v in eval_metrics.items()})
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
            logging.info("Batch = " + str(n).zfill(6) + " " + metrics_string)
            n = n+1
        except tf.errors.OutOfRangeError:
            break

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)

    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)

    return metrics_val


def evaluate(model_spec, model_dir, restore_from):
    """Evaluate the model
    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        restore_from: (string) directory or file containing weights to restore the graph
    """

    # Initialize tf.Saver
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        # Reload weights from the weights subdirectory
        logging.info("Loading weights from model directory...")
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)
        logging.info("- done.")

        # Evaluate
        metrics = evaluate_sess(sess, model_spec)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
        save_dict_to_json(metrics, save_path)
