import tensorflow as tf
from utilities.losses import loss_picker
import logging
import os
import shutil

# callbacks
"""
on_epoch_end: logs include `acc` and `loss`, and
    optionally include `val_loss`
    (if validation is enabled in `fit`), and `val_acc`
    (if validation and accuracy monitoring are enabled).
on_batch_begin: logs include `size`,
    the number of samples in the current batch.
on_batch_end: logs include `loss`, and optionally `acc`
    (if accuracy monitoring is enabled).
"""

"""
TODO:
implement loss decay
implement other custom callbacks like tensorboard
"""

# custom training function
def custom_training(train_dataset, eval_dataset, completed_epochs, model, params):

    # get loss and optimizer
    loss = loss_picker(params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate[0])  # hardcode constant loss for now

    # Prepare the metrics
    epoch_loss_avg = tf.keras.metrics.Mean()
    train_acc_metric = tf.keras.metrics.MeanSquaredError()
    val_loss_avg = tf.keras.metrics.Mean()
    val_acc_metric = tf.keras.metrics.MeanSquaredError()
    val_acc_best = float('inf')

    # set up logs
    logs = {'loss': None,
            'acc': None,
            'val_loss': None,
            'val_acc': None,
            'size': params.batch_size,
            'batch': None}

    # start of training callbacks
    #for callback in callbacks:
    #    callback.on_train_begin(logs=logs)

    # epoch loop
    epochs_todo = params.num_epochs-completed_epochs
    logging.info("Starting training for {} of {} total epochs".format(epochs_todo, params.num_epochs))
    for epoch in [el + 1 for el in list(range(epochs_todo))]:
        logging.info('Start of epoch {:d}'.format(epoch))

        # epoch loop

        # start of epoch callbacks
        #for callback in callbacks:
        #    callback.on_epoch_begin(epoch + completed_epochs, logs=logs)

        # iterate through dataset until exhausted?? git rid of repeat????
        # Iterate over the batches of the dataset.
        for batch, (x, y, weights) in enumerate(train_dataset):

            # start of batch callbacks
            #for callback in callbacks:
            #    callback.on_train_batch_begin(batch, logs=logs)

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                y_pred = model(x, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss(y, y_pred, sample_weight=weights)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one batch of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # # Update training metrics and log
            train_acc_metric.update_state(y, y_pred)
            epoch_loss_avg.update_state(loss_value)
            # logs["acc"] = train_acc_metric.result()
            # logs["loss"] = epoch_loss_avg.result()

            # Log every n batches
            if batch % 1 == 0:
                logging.info("Epoch {:d}/{:d}, Batch {:06d}, Batch Loss {:06e}, Epoch Loss {:06e}"
                             .format(epoch, epochs_todo, batch, loss_value, epoch_loss_avg.result()))

            # end of batch callbacks
            #for callback in callbacks:
            #    callback.on_train_batch_end(batch, logs=logs)

        # save weights after epoch
        ckpt = os.path.join(params.model_dir, 'checkpoints/epoch_{:d}.hdf5'.format(epoch))
        model.save(ckpt)

        # Run a validation loop at the end of each epoch.
        logging.info("Running validation after epoch {:d}".format(epoch))
        for x_val, y_val in eval_dataset:
            y_val_pred = model(x_val)
            # Update val metrics
            val_loss_avg.update_state(y_val, y_val_pred)
            val_acc_metric.update_state(y_val, y_val_pred)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        val_loss = val_loss_avg.result()
        val_loss_avg.reset_states()
        logs["val_loss"] = val_loss
        logs["val_acc"] = val_acc
        if val_acc < val_acc_best:
            logging.info("Validation loss for epoch {:d} improved from {:06e} to {:06e}"
                         .format(epoch, val_acc_best, val_acc))
            best_ckpt = os.path.join(params.model_dir, "checkpoints/epoch_{:d}_valloss_{:06e}.hdf5"
                                     .format(epoch, val_acc))
            logging.info("New best checkpoint {}".format(best_ckpt))
            shutil.copy(ckpt, best_ckpt)
            val_acc_best = val_acc
        else:
            logging.info("Validation loss for epoch {:d} ({:06e}) did not improve from {:06e}"
                         .format(epoch, val_acc, val_acc_best))

        # end of epoch callbacks
        #for callback in callbacks:
        #    callback.on_epoch_end(epoch + completed_epochs, logs=logs)

    # end of training callbacks
    #for callback in callbacks:
    #    callback.on_train_end(logs=logs)