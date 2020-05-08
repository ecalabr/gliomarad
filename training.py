import tensorflow as tf
from model.net_builder import net_builder
from utilities.losses import loss_picker


# custom training function
def custom_training(train_dataset, eval_dataset, params):

    # build model objects
    model = net_builder(params)
    loss = loss_picker(params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate[0])  # implement decay?

    # Prepare the metrics.
    train_acc_metric = tf.keras.metrics.MSE()
    val_acc_metric = tf.keras.metrics.MSE()

    # epoch loop
    for epoch in range(params.num_epochs):
        print('Start of epoch {:d}'.format(epoch))

        # Iterate over the batches of the dataset.
        for step, (x, y, weights) in enumerate(train_dataset):

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

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric(y, y_pred)

            # Log every 200 batches.
            if step % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))

        # Run a validation loop at the end of each epoch.
        for x_val, y_val in eval_dataset:
            y_val_pred = model(x_val)
            # Update val metrics
            val_acc_metric(y_val, y_val_pred)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print('Validation acc: %s' % (float(val_acc),))
