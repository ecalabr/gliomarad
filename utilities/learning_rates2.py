from tensorflow.keras.optimizers.schedules import ExponentialDecay


# constant (no decay)
def constant(learn_rate):
    if isinstance(learn_rate, (list, tuple)):
        learn_rate = learn_rate[0]
    # simple function that always returns learn_rate regardless of epoch
    def constant_lr(_epoch):
        return learn_rate
    return constant_lr


# exponential decay
def exponential(learn_rate):
    if not isinstance(learn_rate, (list, tuple)):
        raise ValueError("Exponential decay requres three values: starting learning rate, steps, and decay factor")
    start_lr = learn_rate[0]
    steps = learn_rate[1]
    decay = learn_rate[2]
    learning_rate_sced = ExponentialDecay(start_lr, steps, decay, staircase=True)
    return learning_rate_sced


def learning_rate_picker(init_learn_rate, decay_method):

    # sanity checks
    if not isinstance(init_learn_rate, (float, list, tuple)):
        raise ValueError("Learning rate must be a float or list/tuple")
    if not isinstance(decay_method, str):
        raise ValueError("Learning rate decay parameter must be a string")

    # check for specified loss method and error if not found
    if decay_method in globals():
        learning_rate_sched = globals()[decay_method](init_learn_rate)
    else:
        raise NotImplementedError("Specified learning rate decay method is not implemented in learning_rates.py: "
                                  + decay_method)

    return learning_rate_sched
