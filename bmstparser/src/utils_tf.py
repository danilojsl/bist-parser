import tensorflow as tf


def Parameter(shape=None, name='param'):
    shape = (1, shape) if type(shape) == int else shape
    initializer = tf.keras.initializers.GlorotUniform()  # Xavier uniform
    values = initializer(shape=shape)
    return tf.Variable(values, name=name, trainable=True)


def concatenate_tensors(arrays):
    valid_l = [x for x in arrays if x is not None]  # This code removes None elements from an array
    dimension = len(valid_l[0].shape) - 1
    return tf.concat(valid_l, dimension)
