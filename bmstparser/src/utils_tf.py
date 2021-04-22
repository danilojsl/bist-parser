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


def concatenate_layers(array1, array2, num_vec):
    concat_size = array1.shape[1] + array2.shape[1]
    concat_result = [tf.reshape(tf.concat([array1[i], array2[num_vec - i - 1]], 0), shape=(1, concat_size))
                     for i in range(num_vec)]
    return concat_result
