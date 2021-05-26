import tensorflow as tf


class EmbeddingsLookup:

    def __init__(self, embeddings_size, vocabulary_size, training=True):
        # TODO: Shape changes between prediction and training
        if training:
            self.shape = tf.constant([embeddings_size, vocabulary_size])
        else:
            self.shape = tf.constant([vocabulary_size, embeddings_size])
        self.seed = tf.constant([1, 1], dtype=tf.int64)
        self.embeddings_table = tf.random.stateless_normal(self.shape, self.seed)

    def lookup(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings_table, inputs)

