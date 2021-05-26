import tensorflow as tf
from parser_modules_tf import EmbeddingsLookup

if __name__ == '__main__':
    embeddings_lookup = EmbeddingsLookup(10, 20)
    embeddings_table = embeddings_lookup.embeddings_table
    print(embeddings_table)
    inputs = tf.constant([2, 7])
    output = embeddings_lookup.lookup(inputs)
    print(output)
