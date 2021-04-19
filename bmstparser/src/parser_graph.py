import os
import shutil
from optparse import OptionParser
from os import path

import tensorflow as tf
from absl import logging

import mstlstm_graph
import utils


def train_parser(bi_lstm_input):
    # Reduce logging output.
    logging.set_verbosity(logging.ERROR)

    # Begins Graph Session
    tf.compat.v1.disable_eager_execution()

    bi_lstm_input_ph = tf.compat.v1.placeholder(tf.float32, shape=None)
    weight_matrix_ph = tf.compat.v1.placeholder(tf.float32, shape=None)
    y_true_ph = tf.compat.v1.placeholder(tf.float32, shape=None)
    y_pred_ph = tf.compat.v1.placeholder(tf.float32, shape=None)

    loss = lambda: tf.reduce_sum(tf.concat([y_true_ph, y_pred_ph], 0))

    with tf.compat.v1.Session() as session:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8)

        output = session.run(fetches={'bi_lstm_input': bi_lstm_input_ph,
                                      'weight_matrix': weight_matrix_ph},
                             feed_dict={bi_lstm_input_ph: bi_lstm_input,
                                        weight_matrix_ph: weight_matrix_np,
                                        y_true_ph: y_true_np,
                                        y_pred_ph: y_pred_np})

        bi_lstm_input = output['bi_lstm_input']
        print(bi_lstm_input.shape)
        ini_cell_state = tf.zeros(shape=[sample_size, lstm_dims], name='c_first_lstm')
        ini_hidden_state = tf.zeros(shape=[sample_size, lstm_dims], name='h_first_lstm')
        bias = tf.zeros(shape=[lstm_dims * 4], name='b_first_lstm')

        weight_matrix = tf.Variable(tf.convert_to_tensor(output['weight_matrix']), name='w_first_lstm')

        initializer = tf.keras.initializers.GlorotUniform()  # Xavier uniform

        values = initializer(shape=[lstm_dims])
        weight_input_gate = tf.Variable(values, name='wig_first_lstm')
        weight_forget_gate = tf.Variable(values, name='wfg_first_lstm')
        weight_output_gate = tf.Variable(values, name='wog_first_lstm')

        vec_for = bi_lstm_input[0]

        block_lstm = tf.raw_ops.BlockLSTM(seq_len_max=time_steps, x=vec_for, cs_prev=ini_cell_state,
                                          h_prev=ini_hidden_state, w=weight_matrix,
                                          wci=weight_input_gate, wcf=weight_forget_gate,
                                          wco=weight_output_gate, b=bias)

        optimizer.minimize(loss=loss, var_list=weight_matrix)
        print("after minimize")
        print(weight_matrix)


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("--outdir", type="string", dest="output", default="/model-tiny-tf")

    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                      default="/corpus/en-tiny-ud-dev.conllu")

    # multi-task has been deleted for bloated code

    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)

    parser.add_option("--epochs", type="int", dest="epochs", default=7)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="lr", default=1e-3)
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=126)

    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="neuralfirstorder.model")

    (options, args) = parser.parse_args()

    # TODO: Check if we can add operation parallelism on CPU with Tensorflow
    # Added to run from IntelliJ
    os.chdir("../../")
    print('Current directory: ' + os.getcwd())
    output_path = os.getcwd() + options.output
    model_name = options.model
    utils_path = os.getcwd() + '/bmstparser/src/utils/'  # 'src/utils/'
    # Added to run from IntelliJ
    if path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    # Training classifier
    print(f'Training with file {options.conll_train}')
    # Added to run from IntelliJ
    train_file = os.getcwd() + options.conll_train
    # Added to run from IntelliJ

    print('Preparing vocabulary table')
    words, enum_word, pos, rels, onto, cpos = list(utils.vocab(train_file))
    print('Finished collecting vocabulary')

    parser = mstlstm_graph.MSTParserLSTM(words, rels, enum_word, options)
    document_features = parser.get_document_embeddings(train_file)
    document_embeddings = document_features[0]
    document_bi_lstm = document_features[1]

    # LSTM Network Layers
    lstm_dims = 126
    sample_size = 1
    input_size = 100
    initializer = tf.keras.initializers.GlorotUniform()  # Xavier uniform
    values = initializer(shape=[input_size + lstm_dims, lstm_dims * 4])
    weight_matrix_tf = tf.Variable(values, name='w_first_lstm')
    weight_matrix_np = weight_matrix_tf.numpy()
    print(weight_matrix_np)

    # Mocking final output
    y_true_tf = tf.constant([0.1, 0.2])
    y_pred_tf = tf.constant([0.1, 0.3])
    result = tf.reduce_sum(tf.concat([y_true_tf, y_pred_tf], 0))
    print(f'result: {result}')
    for sentence_bi_lstm in document_bi_lstm:
        y_true_np = y_true_tf.numpy()
        y_pred_np = y_pred_tf.numpy()
        time_steps = sentence_bi_lstm[0].shape[0]
        train_parser(sentence_bi_lstm)
