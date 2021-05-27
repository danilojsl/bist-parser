import tensorflow as tf
import utils
import os
import mstlstm_predict_tf


def compute_accuracy():
    utils.write_conll(output_file, conll_document)
    print("Computing accuracy...")
    utils_path = os.getcwd() + '/utils/'
    python_command = 'python3 ' + utils_path + 'evaluation_script/conll17_ud_eval.py -v -w ' + utils_path + \
                     'evaluation_script/weights.clas ' + test_file + ' ' + output_file + ' > ' + output_file + '.txt'
    print(python_command)
    os.system(python_command)
    with open(output_file + '.txt', 'r') as f:
        for l in f:
            if l.startswith('UAS'):
                print('UAS:%s' % l.strip().split()[-1])
            elif l.startswith('LAS'):
                print('LAS:%s' % l.strip().split()[-1])


if __name__ == '__main__':

    # model_path = f"../../model-light-small-tf/dp-parser.model/"
    model_path = "/home/danilo/JSL/models/dependency_parser_dl/model-small-tf/dp-parser.model1/"
    output_file = "/home/danilo/JSL/models/dependency_parser_dl/model-small-tf/test.conllu"
    print(f"Predicting model {model_path}")
    loaded_bi_lstm = tf.saved_model.load(model_path + "BiLSTM")
    loaded_heads = tf.saved_model.load(model_path + "Heads")
    loaded_relations = tf.saved_model.load(model_path + "Relations")

    # Hardcoded parameters
    embeddings_dims = 100

    # LSTM Layers
    w_first_lstm = tf.Variable(loaded_bi_lstm.blockLstm.weight_matrix)
    wig_first_lstm = tf.Variable(loaded_bi_lstm.blockLstm.weight_input_gate)
    wfg_first_lstm = tf.Variable(loaded_bi_lstm.blockLstm.weight_forget_gate)
    wog_first_lstm = tf.Variable(loaded_bi_lstm.blockLstm.weight_output_gate)

    weights_first_lstm = [w_first_lstm, wig_first_lstm, wfg_first_lstm, wog_first_lstm]

    w_next_lstm = tf.Variable(loaded_bi_lstm.nextBlockLstm.weight_matrix)
    wig_next_lstm = tf.Variable(loaded_bi_lstm.nextBlockLstm.weight_input_gate)
    wfg_next_lstm = tf.Variable(loaded_bi_lstm.nextBlockLstm.weight_forget_gate)
    wog_next_lstm = tf.Variable(loaded_bi_lstm.nextBlockLstm.weight_output_gate)
    weights_next_lstm = [w_next_lstm, wig_next_lstm, wfg_next_lstm, wog_next_lstm]

    weights_bi_lstm = [weights_first_lstm, weights_next_lstm]

    # Heads Layer
    hid_layer_foh = loaded_heads.hidLayerFOH
    hid_layer_fom = loaded_heads.hidLayerFOM
    hid_bias = loaded_heads.hidBias
    out_layer = loaded_heads.outLayer
    out_bias = loaded_heads.outBias
    heads_variables = [hid_layer_foh, hid_layer_fom, hid_bias, out_layer, out_bias]

    # Relations Layer
    r_hid_layer_foh = loaded_relations.rhidLayerFOH
    r_hid_layer_fom = loaded_relations.rhidLayerFOM
    r_hid_bias = loaded_relations.rhidBias
    r_out_layer = loaded_relations.routLayer
    r_out_bias = loaded_relations.routBias
    relations_vocabulary = loaded_relations.relations_vocabulary
    relations_variables = [r_hid_layer_foh, r_hid_layer_fom, r_hid_bias, r_out_layer, r_out_bias,
                           relations_vocabulary]

    # test_file = "../../corpus/en-ud-test.conllu"
    test_file = "../../corpus/en-ud-debug.conllu"
    # Prediction
    print(f'Testing with file {test_file}')

    conll_document = mstlstm_predict_tf.predict(test_file, embeddings_dims, weights_bi_lstm,
                                                heads_variables, relations_variables)

    compute_accuracy()
