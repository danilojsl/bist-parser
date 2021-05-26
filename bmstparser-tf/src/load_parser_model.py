import random
from operator import itemgetter

import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as F

import decoder_tf
import utils
import os
from parser_modules_tf import EmbeddingsLookup


def get_embeddings_input(entry):
    c = float(words.get(entry.norm, 0))
    vocab = {word: ind + 3 for word, ind in enum_word.items()}
    vocab['*PAD*'] = 1
    vocab['*INITIAL*'] = 2
    dropFlag = (random.random() < (c / (0.25 + c)))
    w_index = np.array(vocab.get(entry.norm, 0)).reshape(1) if dropFlag else np.array(0).reshape(1)

    return w_index


def get_bi_lstm_input(embeddings):
    input_size = embeddings[0].shape[1]
    features_for = embeddings
    features_back = list(reversed(embeddings))
    vec_for = tf.reshape(tf.concat(features_for, 0), shape=(time_steps, sample_size, input_size))
    vec_back = tf.reshape(tf.concat(features_back, 0), shape=(time_steps, sample_size, input_size))

    return [vec_for, vec_back]


def get_lstm_output(input_sequence, weights):
    ini_cell_state = tf.zeros(shape=[sample_size, lstm_dims])
    ini_hidden_state = tf.zeros(shape=[sample_size, lstm_dims])
    bias = tf.zeros(shape=[lstm_dims * 4])
    weight_matrix = weights[0]
    weight_input_gate, weight_forget_gate, weight_output_gate = weights[1], weights[2], weights[3]

    block_lstm = tf.raw_ops.BlockLSTM(seq_len_max=time_steps, x=input_sequence, cs_prev=ini_cell_state,
                                      h_prev=ini_hidden_state, w=weight_matrix,
                                      wci=weight_input_gate, wcf=weight_forget_gate,
                                      wco=weight_output_gate, b=bias)
    return block_lstm


def compute_bi_lstm_output(embeddings):
    bi_lstm_input = get_bi_lstm_input(embeddings)

    block_lstm_for_1 = get_lstm_output(bi_lstm_input[0], weights_first_lstm)
    block_lstm_back_1 = get_lstm_output(bi_lstm_input[1], weights_first_lstm)

    next_lstm_inputs = compute_next_lstm_input([block_lstm_for_1, block_lstm_back_1])
    block_lstm_for_2 = get_lstm_output(next_lstm_inputs[0], weights_next_lstm)   # Shape (30, 1, 126)
    block_lstm_back_2 = get_lstm_output(next_lstm_inputs[1], weights_next_lstm)  # Shape (30, 1, 126)

    return [tf.reshape(block_lstm_for_2.h, shape=[time_steps, lstm_dims]),  # Shape (30, 126)
            tf.reshape(block_lstm_back_2.h, shape=[time_steps, lstm_dims])]  # Shape (30, 126)


def compute_next_lstm_input(lstm_outputs):
    block_lstm_for_1, block_lstm_back_1 = lstm_outputs[0], lstm_outputs[1]  # Shape(30, 1, 126)
    res_shape = [sample_size, time_steps, lstm_dims]
    res_forward = tf.reshape(block_lstm_for_1.h, shape=res_shape)  # Shape (1, 30, 126)
    res_backward = tf.reshape(block_lstm_back_1.h, shape=res_shape)  # Shape (1, 30, 126)

    vec_shape = [time_steps, sample_size, -1]
    vec_forward = tf.reshape(tf.concat([res_forward, tf.reverse(res_backward, axis=[1])], 2), shape=vec_shape)
    vec_backward = tf.reverse(vec_forward, axis=[0])

    return [vec_forward, vec_backward]  # Shape (30, 1, 252)


def compute_scores(sentence_lstms):

    head_vector = []
    for index in range(len(sentence_lstms)):
        lstms_0 = sentence_lstms[index][0]
        lstms_1 = sentence_lstms[index][1]
        concatenated_lstm = tf.concat([lstms_0, lstms_1], 1)
        head_fov = tf.matmul(concatenated_lstm, hid_layer_foh)
        mod_fov = tf.matmul(concatenated_lstm, hid_layer_fom)
        head_vector.append([head_fov, mod_fov])

    exprs = [[get_expr(head_vector, i, j) for j in range(len(head_vector))] for i in range(len(head_vector))]
    scores = tf.reshape(tf.stack(exprs), shape=(time_steps, time_steps))

    return scores


def get_expr(head_vector, i, j):
    head_fov = head_vector[i][0]
    mod_fov = head_vector[j][1]
    activation_result = F.tanh(head_fov + mod_fov + hid_bias)
    output = tf.matmul(activation_result, out_layer) + out_bias

    return output


def predict_relations(sentence_lstms, heads):
    predicted_relations = []
    for modifier, head in enumerate(heads[1:]):
        concatenated_lstm_head = tf.concat([sentence_lstms[head][0], sentence_lstms[head][1]], 1)
        concatenated_lstm_mod = tf.concat([sentence_lstms[modifier + 1][0], sentence_lstms[modifier + 1][1]], 1)
        r_head_fov = tf.matmul(concatenated_lstm_head, r_hid_layer_foh)
        r_mod_fov = tf.matmul(concatenated_lstm_mod, r_hid_layer_fom)
        activation_result = F.tanh(r_head_fov + r_mod_fov + r_hid_bias)
        r_scores = tf.matmul(activation_result, r_out_layer) + r_out_bias
        max_value = max(enumerate(r_scores.numpy()[0]), key=itemgetter(1))
        relations = relations_vocabulary[max_value[0]]
        predicted_relations.append(relations)
    return predicted_relations


def compute_accuracy(model_number):
    output_file = f"../../model-light-small-tf/test_pred{model_number}.conllu"
    utils.write_conll(output_file, conll_document)
    print("Computing accuracy...")
    utils_path = '../../bmstparser-pytorch/src/utils/'
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

    for i in range(1, 61):
        model_path = f"../../model-light-small-tf/dp-parser.model{i}/"
        print(f"Predicting model {model_path}")
        loaded_bi_lstm = tf.saved_model.load(model_path + "BiLSTM")
        loaded_heads = tf.saved_model.load(model_path + "Heads")
        loaded_relations = tf.saved_model.load(model_path + "Relations")

        # Hardcoded parameters
        sample_size = 1
        lstm_dims = 126
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

        # Heads Layer
        hid_layer_foh = loaded_heads.hidLayerFOH
        hid_layer_fom = loaded_heads.hidLayerFOM
        hid_bias = loaded_heads.hidBias
        out_layer = loaded_heads.outLayer
        out_bias = loaded_heads.outBias

        # Relations Layer
        r_hid_layer_foh = loaded_relations.rhidLayerFOH
        r_hid_layer_fom = loaded_relations.rhidLayerFOM
        r_hid_bias = loaded_relations.rhidBias
        r_out_layer = loaded_relations.routLayer
        r_out_bias = loaded_relations.routBias
        relations_vocabulary = loaded_relations.relations_vocabulary

        test_file = "/corpus/en-ud-test.conllu"
        # Prediction
        print(f'Testing with file {test_file}')
        # Added to run from IntelliJ
        print('Preparing vocabulary table')
        words, enum_word, pos, rels, onto, cpos = list(utils.vocab(test_file))
        print('Finished collecting vocabulary')

        vocab_size = len(words) + 3
        # tf.random.set_seed(1)
        # w_lookup = Embedding(vocab_size, embeddings_dims, name='embedding_vocab',
        #                      embeddings_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1.0, seed=1))
        # tf.nn.embedding_lookup`
        embeddings = EmbeddingsLookup(embeddings_dims, vocab_size, False)
        conll_document = []
        with open(test_file, 'r') as conllFP:
            for iSentence, sentence in enumerate(utils.read_conll(conllFP)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                sentence_embeddings = []
                for entry in conll_sentence:
                    embeddings_input = get_embeddings_input(entry)
                    word_vec = embeddings.lookup(embeddings_input)
                    sentence_embeddings.append(word_vec)

                # Temp for loading as np array
                # sentence_embeddings = []
                # with open('embedding_debug.npy', 'rb') as file:
                #     sentence_embeddings_np = np.load(file)
                # for x in sentence_embeddings_np:
                #     for word_vec in x:
                #         sentence_embeddings.append(tf.reshape(tf.convert_to_tensor(word_vec), shape=(1, -1)))

                # data = np.asarray(sentence_embeddings_np.reshape(8, 100))
                # np.savetxt('embeddings_s1.csv', data, delimiter=',')

                # Temp

                # Temp for saving as np array
                # sentence_embeddings_np = np.array(sentence_embeddings).reshape((1, 8, 100))
                # with open('embedding_debug.npy', 'wb') as file:
                #     np.save(file, sentence_embeddings_np)
                # Temp

                time_steps = len(sentence_embeddings)
                bi_lstms_output = compute_bi_lstm_output(sentence_embeddings)
                res_for_2, res_back_2 = bi_lstms_output[0], bi_lstms_output[1]  # Shape (30, 126)
                sentence_bi_lstm = []
                for i in range(time_steps):
                    lstms_0 = tf.slice(res_for_2, begin=[i, 0], size=[1, -1])  # Shape (1, 126)
                    lstms_1 = tf.slice(res_back_2, begin=[time_steps - i - 1, 0], size=[1, -1])
                    sentence_bi_lstm.append([lstms_0, lstms_1])

                predicted_scores = compute_scores(sentence_bi_lstm)
                predicted_heads = decoder_tf.parse_proj(predicted_scores)
                predicted_relations = predict_relations(sentence_bi_lstm, predicted_heads)
                # print(f"predicted_heads: {predicted_heads}")
                # print("predicted relations:")
                for entry in conll_sentence:
                    if entry.id == 0:
                        entry.pred_parent_id, entry.pred_relation = -1, '_'
                    else:
                        entry.pred_parent_id = predicted_heads[entry.id]
                        relation = str(predicted_relations[entry.id-1].numpy())
                        entry.pred_relation = relation[2:len(relation)-1]
                conll_document.append(conll_sentence)

        compute_accuracy(i)
