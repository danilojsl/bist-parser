import random
from operator import itemgetter

import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as F
from tensorflow.keras.layers import LSTM

import decoder_tf
import utils
import utils_tf
from parser_modules_tf import EmbeddingsLookup

sample_size = 1
lstm_dims = 126


def predict(test_file, embeddings_dims, bi_lstms, heads_variables, relations_variables):
    print('Preparing vocabulary table')
    words, enum_word, pos, rels, onto, cpos = list(utils.vocab(test_file))
    print('Finished collecting vocabulary')

    vocab_size = len(words) + 3
    embeddings = EmbeddingsLookup(embeddings_dims, vocab_size, False)
    conll_document = []
    with open(test_file, 'r') as conllFP:
        for iSentence, sentence in enumerate(utils.read_conll(conllFP)):
            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

            sentence_embeddings = []
            for entry in conll_sentence:
                embeddings_input = get_embeddings_input(entry, words, enum_word)
                word_vec = embeddings.lookup(embeddings_input)
                sentence_embeddings.append(word_vec)

            time_steps = len(sentence_embeddings)
            bi_lstms_output = compute_bi_lstm_output(bi_lstms, sentence_embeddings, time_steps)
            res_for_2, res_back_2 = bi_lstms_output[0], bi_lstms_output[1]  # Shape (30, 126)
            sentence_bi_lstm = []
            for i in range(time_steps):
                lstms_0 = tf.slice(res_for_2, begin=[i, 0], size=[1, -1])  # Shape (1, 126)
                lstms_1 = tf.slice(res_back_2, begin=[time_steps - i - 1, 0], size=[1, -1])
                sentence_bi_lstm.append([lstms_0, lstms_1])

            predicted_scores = compute_scores(sentence_bi_lstm, time_steps, heads_variables)
            predicted_heads = decoder_tf.parse_proj(predicted_scores)
            predicted_relations = predict_relations(sentence_bi_lstm, predicted_heads, relations_variables)
            for entry in conll_sentence:
                if entry.id == 0:
                    entry.pred_parent_id, entry.pred_relation = -1, '_'
                else:
                    entry.pred_parent_id = predicted_heads[entry.id]
                    relation = str(predicted_relations[entry.id-1].numpy())
                    entry.pred_relation = relation[2:len(relation)-1]
            conll_document.append(conll_sentence)

    return conll_document


def get_embeddings_input(entry, words, enum_word):
    c = float(words.get(entry.norm, 0))
    vocab = {word: ind + 3 for word, ind in enum_word.items()}
    vocab['*PAD*'] = 1
    vocab['*INITIAL*'] = 2
    dropFlag = (random.random() < (c / (0.25 + c)))
    w_index = np.array(vocab.get(entry.norm, 0)).reshape(1) if dropFlag else np.array(0).reshape(1)

    return w_index


# def compute_bi_lstm_output(embeddings, time_steps, weights_bi_lstm):
#     weights_first_lstm, weights_next_lstm = weights_bi_lstm[0], weights_bi_lstm[1]
#     bi_lstm_input = get_bi_lstm_input(embeddings, time_steps)
#
#     block_lstm_for_1 = get_lstm_output(bi_lstm_input[0], weights_first_lstm, time_steps)
#     block_lstm_back_1 = get_lstm_output(bi_lstm_input[1], weights_first_lstm, time_steps)
#
#     next_lstm_inputs = compute_next_lstm_input([block_lstm_for_1, block_lstm_back_1], time_steps)
#     block_lstm_for_2 = get_lstm_output(next_lstm_inputs[0], weights_next_lstm, time_steps)   # Shape (30, 1, 126)
#     block_lstm_back_2 = get_lstm_output(next_lstm_inputs[1], weights_next_lstm, time_steps)  # Shape (30, 1, 126)
#
#     return [tf.reshape(block_lstm_for_2.h, shape=[time_steps, lstm_dims]),  # Shape (30, 126)
#             tf.reshape(block_lstm_back_2.h, shape=[time_steps, lstm_dims])]  # Shape (30, 126)


def compute_bi_lstm_output(bi_lstms, embeddings, time_steps):
    first_bi_lstm = bi_lstms[0]
    next_bi_lstm = bi_lstms[1]

    bi_lstm_input = get_bi_lstm_input(embeddings, time_steps)
    lstm_for_1 = get_lstm_output(first_bi_lstm[0], bi_lstm_input[0])
    lstm_back_1 = get_lstm_output(first_bi_lstm[1], bi_lstm_input[1])

    next_lstm_inputs = compute_next_lstm_input([lstm_for_1, lstm_back_1], time_steps)
    lstm_for_2 = get_lstm_output(next_bi_lstm[0], next_lstm_inputs[0])   # Shape (30, 1, 126)
    lstm_back_2 = get_lstm_output(next_bi_lstm[1], next_lstm_inputs[1])  # Shape (30, 1, 126)

    return [tf.reshape(lstm_for_2, shape=[time_steps, lstm_dims]),  # Shape (30, 126)
            tf.reshape(lstm_back_2, shape=[time_steps, lstm_dims])]  # Shape (30, 126)


def get_bi_lstm_input(embeddings, time_steps):
    input_size = embeddings[0].shape[1]
    features_for = embeddings
    features_back = list(reversed(embeddings))
    vec_for = tf.reshape(tf.concat(features_for, 0), shape=(sample_size, time_steps, input_size))
    vec_back = tf.reshape(tf.concat(features_back, 0), shape=(sample_size, time_steps, input_size))

    return [vec_for, vec_back]


# def get_lstm_output(input_sequence, weights, time_steps):
#     ini_cell_state = tf.zeros(shape=[sample_size, lstm_dims])
#     ini_hidden_state = tf.zeros(shape=[sample_size, lstm_dims])
#     bias = tf.zeros(shape=[lstm_dims * 4])
#     weight_matrix = weights[0]
#     weight_input_gate, weight_forget_gate, weight_output_gate = weights[1], weights[2], weights[3]
#
#     block_lstm = tf.raw_ops.BlockLSTM(seq_len_max=time_steps, x=input_sequence, cs_prev=ini_cell_state,
#                                       h_prev=ini_hidden_state, w=weight_matrix,
#                                       wci=weight_input_gate, wcf=weight_forget_gate,
#                                       wco=weight_output_gate, b=bias)
#     return block_lstm


def get_lstm_output(lstm_model, input_sequence):
    # lstm = LSTM(lstm_dims, return_sequences=True, return_state=True)
    ini_hid_state = tf.zeros(shape=[1, lstm_dims]), tf.zeros(shape=[1, lstm_dims])
    output = lstm_model(input_sequence, initial_state=ini_hid_state)
    hidden_states, hidden_state, cell_state = output[0], output[1], output[2]
    return hidden_states


def compute_next_lstm_input(lstm_outputs, time_steps):
    res_forward, res_backward = lstm_outputs[0], lstm_outputs[1]  # Shape(30, 1, 126)

    vec_shape = [sample_size, time_steps, -1]
    vec_forward = tf.reshape(tf.concat([res_forward, tf.reverse(res_backward, axis=[1])], 2), shape=vec_shape)
    vec_backward = tf.reverse(vec_forward, axis=[0])

    return [vec_forward, vec_backward]  # Shape (30, 1, 252)


def compute_scores(sentence_lstms, time_steps, heads_variables):
    hid_layer_foh, hid_layer_fom = heads_variables[0], heads_variables[1]
    head_vector = []
    for index in range(len(sentence_lstms)):
        lstms_0 = sentence_lstms[index][0]
        lstms_1 = sentence_lstms[index][1]
        concatenated_lstm = tf.concat([lstms_0, lstms_1], 1)
        head_fov = tf.matmul(concatenated_lstm, hid_layer_foh)
        mod_fov = tf.matmul(concatenated_lstm, hid_layer_fom)
        head_vector.append([head_fov, mod_fov])

    exprs = [[get_expr(head_vector, i, j, heads_variables) for j in range(len(head_vector))] for i in range(len(head_vector))]
    scores = tf.reshape(tf.stack(exprs), shape=(time_steps, time_steps))

    return scores


def get_expr(head_vector, i, j, heads_variables):
    hid_bias, out_layer, out_bias = heads_variables[2], heads_variables[3], heads_variables[4]
    head_fov = head_vector[i][0]
    mod_fov = head_vector[j][1]
    activation_result = F.tanh(head_fov + mod_fov + hid_bias)
    output = tf.matmul(activation_result, out_layer) + out_bias

    return output


def predict_relations(sentence_lstms, heads, relations_variables):
    r_hid_layer_foh, r_hid_layer_fom = relations_variables[0], relations_variables[1]
    r_hid_bias, r_out_layer, r_out_bias = relations_variables[2], relations_variables[3], relations_variables[4]
    relations_vocabulary = relations_variables[5]
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

