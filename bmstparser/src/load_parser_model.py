import tensorflow as tf
import utils
import random
import numpy as np
from tensorflow.keras.layers import Embedding
import tensorflow.keras.activations as F
import utils_tf
import decoder_tf


def get_embeddings_input(entry):
    c = float(words.get(entry.norm, 0))
    vocab = {word: ind + 3 for word, ind in enum_word.items()}
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


def get_bi_lstm_output():
    bi_lstm_input = get_bi_lstm_input(sentence_embeddings)

    block_lstm_for_1 = get_lstm_output(bi_lstm_input[0], weights_first_lstm)
    block_lstm_back_1 = get_lstm_output(bi_lstm_input[1], weights_first_lstm)
    res_shape = [sample_size, time_steps, lstm_dims]
    res_for_1 = tf.reshape(block_lstm_for_1.h, shape=res_shape)
    res_back_1 = tf.reshape(block_lstm_back_1.h, shape=res_shape)

    vec_shape = [time_steps, sample_size, input_size_next_lstm]
    vec_cat = concatenate_layers(res_for_1[0], res_back_1[0], time_steps)
    vec_for_2 = tf.reshape(tf.concat(vec_cat, 0), shape=vec_shape)
    vec_back_2 = tf.reshape(tf.concat(list(reversed(vec_cat)), 0), shape=vec_shape)

    block_lstm_for_2 = get_lstm_output(vec_for_2, weights_next_lstm)
    block_lstm_back_2 = get_lstm_output(vec_back_2, weights_next_lstm)
    res_for_2 = tf.reshape(block_lstm_for_2.h, shape=res_shape)
    res_back_2 = tf.reshape(block_lstm_back_2.h, shape=res_shape)

    return [res_for_2, res_back_2]


def concatenate_layers(array1, array2, num_vec):
    concat_size = array1.shape[1] + array2.shape[1]
    concat_result = [tf.reshape(tf.concat([array1[i], array2[num_vec - i - 1]], 0), shape=(1, concat_size))
                     for i in range(num_vec)]
    return concat_result


def evaluate(sentence_lstms):

    def transform_tensor(tensor_list):
        return list(map(lambda tensor: tensor[0, 0], tensor_list))

    head_vector = []
    for index in range(len(sentence_lstms)):
        lstms_0 = sentence_lstms[index][0]
        lstms_1 = sentence_lstms[index][1]
        concatenated_lstm = tf.reshape(utils_tf.concatenate_tensors([lstms_0, lstms_1]), shape=(1, -1))
        head_fov = tf.matmul(concatenated_lstm, hid_layer_foh)
        mod_fov = tf.matmul(concatenated_lstm, hid_layer_fom)
        head_vector.append([head_fov, mod_fov])

    exprs = [[getExpr(head_vector, i, j) for j in range(len(head_vector))] for i in range(len(head_vector))]
    output_tensor = [[output for output in exprsRow] for exprsRow in exprs]
    scores = tf.stack([transform_tensor(output) for output in output_tensor])

    return scores


def getExpr(head_vector, i, j):
    head_fov = head_vector[i][0]
    mod_fov = head_vector[j][1]
    activation_result = F.tanh(head_fov + mod_fov + hid_bias)
    output = tf.matmul(activation_result, out_layer) + out_bias

    return output


if __name__ == '__main__':

    test_file = "/home/dburbano/IdeaProjects/JSL/bist-parser-tensorflow/corpus/en-tiny-ud-train.conllu"
    # Prediction
    print(f'Testing with file {test_file}')
    # Added to run from IntelliJ
    print('Preparing vocabulary table')
    words, enum_word, pos, rels, onto, cpos = list(utils.vocab(test_file))
    # TODO: Check if pickle serialization is required
    print('Finished collecting vocabulary')

    model_path = "/home/dburbano/IdeaProjects/JSL/bist-parser-tensorflow/model-tiny-tf/neuralfirstorder.model3"
    # loaded = tf.saved_model.load(model_path)
    # tf.keras.models.save_model
    loaded_keras = tf.keras.models.load_model(model_path)
    # embeddings_module = loaded.embeddings
    # print(list(loaded_keras.signatures.keys()))
    infer = loaded_keras.signatures["serving_default"]
    # print(infer.structured_outputs)

    sample_size = 1
    # LSTM Layers
    lstm_dims = 126
    w_first_lstm = tf.Variable(infer.trainable_variables[0])
    wig_first_lstm = tf.Variable(infer.trainable_variables[1])
    wfg_first_lstm = tf.Variable(infer.trainable_variables[2])
    wog_first_lstm = tf.Variable(infer.trainable_variables[3])
    weights_first_lstm = [w_first_lstm, wig_first_lstm, wfg_first_lstm, wog_first_lstm]

    input_size_next_lstm = 252
    w_next_lstm = tf.Variable(infer.trainable_variables[4])
    wig_next_lstm = tf.Variable(infer.trainable_variables[5])
    wfg_next_lstm = tf.Variable(infer.trainable_variables[6])
    wog_next_lstm = tf.Variable(infer.trainable_variables[7])
    weights_next_lstm = [w_next_lstm, wig_next_lstm, wfg_next_lstm, wog_next_lstm]

    # Concat Layers
    hid_layer_foh = infer.trainable_variables[8]
    hid_layer_fom = infer.trainable_variables[9]
    hid_bias = infer.trainable_variables[10]
    out_layer = infer.trainable_variables[11]
    out_bias = infer.trainable_variables[12]

    vocab_size = len(words) + 3
    embeddings_dims = 100
    w_lookup = Embedding(vocab_size, embeddings_dims, name='embedding_vocab',
                         embeddings_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1.0))

    with open(test_file, 'r') as conllFP:
        for iSentence, sentence in enumerate(utils.read_conll(conllFP)):
            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

            sentence_embeddings = []
            for entry in conll_sentence:
                embeddings_input = get_embeddings_input(entry)
                word_vec = w_lookup(embeddings_input)
                sentence_embeddings.append(word_vec)

            time_steps = len(sentence_embeddings)
            bi_lstms_output = get_bi_lstm_output()

            res_for_2 = tf.reshape(bi_lstms_output[0], shape=(time_steps, lstm_dims))
            res_back_2 = tf.reshape(bi_lstms_output[1], shape=(time_steps, lstm_dims))
            concat_input = []
            for i in range(time_steps):
                lstms_0 = res_for_2[i]
                lstms_1 = res_back_2[time_steps - i - 1]
                concat_input.append([lstms_0, lstms_1])

            predicted_scores = evaluate(concat_input)
            predicted_heads = decoder_tf.parse_proj(predicted_scores)

            print(predicted_heads)
            # for entry, head in zip(sentence, heads):
            #     entry.pred_parent_id = head
            #     entry.pred_relation = '_'


