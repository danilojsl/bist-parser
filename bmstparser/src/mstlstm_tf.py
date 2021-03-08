import random
import shutil
import time
from operator import itemgetter

import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as F
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

import decoder
import utils
from utils import read_conll


def concatenate_tensors(arrays):
    valid_l = [x for x in arrays if x is not None]  # This code removes None elements from an array
    dimension = len(valid_l[0].shape) - 1
    return tf.concat(valid_l, dimension)


def concatenate_layers(array1, array2, num_vec):
    concat_size = array1.shape[1] + array1.shape[1]
    concat_result = [tf.reshape(tf.concat([array1[i], array2[num_vec - i - 1]], 0), shape=(1, concat_size))
                     for i in range(num_vec)]
    return concat_result


def Parameter(shape=None, name='param'):
    shape = (1, shape) if type(shape) == int else shape
    initializer = tf.keras.initializers.GlorotUniform()  # Xavier uniform
    values = initializer(shape=shape)
    return tf.Variable(values, name=name, trainable=True)


def init_hidden(dim):
    return tf.zeros(shape=[1, dim]), tf.zeros(shape=[1, dim])


def get_optim(opt):
    if opt.optim == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=opt.lr)
    elif opt.optim == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=opt.lr, epsilon=1e-8)


def loss_function(y_true, y_pred):
    l_variable = y_true + y_pred
    return tf.reduce_sum(concatenate_tensors(l_variable))


class EmbeddingsModule(tf.keras.layers.Layer):

    def __init__(self, vocab_size, wdims):
        super(EmbeddingsModule, self).__init__()
        self.wdims = wdims

        self.wlookup = Embedding(vocab_size, self.wdims, name='embedding_vocab',
                                 embeddings_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1.0))

    def call(self, inputs):
        # Forward pass
        word_vec = self.wlookup(inputs) if self.wdims > 0 else None
        return word_vec


class FirstBiLSTMModule(tf.keras.layers.Layer):

    def __init__(self, lstm_dims):
        super(FirstBiLSTMModule, self).__init__()
        # Input data
        self.sample_size = 1  # batch size
        self.lstm_for_1 = LSTM(lstm_dims, return_sequences=True, return_state=True)
        self.lstm_back_1 = LSTM(lstm_dims, return_sequences=True, return_state=True)
        # Initializing hidden and cell states values to 0
        self.hid_for_1, self.hid_back_1 = [init_hidden(lstm_dims) for _ in range(2)]
        self.cat = tf.keras.layers.Concatenate(1)

    def call(self, inputs, num_vec):
        # Forward pass
        vec_for = inputs[0]
        vec_back = inputs[1]
        res_for_1, self.hid_for_1 = self.get_lstm_output(self.lstm_for_1, vec_for, self.hid_for_1)
        res_back_1, self.hid_back_1 = self.get_lstm_output(self.lstm_back_1, vec_back, self.hid_back_1)

        vec_cat = concatenate_layers(res_for_1[0], res_back_1[0], num_vec)
        # vec_cat_v2 = self.cat([res_for_1, res_back_1]) # Test this could assure a forward and backpropagation process

        vec_for_2 = tf.reshape(tf.concat(vec_cat, 0), shape=(self.sample_size, num_vec, -1))
        vec_back_2 = tf.reshape(tf.concat(list(reversed(vec_cat)), 0), shape=(self.sample_size, num_vec, -1))

        return [vec_for_2, vec_back_2]

    @staticmethod
    def get_lstm_output(lstm_model, input_sequence, initial_state):
        output = lstm_model(input_sequence, initial_state=initial_state)
        hidden_states, hidden_state, cell_state = output[0], output[1], output[2]
        return hidden_states, (hidden_state, cell_state)


class NextBiLSTMModule(tf.keras.layers.Layer):

    def __init__(self, lstm_dims):
        super(NextBiLSTMModule, self).__init__()
        # Input data
        self.lstm_for_2 = LSTM(lstm_dims, return_sequences=True, return_state=True)
        self.lstm_back_2 = LSTM(lstm_dims, return_sequences=True, return_state=True)
        # Initializing hidden and cell states values to 0
        self.hid_for_2, self.hid_back_2 = [init_hidden(lstm_dims) for _ in range(2)]

    def call(self, inputs):
        # Forward pass
        vec_for_2 = inputs[0]
        vec_back_2 = inputs[1]
        res_for_2, self.hid_for_2 = self.get_lstm_output(self.lstm_for_2, vec_for_2, self.hid_for_2)
        res_back_2, self.hid_back_2 = self.get_lstm_output(self.lstm_back_2, vec_back_2, self.hid_back_2)

        return [res_for_2, res_back_2]

    @staticmethod
    def get_lstm_output(lstm_model, input_sequence, initial_state):
        output = lstm_model(input_sequence, initial_state=initial_state)
        hidden_states, hidden_state, cell_state = output[0], output[1], output[2]
        return hidden_states, (hidden_state, cell_state)


class BiLSTMModule(tf.keras.layers.Layer):

    def __init__(self, lstm_dims):
        super(BiLSTMModule, self).__init__()
        self.biLstm1 = FirstBiLSTMModule(lstm_dims)
        self.biLstm2 = NextBiLSTMModule(lstm_dims)

    def call(self, inputs, num_vec):
        # Forward pass
        bi_lstm1_output = self.biLstm1(inputs, num_vec)
        bi_lstm2_output = self.biLstm2(bi_lstm1_output)
        return bi_lstm2_output


class ConcatHeadModule(tf.keras.layers.Layer):

    def __init__(self, ldims, hidden_units, hidden2_units):
        super(ConcatHeadModule, self).__init__()
        # Weight Initialization
        hidden_units = hidden_units
        hidden2_units = hidden2_units
        self.hidLayerFOH = Parameter((ldims * 2, hidden_units), 'hidLayerFOH')
        self.hidLayerFOM = Parameter((ldims * 2, hidden_units), 'hidLayerFOM')
        self.hidBias = Parameter(hidden_units, 'hidBias')
        self.catBias = Parameter(hidden_units * 2, 'catBias')

        if hidden2_units:
            self.hid2Layer = Parameter((hidden_units * 2, hidden2_units), 'hid2Layer')
            self.hid2Bias = Parameter(hidden2_units, 'hid2Bias')

        self.outLayer = Parameter((hidden2_units if hidden2_units > 0 else hidden_units, 1), 'outLayer')
        # TODO: Verify if it works with Parameter(1) instead of 0
        self.outBias = Parameter(1)  # 0

    def call(self, inputs):
        # Forward pass
        lstms_i_0 = inputs[0]
        lstms_i_1 = inputs[1]
        lstms_j_0 = inputs[2]
        lstms_j_1 = inputs[3]
        concatenated_lstm = tf.reshape(concatenate_tensors([lstms_i_0, lstms_i_1]), shape=(1, -1))
        headfov_i = tf.matmul(concatenated_lstm, self.hidLayerFOH)
        concatenated_lstm = tf.reshape(concatenate_tensors([lstms_j_0, lstms_j_1]), shape=(1, -1))
        modfov_j = tf.matmul(concatenated_lstm, self.hidLayerFOM)

        if self.hidden2_units > 0:
            concatenated_result = concatenate_tensors([headfov_i, modfov_j])
            activation_result = self.activation(concatenated_result + self.catBias)
            matmul_result = tf.matmul(activation_result, self.hid2Layer)
            next_activation_result = self.activation(self.hid2Bias + matmul_result)
            output = tf.matmul(next_activation_result, self.outLayer) + self.outBias
        else:
            activation_result = self.activation(headfov_i + modfov_j + self.hidBias)
            output = tf.matmul(activation_result, self.outLayer) + self.outBias

        return output


class ConcatRelationModule(tf.keras.layers.Layer):

    def __init__(self, rels, ldims, hidden_units, hidden2_units):
        super(ConcatRelationModule, self).__init__()
        self.rhidLayerFOH = Parameter((2 * ldims, hidden_units), 'rhidLayerFOH')
        self.rhidLayerFOM = Parameter((2 * ldims, hidden_units), 'rhidLayerFOM')
        self.rhidBias = Parameter(hidden_units, 'rhidBias')
        self.rcatBias = Parameter(hidden_units * 2, 'rcatBias')
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.rel_list = rels

        if hidden2_units:
            self.rhid2Layer = Parameter((hidden_units * 2, hidden2_units), 'rhid2Layer')
            self.rhid2Bias = Parameter(hidden2_units, 'rhid2Bias')

        self.routLayer = Parameter((hidden2_units if hidden2_units > 0 else hidden_units, len(self.rel_list)),
                                   'routLayer')
        self.routBias = Parameter((len(self.rel_list)), 'routBias')

    def call(self, inputs):
        # Forwad pass
        lstms_i_0 = inputs[0]
        lstms_i_1 = inputs[1]
        lstms_j_0 = inputs[2]
        lstms_j_1 = inputs[3]

        concatenated_lstm = tf.reshape(concatenate_tensors([lstms_i_0, lstms_i_1]), shape=(1, -1))
        rheadfov_i = tf.matmul(concatenated_lstm, self.rhidLayerFOH)
        concatenated_lstm = tf.reshape(concatenate_tensors([lstms_j_0, lstms_j_1]), shape=(1, -1))
        rmodfov_j = tf.matmul(concatenated_lstm, self.rhidLayerFOM)

        if self.hidden2_units > 0:
            concatenated_result = concatenate_tensors([rheadfov_i, rmodfov_j])
            activation_result = self.activation(concatenated_result + self.rcatBias)
            matmul_result = tf.matmul(activation_result, self.rhid2Layer)
            next_activation_result = self.activation(self.rhid2Bias + matmul_result)
            output = tf.matmul(next_activation_result, self.routLayer) + self.routBias
        else:
            activation_result = self.activation(rheadfov_i + rmodfov_j + self.rhidBias)
            output = tf.matmul(activation_result, self.routLayer) + self.routBias

        return output


class MSTParserLSTMModel(tf.keras.Model):

    def __init__(self, vocab, rels, enum_word, options):
        super(MSTParserLSTMModel, self).__init__()
        random.seed(1)

        # Activation Functions for MLP
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu}
        self.activation = self.activations[options.activation]

        # Embeddings Layers
        self.ldims = options.lstm_dims
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in enum_word.items()}
        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2

        self.embeddings = EmbeddingsModule(len(vocab) + 3, options.wembedding_dims)

        # LSTM Network Layers
        self.ldims = options.lstm_dims
        self.biLstms = BiLSTMModule(self.ldims)

        # Concatenation Layers
        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.concatHeads = ConcatHeadModule(self.ldims, self.hidden_units, self.hidden2_units)
        self.concatRelations = ConcatRelationModule(rels, self.ldims, self.hidden_units, self.hidden2_units)

    def call(self, inputs, num_vec):
        # Forward pass
        #TODO: Implement concat module
        return inputs


class MSTParserLSTM:

    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos):
        self.model = MSTParserLSTMModel(vocab, rels, enum_word, options)
        self.trainer = get_optim(options)

        # Input data
        self.sample_size = 1  # batch size

    def train(self, conll_path):
        print('tensorflow version: ', tf.version.VERSION)
        batch = 1
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        iSentence = 0
        start = time.time()
        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)
            errs = []
            lerrs = []
            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print('Processing sentence number:', iSentence,
                          'eloss:', eloss,
                          'etotal:', etotal,
                          'Loss:', eloss / etotal,
                          'eerrors:', float(eerrors),
                          'Errors:', (float(eerrors)) / etotal,
                          'Time', time.time() - start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tape:

                    for entry in conll_sentence:
                        embeddings_input = self.get_embeddings_input(entry)
                        word_vec = self.model.embeddings(embeddings_input)
                        # TODO: entry.vec returns a 3D numpy array check if it requires a 2D one
                        entry.vec = word_vec
                        entry.lstms = [entry.vec, entry.vec]
                        entry.headfov = None
                        entry.modfov = None

                        entry.rheadfov = None
                        entry.rmodfov = None

                    bi_lstm_input = self.get_model_input(conll_sentence)
                    num_vec = len(conll_sentence)
                    bi_lstms_output = self.model.biLstms(bi_lstm_input, num_vec)
                    res_for_2 = tf.reshape(bi_lstms_output[0], shape=(num_vec, self.model.ldims))
                    res_back_2 = tf.reshape(bi_lstms_output[1], shape=(num_vec, self.model.ldims))

                    concat_input = []
                    for i in range(num_vec):
                        concat_input.append(res_for_2[i])
                        concat_input.append(res_back_2[num_vec - i - 1])

                    print('Come on JSL, Come on!!!')
                    e_output = self.model(conll_sentence, errs, lerrs)
                    # here the errs and lerrs are output variables with tensor values after the forward
                    eerrors += e_output
                    eloss += e_output
                    mloss += e_output
                    etotal += len(sentence)

                    if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                        if len(errs) > 0 or len(lerrs) > 0:
                            reshaped_lerrs = [tf.reshape(item, [1]) for item in lerrs]
                            eerrs_sum = loss_function(errs, reshaped_lerrs)

                if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                    if len(errs) > 0 or len(lerrs) > 0:
                        grads = tape.gradient(eerrs_sum, self.model.trainable_variables)
                        # print("Before apply gradients")
                        # print(self.model.sources)
                        self.trainer.apply_gradients(zip(grads, self.model.trainable_variables))
                        # print("After apply gradients")sources
                        # print(self.model.sources)
                        errs = []
                        lerrs = []
        # # if len(shuffledData) - 1 == iSentence:
        #     print('hidLayerFOM values on last iteration within an epoch')
        #     print(self.model.hidLayerFOM)
        print("Loss: ", mloss / iSentence)

    def get_embeddings_input(self, entry):
        c = float(self.model.wordsCount.get(entry.norm, 0))
        dropFlag = (random.random() < (c / (0.25 + c)))
        w_index = np.array(self.model.vocab.get(entry.norm, 0)).reshape(1) if dropFlag else np.array(0).reshape(1)
        return w_index

    def get_model_input(self, sentence):
        num_vec = len(sentence)
        features_for = [entry.vec for entry in sentence]
        features_back = [entry.vec for entry in reversed(sentence)]
        vec_for = tf.reshape(tf.concat(features_for, 0), shape=(self.sample_size, num_vec, -1))
        vec_back = tf.reshape(tf.concat(features_back, 0), shape=(self.sample_size, num_vec, -1))
        return [vec_for, vec_back]

    def save(self, fn):
        tmp = fn + '.tmp'
        tf.saved_model.save(self.model, tmp)
        shutil.move(tmp, fn)
        # tf.saved_model.simple_save
