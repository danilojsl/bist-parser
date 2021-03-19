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
import decoder_tf
import decoder_tf_v2
import utils
from utils import read_conll


def concatenate_tensors(arrays):
    valid_l = [x for x in arrays if x is not None]  # This code removes None elements from an array
    dimension = len(valid_l[0].shape) - 1
    return tf.concat(valid_l, dimension)


def concatenate_layers(array1, array2, num_vec):
    concat_size = array1.shape[1] + array2.shape[1]
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


class EmbeddingsModule(tf.keras.layers.Layer):

    def __init__(self, vocab_size, wdims):
        super().__init__()
        self.wdims = wdims

        self.wlookup = Embedding(vocab_size, self.wdims, name='embedding_vocab',
                                 embeddings_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1.0))

    def call(self, inputs):
        # Forward pass
        word_vec = self.wlookup(inputs) if self.wdims > 0 else None
        return word_vec


class FirstBiLSTMModule(tf.keras.layers.Layer):

    def __init__(self, lstm_dims):
        super().__init__()
        # Input data
        self.sample_size = 1  # batch size
        self.lstm_for_1 = LSTM(lstm_dims, return_sequences=True, return_state=True)
        self.lstm_back_1 = LSTM(lstm_dims, return_sequences=True, return_state=True)
        # Initializing hidden and cell states values to 0
        self.hid_for_1, self.hid_back_1 = [init_hidden(lstm_dims) for _ in range(2)]

        self.__num_vec = None

    def call(self, inputs):
        # Forward pass
        # TODO: Raise error when inputs[0].shape.dims != inputs[1].shape.dims
        self.__num_vec = inputs[0].shape.dims[1]

        vec_for = inputs[0]
        vec_back = inputs[1]
        res_for_1, self.hid_for_1 = self.get_lstm_output(self.lstm_for_1, vec_for, self.hid_for_1)
        res_back_1, self.hid_back_1 = self.get_lstm_output(self.lstm_back_1, vec_back, self.hid_back_1)

        vec_cat = concatenate_layers(res_for_1[0], res_back_1[0], self.__num_vec)

        vec_for_2 = tf.reshape(tf.concat(vec_cat, 0), shape=(self.sample_size, self.__num_vec, -1))
        vec_back_2 = tf.reshape(tf.concat(list(reversed(vec_cat)), 0), shape=(self.sample_size, self.__num_vec, -1))

        return [vec_for_2, vec_back_2]

    @staticmethod
    def get_lstm_output(lstm_model, input_sequence, initial_state):
        output = lstm_model(input_sequence, initial_state=initial_state)
        hidden_states, hidden_state, cell_state = output[0], output[1], output[2]
        return hidden_states, (hidden_state, cell_state)


class NextBiLSTMModule(tf.keras.layers.Layer):

    def __init__(self, lstm_dims):
        super().__init__()
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
        super().__init__()
        self.biLstm1 = FirstBiLSTMModule(lstm_dims)
        self.biLstm2 = NextBiLSTMModule(lstm_dims)
        self.__sentence = None

    def set_sentence(self, sentence):
        self.__sentence = sentence

    def call(self, inputs):
        # Forward pass
        bi_lstm1_output = self.biLstm1(inputs)
        bi_lstm2_output = self.biLstm2(bi_lstm1_output)
        return bi_lstm2_output


class ConcatHeadModule(tf.keras.layers.Layer):

    def __init__(self, ldims, hidden_units, hidden2_units, activation):
        super().__init__()

        self.activation = activation
        self.hidden2_units = hidden2_units
        self.hidLayerFOH = Parameter((ldims * 2, hidden_units), 'hidLayerFOH')
        self.hidLayerFOM = Parameter((ldims * 2, hidden_units), 'hidLayerFOM')
        self.hidBias = Parameter(hidden_units, 'hidBias')
        self.catBias = Parameter(hidden_units * 2, 'catBias')

        if self.hidden2_units:
            self.hid2Layer = Parameter((hidden_units * 2, hidden2_units), 'hid2Layer')
            self.hid2Bias = Parameter(hidden2_units, 'hid2Bias')

        self.outLayer = Parameter((hidden2_units if hidden2_units > 0 else hidden_units, 1), 'outLayer')
        self.outBias = Parameter(1)

        self.__sentence = None

    def set_sentence(self, sentence):
        self.__sentence = sentence

    def call(self, inputs, training):

        # @tf.function
        # def get_heads():
        #     heads_tf = decoder_tf.parse_proj(scores, gold)
        #     return heads_tf

        # Forward pass
        scores, exprs = self.__evaluate(inputs)
        if training:
            gold = [entry.parent_id for entry in self.__sentence]
            heads = decoder_tf_v2.parse_proj(scores, gold)

            e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
            errs = []
            if e > 0:
                errs += [(exprs[h][i] - exprs[g][i])[0] for i, (h, g) in enumerate(zip(heads, gold)) if h != g]
            output = [errs, tf.constant([e])]
        else:
            output = []
        return output

    def __evaluate(self, inputs):

        # @tf.function
        # def convert_to_numpy(tensor_list):
        #     return list(map(lambda t: t.numpy()[0, 0], tensor_list))

        def transform_tensor(tensor_list):
            return list(map(lambda tensor: tensor[0, 0], tensor_list))

        head_vector = []  # TODO: Check if this array must be a Tensor variable to save it in the model
        for index in range(len(inputs)):
            lstms_0 = inputs[index][0]
            lstms_1 = inputs[index][1]
            concatenated_lstm = tf.reshape(concatenate_tensors([lstms_0, lstms_1]), shape=(1, -1))
            headfov = tf.matmul(concatenated_lstm, self.hidLayerFOH)
            concatenated_lstm = tf.reshape(concatenate_tensors([lstms_0, lstms_1]), shape=(1, -1))
            modfov = tf.matmul(concatenated_lstm, self.hidLayerFOM)
            # sentence[index].headfov = headfov
            # sentence[index].modfov = modfov
            head_vector.append([headfov, modfov])

        exprs = [[self.__getExpr(head_vector, i, j) for j in range(len(head_vector))] for i in range(len(head_vector))]

        output_tensor = [[output for output in exprsRow] for exprsRow in exprs]
        # scores = np.array([convert_to_numpy(output) for output in output_tensor])
        scores = tf.stack([transform_tensor(output) for output in output_tensor])

        return scores, exprs

    def __getExpr(self, head_vector, i, j):
        headfov = head_vector[i][0]
        modfov = head_vector[j][1]
        if self.hidden2_units > 0:
            concatenated_result = concatenate_tensors([headfov, modfov])
            activation_result = self.activation(concatenated_result + self.catBias)
            matmul_result = tf.matmul(activation_result, self.hid2Layer)
            next_activation_result = self.activation(self.hid2Bias + matmul_result)
            output = tf.matmul(next_activation_result, self.outLayer) + self.outBias
        else:
            activation_result = self.activation(headfov + modfov + self.hidBias)
            output = tf.matmul(activation_result, self.outLayer) + self.outBias

        return output


class ConcatRelationModule(tf.keras.layers.Layer):

    def __init__(self, rels, ldims, hidden_units, hidden2_units, activation):
        super().__init__()

        self.activation = activation
        self.hidden2_units = hidden2_units
        self.rhidLayerFOH = Parameter((2 * ldims, hidden_units), 'rhidLayerFOH')
        self.rhidLayerFOM = Parameter((2 * ldims, hidden_units), 'rhidLayerFOM')
        self.rhidBias = Parameter(hidden_units, 'rhidBias')
        self.rcatBias = Parameter(hidden_units * 2, 'rcatBias')
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.rel_list = rels

        if self.hidden2_units:
            self.rhid2Layer = Parameter((hidden_units * 2, hidden2_units), 'rhid2Layer')
            self.rhid2Bias = Parameter(hidden2_units, 'rhid2Bias')

        self.routLayer = Parameter((hidden2_units if hidden2_units > 0 else hidden_units, len(self.rel_list)),
                                   'routLayer')
        self.routBias = Parameter((len(self.rel_list)), 'routBias')

        self.__sentence = None

    def set_sentence(self, sentence):
        self.__sentence = sentence

    def call(self, inputs):
        # Forwad pass
        gold = [entry.parent_id for entry in self.__sentence]
        lerrs = []
        for modifier, head in enumerate(gold[1:]):
            lstms_0 = inputs[head][0]
            lstms_1 = inputs[modifier + 1][1]

            if self.__sentence[head].rheadfov is None:
                concatenated_lstm = tf.reshape(concatenate_tensors([lstms_0, lstms_1]), shape=(1, -1))
                self.__sentence[head].rheadfov = tf.matmul(concatenated_lstm, self.rhidLayerFOH)

            if self.__sentence[modifier + 1].modfov is None:
                concatenated_lstm = tf.reshape(concatenate_tensors([lstms_0, lstms_1]), shape=(1, -1))
                self.__sentence[modifier + 1].modfov = tf.matmul(concatenated_lstm, self.rhidLayerFOM)

            rscores, rexprs = self.__evaluateLabel(self.__sentence[head].rheadfov, self.__sentence[modifier + 1].modfov)
            goldLabelInd = self.rels[self.__sentence[modifier + 1].relation]
            wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
            if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                lerrs += [rexprs[wrongLabelInd] - rexprs[goldLabelInd]]

        return lerrs

    def __evaluateLabel(self, rheadfov, rmodfov):

        if self.hidden2_units > 0:
            concatenated_result = concatenate_tensors([rheadfov, rmodfov])
            activation_result = self.activation(concatenated_result + self.rcatBias)
            matmul_result = tf.matmul(activation_result, self.rhid2Layer)
            next_activation_result = self.activation(self.rhid2Bias + matmul_result)
            output = tf.matmul(next_activation_result, self.routLayer) + self.routBias
        else:
            activation_result = self.activation(rheadfov + rmodfov + self.rhidBias)
            output = tf.matmul(activation_result, self.routLayer) + self.routBias

        return output.numpy()[0], output[0]


class MSTParserLSTMModel(tf.keras.Model):

    def __init__(self, vocab, rels, enum_word, options):
        super().__init__()
        random.seed(1)
        # Activation Functions for Concatenation Modules MLP
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
        self.concatHeads = ConcatHeadModule(self.ldims, options.hidden_units, options.hidden2_units, self.activation)
        self.concatRelations = ConcatRelationModule(rels, self.ldims, options.hidden_units, options.hidden2_units,
                                                    self.activation)

        self.__sentence = None
        self.__num_vec = None

    def set_sentence(self, sentence):
        self.__sentence = sentence

    def call(self, inputs, training):
        # Forward pass
        # TODO: Raise error when inputs[0].shape.dims != inputs[1].shape.dims
        self.__num_vec = inputs[0].shape.dims[1]
        self.biLstms.set_sentence(self.__sentence)

        bi_lstms_output = self.biLstms(inputs)
        res_for_2 = tf.reshape(bi_lstms_output[0], shape=(self.__num_vec, self.ldims))
        res_back_2 = tf.reshape(bi_lstms_output[1], shape=(self.__num_vec, self.ldims))

        concat_input = []
        for i in range(self.__num_vec):
            lstms_0 = res_for_2[i]
            lstms_1 = res_back_2[self.__num_vec - i - 1]
            # sentence[i].lstms[0] = lstms_0
            # sentence[i].lstms[1] = lstms_1
            concat_input.append([lstms_0, lstms_1])

        self.concatHeads.set_sentence(self.__sentence)
        output = self.concatHeads(concat_input, training)
        errs = output[0]
        e_tensor = output[1]

        self.concatRelations.set_sentence(self.__sentence)
        lerrs = self.concatRelations(concat_input)

        return [e_tensor, errs, lerrs]


class MSTParserLSTM:

    def __init__(self, vocab, rels, enum_word, options):
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

                with tf.GradientTape() as tape:

                    for entry in conll_sentence:
                        embeddings_input = self.get_embeddings_input(entry)
                        word_vec = self.model.embeddings(embeddings_input)
                        entry.vec = word_vec
                        entry.lstms = [entry.vec, entry.vec]
                        entry.headfov = None
                        entry.modfov = None

                        entry.rheadfov = None
                        entry.rmodfov = None

                    bi_lstm_input = self.get_bi_lstm_input(conll_sentence)

                    self.model.set_sentence(conll_sentence)
                    e_output, errs, lerrs = self.model(bi_lstm_input, True)

                    e_output = e_output.numpy()[0]
                    eerrors += e_output
                    eloss += e_output
                    mloss += e_output
                    etotal += len(sentence)

                    if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                        if len(errs) > 0 or len(lerrs) > 0:
                            reshaped_lerrs = [tf.reshape(item, [1]) for item in lerrs]
                            eerrs_sum = self.loss_function(errs, reshaped_lerrs)

                if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                    if len(errs) > 0 or len(lerrs) > 0:
                        grads = tape.gradient(eerrs_sum, self.model.trainable_variables)
                        self.trainer.apply_gradients(zip(grads, self.model.trainable_variables))
        print("Loss: ", mloss / iSentence)

    @staticmethod
    def loss_function(y_true, y_pred):
        l_variable = y_true + y_pred
        return tf.reduce_sum(concatenate_tensors(l_variable))

    def get_embeddings_input(self, entry):
        c = float(self.model.wordsCount.get(entry.norm, 0))
        dropFlag = (random.random() < (c / (0.25 + c)))
        w_index = np.array(self.model.vocab.get(entry.norm, 0)).reshape(1) if dropFlag else np.array(0).reshape(1)
        return w_index

    def get_bi_lstm_input(self, sentence):
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
