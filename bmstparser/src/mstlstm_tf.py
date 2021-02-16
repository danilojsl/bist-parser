import random
from operator import itemgetter

import numpy as np
import time
import tensorflow as tf
import tensorflow.keras.activations as F
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

import decoder
import utils
from utils import read_conll


def concatenate_arrays(arrays, array_type):
    valid_l = [x for x in arrays if x is not None]  # This code removes None elements from an array
    dimension = len(valid_l[0].shape) - 1
    if array_type == 'tensor':
        return tf.concat(valid_l, dimension)
    else:
        return np.concatenate(valid_l, dimension)


def concatenate_layers(array1, array2, num_vec):
    # TODO: Check if we can call concatenate_numpy_arrays in return clause
    concat_size = array1.shape[1] + array1.shape[1]
    return [np.concatenate([array1[i], array2[num_vec - i - 1]], 0).reshape(1, concat_size) for i in range(num_vec)]


def Parameter(shape=None, name='param'):
    shape = (1, shape) if type(shape) == int else shape
    random_tensor = tf.get_variable('random_tensor', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return tf.Variable(random_tensor, name=name)


class MSTParserLSTMModel:

    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos):
        random.seed(1)

        # Activation Functions (For MLP??)
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu}
        self.activation = self.activations[options.activation]

        # Input data
        self.sample_size = 1  # batch size

        # Variables required for prediction
        # TODO: Check if this variables are really required at this level
        # self.hid_for_1 = ()
        # self.hid_back_1 = ()
        # self.hid_for_2 = ()
        # self.hid_back_2 = ()

        # Embeddings layers
        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.odims = options.oembedding_dims
        self.cdims = options.cembedding_dims
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in enum_word.items()}
        self.pos = {word: ind + 3 for ind, word in enumerate(pos)}
        self.onto = {word: ind + 3 for ind, word in enumerate(onto)}
        self.cpos = {word: ind + 3 for ind, word in enumerate(cpos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.rel_list = rels

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1
        self.onto['*PAD*'] = 1
        self.cpos['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2
        self.onto['*INITIAL*'] = 2
        self.cpos['*INITIAL*'] = 2

        self.edim = 0

        self.wlookup = Sequential(Embedding(len(vocab) + 3, self.wdims))
        self.plookup = Sequential(Embedding(len(pos) + 3, self.pdims))
        self.rlookup = Sequential(Embedding(len(rels) + 3, self.rdims))
        self.olookup = Sequential(Embedding(len(onto) + 3, self.odims))
        self.clookup = Sequential(Embedding(len(cpos) + 3, self.cdims))

        # Weight Initialization
        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.hidLayerFOH = Parameter((self.ldims * 2, self.hidden_units), 'hidLayerFOH')
        self.sources = [self.hidLayerFOH]
        self.hidLayerFOM = Parameter((self.ldims * 2, self.hidden_units), 'hidLayerFOM')
        self.sources.append(self.hidLayerFOM)
        self.hidBias = Parameter(self.hidden_units, 'hidBias')
        self.sources.append(self.hidBias)
        self.catBias = Parameter(self.hidden_units * 2, 'catBias')
        self.sources.append(self.catBias)
        self.rhidLayerFOH = Parameter((2 * self.ldims, self.hidden_units), 'rhidLayerFOH')
        self.sources.append(self.rhidLayerFOH)
        self.rhidLayerFOM = Parameter((2 * self.ldims, self.hidden_units), 'rhidLayerFOM')
        self.sources.append(self.rhidLayerFOM)
        self.rhidBias = Parameter(self.hidden_units, 'rhidBias')
        self.sources.append(self.rhidBias)
        self.rcatBias = Parameter(self.hidden_units * 2, 'rcatBias')
        self.sources.append(self.rcatBias)

        if self.hidden2_units:
            self.hid2Layer = Parameter((self.hidden_units * 2, self.hidden2_units), 'hid2Layer')
            self.sources.append(self.hid2Layer)
            self.hid2Bias = Parameter(self.hidden2_units, 'hid2Bias')
            self.sources.append(self.hid2Bias)
            self.rhid2Layer = Parameter((self.hidden_units * 2, self.hidden2_units), 'rhid2Layer')
            self.sources.append(self.rhid2Layer)
            self.rhid2Bias = Parameter(self.hidden2_units, 'rhid2Bias')
            self.sources.append(self.rhid2Bias)

        self.outLayer = Parameter(
            (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, 1), 'outLayer')
        self.outBias = 0  # Parameter(1)
        self.routLayer = Parameter(
            (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, len(self.rel_list)), 'routLayer')
        self.routBias = Parameter((len(self.rel_list)), 'routBias')

    def forward(self, sentence, errs, lerrs):

        self.process_sentence_embeddings(sentence)
        # TODO: Define LSTM architecture on each iteration based on sentence length (time steps)
        num_vec = len(sentence)  # time steps

        # LSTM Network architecture
        # First LSTM Layer
        number_features = self.wdims + self.pdims + self.edim + self.odims + self.cdims
        lstm_for_1 = self.prepare_lstm_model(num_vec, number_features, self.ldims)
        lstm_back_1 = self.prepare_lstm_model(num_vec, number_features, self.ldims)

        # Second LSTM Layer
        lstm_for_2 = self.prepare_lstm_model(num_vec, self.ldims * 2, self.ldims)
        lstm_back_2 = self.prepare_lstm_model(num_vec, self.ldims * 2, self.ldims)

        features_for = [entry.vec for entry in sentence]
        features_back = [entry.vec for entry in reversed(sentence)]
        vec_for = np.concatenate(features_for).reshape(self.sample_size, num_vec, -1)
        vec_back = np.concatenate(features_back).reshape(self.sample_size, num_vec, -1)

        res_for_1, hid_for_1 = self.get_lstm_output(lstm_for_1, vec_for)
        res_back_1, hid_back_1 = self.get_lstm_output(lstm_back_1, vec_back)

        vec_cat = concatenate_layers(res_for_1[0], res_back_1[0], num_vec)
        vec_for_2 = np.concatenate(vec_cat).reshape(self.sample_size, num_vec, -1)
        vec_back_2 = np.concatenate(list(reversed(vec_cat))).reshape(self.sample_size, num_vec, -1)

        res_for_2, hid_for_2 = self.get_lstm_output(lstm_for_2, vec_for_2)
        res_back_2, hid_back_2 = self.get_lstm_output(lstm_back_2, vec_back_2)

        for i in range(num_vec):
            sentence[i].lstms[0] = res_for_2.reshape(num_vec, self.ldims)[i]
            sentence[i].lstms[1] = res_back_2.reshape(num_vec, self.ldims)[num_vec - i - 1]

        scores, exprs = self.__evaluate(sentence)
        gold = [entry.parent_id for entry in sentence]
        heads = decoder.parse_proj(scores, gold)

        for modifier, head in enumerate(gold[1:]):
            rscores, rexprs = self.__evaluateLabel(sentence, head, modifier + 1)
            goldLabelInd = self.rels[sentence[modifier + 1].relation]
            wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd),
                                key=itemgetter(1))[0]
            if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                lerrs += [rexprs[wrongLabelInd] - rexprs[goldLabelInd]]

        e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
        if e > 0:
            errs += [(exprs[h][i] - exprs[g][i])[0]
                     for i, (h, g) in enumerate(zip(heads, gold)) if h != g]
        return e

    def process_sentence_embeddings(self, sentence):
        for entry in sentence:
            c = float(self.wordsCount.get(entry.norm, 0))
            dropFlag = (random.random() < (c / (0.25 + c)))
            w_index = np.array(self.vocab.get(entry.norm, 0)).reshape(1) if dropFlag else np.array(0).reshape(1)
            wordvec = self.wlookup.predict(w_index) if self.wdims > 0 else None
            o_index = np.array(self.onto[entry.onto] if random.random() < 0.9 else np.array(0).reshape(1))
            ontovec = self.olookup(o_index) if self.odims > 0 else None
            cpos_index = np.array(self.cpos[entry.cpos] if random.random() < 0.9 else np.array(0).reshape(1))
            cposvec = self.clookup(cpos_index) if self.cdims > 0 else None
            posvec = self.plookup(self.pos[entry.pos]) if self.pdims > 0 else None
            evec = None
            # The dot notation create attributes for the class ConllEntry in run time
            # TODO: entry.vec returns a 3D numpy array check if it requires a 2D one
            entry.vec = concatenate_arrays([wordvec, posvec, ontovec, cposvec, evec], 'numpy')
            entry.lstms = [entry.vec, entry.vec]
            entry.headfov = None
            entry.modfov = None

            entry.rheadfov = None
            entry.rmodfov = None

    @staticmethod
    def prepare_lstm_model(time_steps, number_features, hidden_state_size):
        inputs = Input(shape=(time_steps, number_features))
        hidden_states, hidden_state, cell_state = \
            LSTM(hidden_state_size, return_sequences=True, return_state=True)(inputs)
        lstm_model = Model(inputs=inputs, outputs=[hidden_states, hidden_state, cell_state])
        return lstm_model

    @staticmethod
    def get_lstm_output(lstm_model, input_sequence):
        # lstm_model.summary()
        output = lstm_model.predict(input_sequence)
        hidden_states, hidden_state, cell_state = output[0], output[1], output[2]
        return hidden_states, (hidden_state, cell_state)

    def __evaluate(self, sentence):
        exprs = [[self.__getExpr(sentence, i, j) for j in range(len(sentence))]
                 for i in range(len(sentence))]
        scores = np.array([[output.numpy()[0, 0] for output in exprsRow]
                           for exprsRow in exprs])
        return scores, exprs

    def __getExpr(self, sentence, i, j):

        if sentence[i].headfov is None:
            concatenated_lstm = concatenate_arrays([sentence[i].lstms[0], sentence[i].lstms[1]],
                                                   'numpy').reshape(1, -1)
            sentence[i].headfov = tf.matmul(tf.convert_to_tensor(concatenated_lstm), self.hidLayerFOH)

        if sentence[j].modfov is None:
            concatenated_lstm = concatenate_arrays([sentence[j].lstms[0], sentence[j].lstms[1]],
                                                   'numpy').reshape(1, -1)
            sentence[j].modfov = tf.matmul(tf.convert_to_tensor(concatenated_lstm), self.hidLayerFOM)

        if self.hidden2_units > 0:
            concatenated_result = concatenate_arrays([sentence[i].headfov, sentence[j].modfov], 'tensor')
            activation_result = self.activation(concatenated_result + self.catBias)
            matmul_result = tf.matmul(activation_result, self.hid2Layer)
            next_activation_result = self.activation(self.hid2Bias + matmul_result)
            output = tf.matmul(next_activation_result, self.outLayer) + self.outBias
        else:
            activation_result = self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias)
            output = tf.matmul(activation_result, self.outLayer) + self.outBias

        return output

    def __evaluateLabel(self, sentence, i, j):

        if sentence[i].rheadfov is None:
            concatenated_lstm = concatenate_arrays([sentence[i].lstms[0], sentence[i].lstms[1]],
                                                   'numpy').reshape(1, -1)
            sentence[i].rheadfov = tf.matmul(tf.convert_to_tensor(concatenated_lstm), self.rhidLayerFOH)

        if sentence[j].rmodfov is None:
            concatenated_lstm = concatenate_arrays([sentence[j].lstms[0], sentence[j].lstms[1]],
                                                   'numpy').reshape(1, -1)
            sentence[j].rmodfov = tf.matmul(concatenated_lstm, self.rhidLayerFOM)

        if self.hidden2_units > 0:
            concatenated_result = concatenate_arrays([sentence[i].rheadfov, sentence[j].rmodfov], 'tensor')
            activation_result = self.activation(concatenated_result + self.rcatBias)
            matmul_result = tf.matmul(activation_result, self.rhid2Layer)
            next_activation_result = self.activation(self.rhid2Bias + matmul_result)
            output = tf.matmul(next_activation_result, self.routLayer) + self.routBias
        else:
            activation_result = (sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias)
            output = tf.matmul(activation_result, self.routLayer) + self.routBias

        return output.numpy()[0], output[0]


class MSTParserLSTM:

    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos):
        self.model = MSTParserLSTMModel(vocab, pos, rels, enum_word, options, onto, cpos)
        self.trainer = tf.keras.optimizers.SGD(learning_rate=0.1)

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
                          'Loss:', eloss / etotal,
                          'Errors:', (float(eerrors)) / etotal,
                          'Time', time.time() - start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                    tape.watch(self.model.sources)
                    e_output = self.model.forward(conll_sentence, errs, lerrs)
                    # here the errs and lerrs are output variables with tensor values after the forward
                    eerrors += e_output
                    eloss += e_output
                    mloss += e_output
                    etotal += len(sentence)

                    if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                        if len(errs) > 0 or len(lerrs) > 0:
                            reshaped_lerrs = [tf.reshape(item, [1]) for item in lerrs]
                            l_variable = errs + reshaped_lerrs
                            eerrs_sum = tf.reduce_sum(concatenate_arrays(l_variable, 'tensor'))

                    if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                        if len(errs) > 0 or len(lerrs) > 0:
                            grads = tape.gradient(eerrs_sum, self.model.sources)
                            # print("Before apply gradients")
                            # print(self.model.sources)
                            self.trainer.apply_gradients(zip(grads, self.model.sources))
                            # print("After apply gradients")
                            # print(self.model.sources)
                            errs = []
                            lerrs = []

        print("Loss: ", mloss / iSentence)
