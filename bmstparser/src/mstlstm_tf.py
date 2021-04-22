import random
import shutil
import time
from operator import itemgetter

import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as F

import decoder_tf
import utils
import utils_tf
from parser_layers_tf import BiLSTMModule
from parser_layers_tf import ConcatHeadModule
from parser_layers_tf import ConcatRelationModule
from parser_layers_tf import EmbeddingsModule
from utils import read_conll


def init_hidden(dim):
    return tf.zeros(shape=[1, dim]), tf.zeros(shape=[1, dim])


def get_optim(opt):
    if opt.optim == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=opt.lr)
    elif opt.optim == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=opt.lr, epsilon=1e-8)


class MSTParserLSTMModel(tf.keras.Model):

    def __init__(self, relations_size, options):
        super().__init__()
        random.seed(1)  # TODO: Check if this seed is useful
        # Activation Functions for Concatenation Modules MLP
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu}
        self.activation = self.activations[options.activation]

        # LSTM Network Layers
        self.ldims = options.lstm_dims
        self.biLstms = BiLSTMModule(self.ldims)

        # Concatenation Layers
        self.concatHeads = ConcatHeadModule(self.ldims, options.hidden_units, options.hidden2_units, self.activation)
        self.concatRelations = ConcatRelationModule(relations_size, self.ldims, options.hidden_units,
                                                    options.hidden2_units, self.activation)

        self.__sentence = None

    def set_sentence(self, sentence):
        self.__sentence = sentence

    def call(self, inputs):
        # Forward pass
        # TODO: Raise error when inputs[0].shape.dims != inputs[1].shape.dims
        num_vec = inputs[0].shape.dims[1]
        self.biLstms.set_sentence(self.__sentence)

        bi_lstms_output = self.biLstms(inputs)

        res_for_2 = tf.reshape(bi_lstms_output[0], shape=(num_vec, self.ldims))
        res_back_2 = tf.reshape(bi_lstms_output[1], shape=(num_vec, self.ldims))

        concat_input = []
        for i in range(num_vec):
            lstms_0 = res_for_2[i]
            lstms_1 = res_back_2[num_vec - i - 1]
            concat_input.append([lstms_0, lstms_1])

        self.concatHeads.set_sentence(self.__sentence)
        heads_output = self.concatHeads(concat_input)

        self.concatRelations.set_sentence(self.__sentence)
        relations_output = self.concatRelations(concat_input)

        self.concatHeads.set_sentence(None)
        self.concatRelations.set_sentence(None)
        return [heads_output, relations_output]


class MSTParserLSTM:

    def __init__(self, vocab, rels, enum_word, options):
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.rel_list = rels

        self.model = MSTParserLSTMModel(len(self.rel_list), options)
        self.trainer = get_optim(options)

        # Embeddings Layers
        self.embeddings = EmbeddingsModule(len(vocab) + 3, options.wembedding_dims)
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in enum_word.items()}
        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2

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
                          'Loss:', eloss / etotal,
                          'Time', time.time() - start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                gold_v2 = []
                for entry in conll_sentence:
                    embeddings_input = self.get_embeddings_input(entry)
                    word_vec = self.embeddings(embeddings_input)
                    entry.vec = word_vec
                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None
                    gold_v2.append(entry.parent_id)

                bi_lstm_input = self.get_bi_lstm_input(conll_sentence)

                with tf.GradientTape() as tape:

                    self.model.set_sentence(conll_sentence)
                    model_output = self.model(bi_lstm_input)
                    heads_output = model_output[0]
                    relations_output = model_output[1]

                    lerrs = self.get_relation_errors(relations_output, conll_sentence)
                    errs, e_output = self.get_head_errors(conll_sentence, heads_output)

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
                self.model.set_sentence(None)

        print("Loss: ", mloss / iSentence)

    @staticmethod
    def loss_function(y_true, y_pred):
        l_variable = y_true + y_pred
        return tf.reduce_sum(utils_tf.concatenate_tensors(l_variable))

    def get_embeddings_input(self, entry):
        c = float(self.wordsCount.get(entry.norm, 0))
        dropFlag = (random.random() < (c / (0.25 + c)))
        w_index = np.array(self.vocab.get(entry.norm, 0)).reshape(1) if dropFlag else np.array(0).reshape(1)

        return w_index

    def get_bi_lstm_input(self, sentence):
        num_vec = len(sentence)
        features_for = [entry.vec for entry in sentence]
        features_back = [entry.vec for entry in reversed(sentence)]
        vec_for = tf.reshape(tf.concat(features_for, 0), shape=(self.sample_size, num_vec, -1))
        vec_back = tf.reshape(tf.concat(features_back, 0), shape=(self.sample_size, num_vec, -1))

        return [vec_for, vec_back]

    @staticmethod
    def get_head_errors(sentence, heads_output):
        scores, exprs = heads_output[0], heads_output[1]

        gold = [entry.parent_id for entry in sentence]
        heads = decoder_tf.parse_proj(scores, gold)

        e_output = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
        errs = []
        if e_output > 0:
            errs += [(exprs[h][i] - exprs[g][i])[0] for i, (h, g) in enumerate(zip(heads, gold)) if h != g]

        return errs, e_output

    def get_relation_errors(self, relations_output, sentence):
        lerrs = []
        for modifier, rscores in enumerate(relations_output):
            goldLabelInd = self.rels[sentence[modifier + 1].relation]
            wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd),
                                key=itemgetter(1))[0]
            if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                lerrs += [rscores[wrongLabelInd] - rscores[goldLabelInd]]

        return lerrs

    def save(self, fn):
        tmp = fn + '.tmp'
        tf.saved_model.save(self.model, tmp)
        shutil.move(tmp, fn)
