import random
import time
from operator import itemgetter

import numpy as np
import tensorflow as tf
import tensorflow.keras.activations as F

import decoder_tf
import utils
import utils_tf
from parser_modules_tf import ConcatHeadModule
from parser_modules_tf import ConcatRelationModule
from parser_modules_tf import EmbeddingsLookup
from parser_modules_tf import FirstBiLSTMModule
from parser_modules_tf import NextBiLSTMModule
from utils import read_conll


def init_hidden(dim):
    return tf.zeros(shape=[1, dim]), tf.zeros(shape=[1, dim])


def get_optim(opt):
    if opt.optim == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=opt.lr)
    elif opt.optim == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=opt.lr, epsilon=1e-8)


class BiLSTMModel(tf.keras.Model):

    def __init__(self, lstm_dims):
        super().__init__()
        self.ldims = lstm_dims
        self.lstm = FirstBiLSTMModule(lstm_dims)
        self.nextLstm = NextBiLSTMModule(lstm_dims)

    def call(self, inputs):
        # Forward pass
        # TODO: Raise error when inputs[0].shape.dims != inputs[1].shape.dims
        lstm1_output = self.lstm(inputs)
        lstm2_output = self.nextLstm(lstm1_output)

        time_steps = inputs[0].shape.dims[1]
        res_for_2 = tf.reshape(lstm2_output[0], shape=(time_steps, self.ldims))
        res_back_2 = tf.reshape(lstm2_output[1], shape=(time_steps, self.ldims))

        output = []
        for i in range(time_steps):
            lstms_0 = res_for_2[i]
            lstms_1 = res_back_2[time_steps - i - 1]
            output.append([lstms_0, lstms_1])

        return output


class MSTParserLSTM:
    # Minimum-Spanning Tree Parser (MST)
    def __init__(self, vocab, relations, enum_word, options):
        self.trainer = get_optim(options)

        # Embeddings Layers
        self.embeddings = EmbeddingsLookup(len(vocab) + 3, options.wembedding_dims)
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in enum_word.items()}
        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2

        # BiLSTM Module
        self.biLSTMModel = BiLSTMModel(options.lstm_dims)

        # Concatenation Layers
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu}
        self.activation = self.activations[options.activation]
        self.relations_vocabulary = {word: ind for ind, word in enumerate(relations)}
        self.concatHeads = ConcatHeadModule(options.lstm_dims, options.hidden_units, options.hidden2_units,
                                            self.activation)
        self.concatRelations = ConcatRelationModule(self.relations_vocabulary, options.lstm_dims, options.hidden_units,
                                                    options.hidden2_units, self.activation)

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
            # random.shuffle(shuffledData)

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

                sentence_embeddings = []
                gold_heads = []
                for entry in conll_sentence:
                    embeddings_input = self.get_embeddings_input(entry)
                    word_vec = self.embeddings.lookup(embeddings_input)
                    sentence_embeddings.append(word_vec)
                    gold_heads.append(entry.parent_id)

                bi_lstm_input = self.get_bi_lstm_input(sentence_embeddings)
                with tf.GradientTape() as tape:

                    bi_lstm_output = self.biLSTMModel(bi_lstm_input)
                    heads_output = self.concatHeads(bi_lstm_output)

                    relations_input = []
                    for modifier, head in enumerate(gold_heads[1:]):
                        lstms_0 = bi_lstm_output[head][0]
                        lstms_1 = bi_lstm_output[modifier + 1][1]
                        concatenated_lstm = tf.reshape(utils_tf.concatenate_tensors([lstms_0, lstms_1]), shape=(1, -1))
                        relations_input.append(concatenated_lstm)

                    relations_output = self.concatRelations(relations_input)

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

                trainable_variables = self.biLSTMModel.trainable_variables + \
                                      self.concatHeads.trainable_variables + \
                                      self.concatRelations.trainable_variables
                if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                    if len(errs) > 0 or len(lerrs) > 0:
                        grads = tape.gradient(eerrs_sum, sources=trainable_variables)
                        self.trainer.apply_gradients(zip(grads, trainable_variables))

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

    def get_bi_lstm_input(self, embeddings):
        time_steps = len(embeddings)
        features_for = embeddings
        features_back = list(reversed(embeddings))
        vec_for = tf.reshape(tf.concat(features_for, 0), shape=(self.sample_size, time_steps, -1))
        vec_back = tf.reshape(tf.concat(features_back, 0), shape=(self.sample_size, time_steps, -1))

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
            goldLabelInd = self.relations_vocabulary[sentence[modifier + 1].relation]
            wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
            if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                lerrs += [rscores[wrongLabelInd] - rscores[goldLabelInd]]

        return lerrs

    def save(self, output_path, epoch):
        main_model_dir = output_path + epoch

        export_dir = main_model_dir + "/BiLSTM"
        tf.saved_model.save(self.biLSTMModel, export_dir)
        export_dir = main_model_dir + "/Heads"
        tf.saved_model.save(self.concatHeads, export_dir)
        export_dir = main_model_dir + "/Relations"
        tf.saved_model.save(self.concatRelations, export_dir)

    def save_light(self, output_path, epoch):
        main_model_dir = output_path + epoch
        export_dir = main_model_dir + "/BiLSTM"
        self.biLSTMModel.save(export_dir, save_traces=False)
        export_dir = main_model_dir + "/Heads"
        self.concatHeads.save(export_dir, save_traces=False)
        export_dir = main_model_dir + "/Relations"
        self.concatRelations.save(export_dir, save_traces=False)

    def get_model_variables(self):
        #w_first_lstm = self.biLSTMModel.trainable_variables[0]
        # wig_first_lstm = self.biLSTMModel.trainable_variables[1]
        # wfg_first_lstm = self.biLSTMModel.trainable_variables[2]
        # wog_first_lstm = self.biLSTMModel.trainable_variables[3]
        # weights_first_lstm = [w_first_lstm, wig_first_lstm, wfg_first_lstm, wog_first_lstm]
        # w_next_lstm = self.biLSTMModel.trainable_variables[4]
        # wig_next_lstm = self.biLSTMModel.trainable_variables[5]
        # wfg_next_lstm = self.biLSTMModel.trainable_variables[6]
        # wog_next_lstm = self.biLSTMModel.trainable_variables[7]
        # weights_next_lstm = [w_next_lstm, wig_next_lstm, wfg_next_lstm, wog_next_lstm]
        #
        # weights_bi_lstm = [weights_first_lstm, weights_next_lstm]
        first_bi_lstm = [self.biLSTMModel.lstm.lstm_for_1, self.biLSTMModel.lstm.lstm_back_1]
        next_bi_lstm = [self.biLSTMModel.nextLstm.lstm_for_2, self.biLSTMModel.nextLstm.lstm_back_2]
        bi_lstms = [first_bi_lstm, next_bi_lstm]

        hid_layer_foh = self.concatHeads.trainable_variables[0]
        hid_layer_fom = self.concatHeads.trainable_variables[1]
        hid_bias = self.concatHeads.trainable_variables[2]
        out_layer = self.concatHeads.trainable_variables[4]
        out_bias = self.concatHeads.trainable_variables[5]
        heads_variables = [hid_layer_foh, hid_layer_fom, hid_bias, out_layer, out_bias]

        r_hid_layer_foh = self.concatRelations.trainable_variables[0]
        r_hid_layer_fom = self.concatRelations.trainable_variables[1]
        r_hid_bias = self.concatRelations.trainable_variables[2]
        r_out_layer = self.concatRelations.trainable_variables[5]
        r_out_bias = self.concatRelations.trainable_variables[6]
        relations_vocabulary = self.concatRelations.trainable_variables[4]
        relations_variables = [r_hid_layer_foh, r_hid_layer_fom, r_hid_bias, r_out_layer, r_out_bias,
                               relations_vocabulary]

        return bi_lstms, heads_variables, relations_variables

