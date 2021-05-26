import random

import numpy as np
import tensorflow as tf

import utils
from parser_modules_tf import EmbeddingsModule
from utils import read_conll


class MSTParserLSTM:

    def __init__(self, vocab, rels, enum_word, options):
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.rel_list = rels

        # LSTM Network Layers
        self.ldims = options.lstm_dims
        # self.biLstms = BiLSTMModule(self.ldims)

        # Embeddings Layers
        self.embeddings = EmbeddingsModule(len(vocab) + 3, options.wembedding_dims)
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in enum_word.items()}
        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2

        # Input data
        self.sample_size = 1  # batch size

    def get_document_embeddings(self, conll_path):

        document_embeddings = []
        document_bi_lstm = []

        with open(conll_path, 'r') as conllFP:
            shuffled_data = list(read_conll(conllFP))
            random.shuffle(shuffled_data)

            for iSentence, sentence in enumerate(shuffled_data):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                sentence_embeddings = []
                for entry in conll_sentence:
                    embeddings_input = self.get_embeddings_input(entry)
                    word_vec = self.embeddings(embeddings_input)
                    sentence_embeddings.append(word_vec)

                bi_lstm_input = self.get_bi_lstm_input(sentence_embeddings)

                sentence_embeddings = [np.array(word_vec) for word_vec in sentence_embeddings]
                sentence_bi_lstm = [np.array(lstm) for lstm in bi_lstm_input]

                document_embeddings.append(sentence_embeddings)
                document_bi_lstm.append(sentence_bi_lstm)

        return document_embeddings, document_bi_lstm

    def get_embeddings_input(self, entry):
        c = float(self.wordsCount.get(entry.norm, 0))
        dropFlag = (random.random() < (c / (0.25 + c)))
        w_index = np.array(self.vocab.get(entry.norm, 0)).reshape(1) if dropFlag else np.array(0).reshape(1)

        return w_index

    def get_bi_lstm_input(self, sentence_embeddings):
        num_vec = len(sentence_embeddings)
        features_for = sentence_embeddings
        features_back = list(reversed(sentence_embeddings))
        vec_for = tf.reshape(tf.concat(features_for, 0), shape=(num_vec, self.sample_size, -1))
        vec_back = tf.reshape(tf.concat(features_back, 0), shape=(num_vec, self.sample_size, -1))

        return [vec_for, vec_back]

