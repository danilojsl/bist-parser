import random
import shutil
import time
from operator import itemgetter

import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.init import *

import decoder
import utils
from utils import read_conll

use_gpu = True if torch.cuda.is_available() else False


def get_data(variable):
    if use_gpu:
        return variable.data.cpu()
    else:
        return variable.data


def Variable(inner):
    return torch.autograd.Variable(inner.cuda() if use_gpu else inner)


def Parameter(shape=None, init=xavier_uniform):
    if hasattr(init, 'shape'):
        assert not shape
        return nn.Parameter(torch.Tensor(init))
    shape = (1, shape) if type(shape) == int else shape
    return nn.Parameter(init(torch.Tensor(*shape)))


def scalar(f):
    if type(f) == int:
        return Variable(torch.LongTensor([f]))
    if type(f) == float:
        return Variable(torch.FloatTensor([f]))


def concatenate_tensors(l, dimension=-1):
    # dimension is always -1
    valid_l = [x for x in l if x is not None]  # This code removes None elements from an array
    if dimension < 0:
        dimension += len(valid_l[0].size())
    return torch.cat(valid_l, dimension)


class MSTParserLSTMModel(nn.Module):
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos):
        super(MSTParserLSTMModel, self).__init__()
        random.seed(1)
        # This should be the activation function for the MLP
        self.activations = {'tanh': F.tanh,
                            'sigmoid': F.sigmoid, 'relu': F.relu}
        self.activation = self.activations[options.activation]

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.odims = options.oembedding_dims
        self.cdims = options.cembedding_dims
        self.layers = options.lstm_layers
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

        # prepare LSTM
        number_features = self.wdims + self.pdims + self.edim + self.odims + self.cdims
        self.lstm_for_1 = nn.LSTM(input_size=number_features, hidden_size=self.ldims)
        self.lstm_back_1 = nn.LSTM(input_size=number_features, hidden_size=self.ldims)
        self.lstm_for_2 = nn.LSTM(input_size=self.ldims * 2, hidden_size=self.ldims)
        self.lstm_back_2 = nn.LSTM(input_size=self.ldims * 2, hidden_size=self.ldims)
        self.hid_for_1, self.hid_back_1, self.hid_for_2, self.hid_back_2 = [
            self.init_hidden(self.ldims) for _ in range(4)]

        self.wlookup = nn.Embedding(len(vocab) + 3, self.wdims)
        self.plookup = nn.Embedding(len(pos) + 3, self.pdims)
        self.rlookup = nn.Embedding(len(rels), self.rdims)
        self.olookup = nn.Embedding(len(onto) + 3, self.odims)
        self.clookup = nn.Embedding(len(cpos) + 3, self.cdims)

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.hidLayerFOH = Parameter((self.ldims * 2, self.hidden_units))
        self.hidLayerFOM = Parameter((self.ldims * 2, self.hidden_units))
        self.hidBias = Parameter((self.hidden_units))
        self.catBias = Parameter((self.hidden_units * 2))
        self.rhidLayerFOH = Parameter((2 * self.ldims, self.hidden_units))
        self.rhidLayerFOM = Parameter((2 * self.ldims, self.hidden_units))
        self.rhidBias = Parameter((self.hidden_units))
        self.rcatBias = Parameter((self.hidden_units * 2))
        #
        if self.hidden2_units:
            self.hid2Layer = Parameter(
                (self.hidden_units * 2, self.hidden2_units))
            self.hid2Bias = Parameter((self.hidden2_units))
            self.rhid2Layer = Parameter(
                (self.hidden_units * 2, self.hidden2_units))
            self.rhid2Bias = Parameter((self.hidden2_units))

        self.outLayer = Parameter(
            (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, 1))
        self.outBias = Parameter(1)
        self.routLayer = Parameter(
            (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, len(self.rel_list)))
        self.routBias = Parameter((len(self.rel_list)))

    def init_hidden(self, dim):
        return (autograd.Variable(torch.zeros(1, 1, dim).cuda() if use_gpu else torch.zeros(1, 1, dim)),
                autograd.Variable(torch.zeros(1, 1, dim).cuda() if use_gpu else torch.zeros(1, 1, dim)))

    def __evaluate(self, sentence):

        head_vector = []
        for index in range(len(sentence)):
            concatenated_lstm = concatenate_tensors([sentence[index].lstms[0], sentence[index].lstms[1]])
            headfov = torch.mm(concatenated_lstm, self.hidLayerFOH)
            concatenated_lstm = concatenate_tensors([sentence[index].lstms[0], sentence[index].lstms[1]])
            modfov = torch.mm(concatenated_lstm, self.hidLayerFOM)
            sentence[index].headfov = headfov
            sentence[index].modfov = modfov
            head_vector.append([headfov, modfov])

        exprs = [[self.__getExpr(head_vector, i, j) for j in range(len(head_vector))] for i in range(len(head_vector))]

        scores = np.array([[get_data(output).numpy()[0, 0]
                            for output in exprsRow] for exprsRow in exprs])
        return scores, exprs

    def __getExpr(self, head_vector, i, j):
        headfov = head_vector[i][0]
        modfov = head_vector[j][1]
        if self.hidden2_units > 0:
            concatenated_result = concatenate_tensors([headfov, modfov])
            activation_result = self.activation(concatenated_result + self.catBias)
            mm_result = torch.mm(activation_result, self.hid2Layer)
            next_activation_result = self.activation(self.hid2Bias + mm_result)
            output = torch.mm(next_activation_result, self.outLayer) + self.outBias
        else:
            activation_result = self.activation(headfov + modfov + self.hidBias)
            output = torch.mm(activation_result, self.outLayer) + self.outBias

        return output

    def __evaluateLabel(self, rheadfov, rmodfov):

        if self.hidden2_units > 0:
            concatenated_result = concatenate_tensors([rheadfov, rmodfov])
            activation_result = self.activation(concatenated_result + self.rcatBias)
            matmul_result = torch.mm(activation_result, self.rhid2Layer)
            next_activation_result = self.activation(self.rhid2Bias + matmul_result)
            output = torch.mm(next_activation_result, self.routLayer) + self.routBias
        else:
            activation_result = self.activation(rheadfov + rmodfov + self.rhidBias)
            output = torch.mm(activation_result, self.routLayer) + self.routBias

        return get_data(output).numpy()[0], output[0]

    def predict(self, sentence):
        self.process_sentence_embeddings(sentence)

        num_vec = len(sentence)
        vec_for = torch.cat(
            [entry.vec for entry in sentence]).view(num_vec, 1, -1)
        vec_back = torch.cat(
            [entry.vec for entry in reversed(sentence)]).view(num_vec, 1, -1)
        res_for_1, self.hid_for_1 = self.lstm_for_1(vec_for, self.hid_for_1)
        res_back_1, self.hid_back_1 = self.lstm_back_1(
            vec_back, self.hid_back_1)

        vec_cat = [concatenate_tensors([res_for_1[i], res_back_1[num_vec - i - 1]])
                   for i in range(num_vec)]

        vec_for_2 = torch.cat(vec_cat).view(num_vec, 1, -1)
        vec_back_2 = torch.cat(list(reversed(vec_cat))).view(num_vec, 1, -1)
        res_for_2, self.hid_for_2 = self.lstm_for_2(vec_for_2, self.hid_for_2)
        res_back_2, self.hid_back_2 = self.lstm_back_2(
            vec_back_2, self.hid_back_2)

        for i in range(num_vec):
            sentence[i].lstms[0] = res_for_2[i]
            sentence[i].lstms[1] = res_back_2[num_vec - i - 1]

        scores, exprs = self.__evaluate(sentence, True)
        heads = decoder.parse_proj(scores)

        for entry, head in zip(sentence, heads):
            entry.pred_parent_id = head
            entry.pred_relation = '_'

        # TODO: Uncomment modifying with new __evaluateLabel arguments
        # head_list = list(heads)
        # for modifier, head in enumerate(head_list[1:]):
        #     scores, exprs = self.__evaluateLabel(
        #         sentence, head, modifier + 1)
        #     sentence[modifier + 1].pred_relation = self.rel_list[max(
        #         enumerate(scores), key=itemgetter(1))[0]]

    def forward(self, sentence):

        self.process_sentence_embeddings(sentence)

        num_vec = len(sentence)
        features_for = [entry.vec for entry in sentence]
        features_back = [entry.vec for entry in reversed(sentence)]
        vec_for = torch.cat(features_for).view(num_vec, 1, -1)
        vec_back = torch.cat(features_back).view(num_vec, 1, -1)
        res_for_1, self.hid_for_1 = self.lstm_for_1(vec_for, self.hid_for_1)
        res_back_1, self.hid_back_1 = self.lstm_back_1(vec_back, self.hid_back_1)

        vec_cat = [concatenate_tensors([res_for_1[i], res_back_1[num_vec - i - 1]])
                   for i in range(num_vec)]

        vec_for_2 = torch.cat(vec_cat).view(num_vec, 1, -1)
        vec_back_2 = torch.cat(list(reversed(vec_cat))).view(num_vec, 1, -1)
        res_for_2, self.hid_for_2 = self.lstm_for_2(vec_for_2, self.hid_for_2)
        res_back_2, self.hid_back_2 = self.lstm_back_2(vec_back_2, self.hid_back_2)

        for i in range(num_vec):
            sentence[i].lstms[0] = res_for_2[i]
            sentence[i].lstms[1] = res_back_2[num_vec - i - 1]

        scores, exprs = self.__evaluate(sentence)
        gold = [entry.parent_id for entry in sentence]
        heads = decoder.parse_proj(scores, gold)

        lerrs = []
        for modifier, head in enumerate(gold[1:]):

            if sentence[head].rheadfov is None:
                sentence[head].rheadfov = torch.mm(concatenate_tensors([sentence[head].lstms[0],
                                                                        sentence[head].lstms[1]]),
                                                   self.rhidLayerFOH)

            if sentence[modifier + 1].rmodfov is None:
                sentence[modifier + 1].rmodfov = torch.mm(concatenate_tensors([sentence[modifier + 1].lstms[0],
                                                                               sentence[modifier + 1].lstms[1]]),
                                                          self.rhidLayerFOM)

            rscores, rexprs = self.__evaluateLabel(sentence[head].rheadfov, sentence[modifier + 1].rmodfov)
            goldLabelInd = self.rels[sentence[modifier + 1].relation]
            wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
            if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                lerrs += [rexprs[wrongLabelInd] - rexprs[goldLabelInd]]

        e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
        errs = []
        if e > 0:
            errs += [(exprs[h][i] - exprs[g][i])[0]
                     for i, (h, g) in enumerate(zip(heads, gold)) if h != g]
        return e, errs, lerrs

    def process_sentence_embeddings(self, sentence):
        for entry in sentence:
            c = float(self.wordsCount.get(entry.norm, 0))
            dropFlag = (random.random() < (c / (0.25 + c)))
            w_index = int(self.vocab.get(entry.norm, 0)) if dropFlag else 0
            wordvec = self.wlookup(scalar(w_index)) if self.wdims > 0 else None

            entry.vec = wordvec
            entry.lstms = [entry.vec, entry.vec]
            entry.headfov = None
            entry.modfov = None

            entry.rheadfov = None
            entry.rmodfov = None


def get_optim(opt, parameters):
    if opt.optim == 'sgd':
        return optim.SGD(parameters, lr=opt.lr)
    elif opt.optim == 'adam':
        return optim.Adam(parameters, lr=opt.lr)


class MSTParserLSTM:
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos):
        model = MSTParserLSTMModel(
            vocab, pos, rels, enum_word, options, onto, cpos)
        self.model = model.cuda() if use_gpu else model
        self.trainer = get_optim(options, self.model.parameters())

    def predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                self.model.hid_for_1, self.model.hid_back_1, self.model.hid_for_2, self.model.hid_back_2 = [
                    self.model.init_hidden(self.model.ldims) for _ in range(4)]
                conll_sentence = [entry for entry in sentence if isinstance(
                    entry, utils.ConllEntry)]
                self.model.predict(conll_sentence)
                yield conll_sentence

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))

    def train(self, conll_path):
        print('pytorch version:', torch.__version__)
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
                # print("Initializing hidden and cell states values to 0")
                self.model.hid_for_1, self.model.hid_back_1, self.model.hid_for_2, self.model.hid_back_2 = [
                    self.model.init_hidden(self.model.ldims) for _ in range(4)]
                # if iSentence == 0:
                #     print('hidLayerFOM values on first iteration within an epoch')
                #     print(self.model.hidLayerFOM)
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
                    # print('hidLayerFOM values:')
                    # print(self.model.hidLayerFOM)

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                e_output, errs, lerrs = self.model.forward(conll_sentence)
                eerrors += e_output
                eloss += e_output
                mloss += e_output
                etotal += len(sentence)
                if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                    if len(errs) > 0 or len(lerrs) > 0:
                        reshaped_lerrs = [item.reshape(1) for item in lerrs]
                        l_variable = errs + reshaped_lerrs
                        eerrs_sum = torch.sum(concatenate_tensors(l_variable))  # This result is a 1d-tensor
                        eerrs_sum.backward()  # automatically calculates gradient (backpropagation)
                        # self.print_model_parameters()
                        self.trainer.step()  # optimizer.step to update weights(to see uncomment print_model_parameters)
                        # self.print_model_parameters()
                self.trainer.zero_grad()
        print("Loss: ", mloss / iSentence)

    def print_model_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
