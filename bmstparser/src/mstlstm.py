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
        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

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
        self.outBias = 0  # Parameter(1)
        self.routLayer = Parameter(
            (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, len(self.rel_list)))
        self.routBias = Parameter((len(self.rel_list)))

    def init_hidden(self, dim):
        return (autograd.Variable(torch.zeros(1, 1, dim).cuda() if use_gpu else torch.zeros(1, 1, dim)),
                autograd.Variable(torch.zeros(1, 1, dim).cuda() if use_gpu else torch.zeros(1, 1, dim)))

    def __getExpr(self, sentence, i, j, train):

        if sentence[i].headfov is None:
            l_variable = [sentence[i].lstms[0], sentence[i].lstms[1]]
            sentence[i].headfov = torch.mm(concatenate_tensors(l_variable), self.hidLayerFOH)

        if sentence[j].modfov is None:
            l_variable = [sentence[j].lstms[0], sentence[j].lstms[1]]
            sentence[j].modfov = torch.mm(concatenate_tensors(l_variable),
                                          self.hidLayerFOM)

        if self.hidden2_units > 0:
            output = torch.mm(
                self.activation(
                    self.hid2Bias +
                    torch.mm(self.activation(concatenate_tensors([sentence[i].headfov, sentence[j].modfov]) + self.catBias),
                             self.hid2Layer)
                ),
                self.outLayer
            ) + self.outBias

        else:
            output = torch.mm(
                self.activation(
                    sentence[i].headfov + sentence[j].modfov + self.hidBias),
                self.outLayer) + self.outBias
        return output

    def __evaluate(self, sentence, train):
        exprs = [[self.__getExpr(sentence, i, j, train)
                  for j in range(len(sentence))]
                 for i in range(len(sentence))]
        scores = np.array([[get_data(output).numpy()[0, 0]
                            for output in exprsRow] for exprsRow in exprs])
        return scores, exprs

    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = torch.mm(concatenate_tensors([sentence[i].lstms[0], sentence[i].lstms[1]]),
                                            self.rhidLayerFOH)

        if sentence[j].rmodfov is None:
            sentence[j].rmodfov = torch.mm(concatenate_tensors([sentence[j].lstms[0], sentence[j].lstms[1]]),
                                           self.rhidLayerFOM)

        if self.hidden2_units > 0:
            output = torch.mm(
                self.activation(
                    self.rhid2Bias +
                    torch.mm(
                        self.activation(
                            concatenate_tensors([sentence[i].rheadfov, sentence[j].rmodfov]) + self.rcatBias),
                        self.rhid2Layer
                    )),
                self.routLayer
            ) + self.routBias

        else:
            output = torch.mm(
                self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias),
                self.routLayer
            ) + self.routBias

        return get_data(output).numpy()[0], output[0]

    def predict(self, sentence):
        for entry in sentence:
            wordvec = self.wlookup(
                scalar(int(self.vocab.get(entry.norm, 0)))) if self.wdims > 0 else None
            posvec = self.plookup(
                scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None
            ontovec = self.olookup(
                scalar(int(self.onto[entry.onto]))) if self.odims > 0 else None
            cposvec = self.clookup(
                scalar(int(self.cpos[entry.cpos]))) if self.cdims > 0 else None
            evec = None  # Used for external embeddings
            entry.vec = concatenate_tensors([wordvec, posvec, ontovec, cposvec, evec])

            entry.lstms = [entry.vec, entry.vec]
            entry.headfov = None
            entry.modfov = None

            entry.rheadfov = None
            entry.rmodfov = None

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

        head_list = list(heads)
        for modifier, head in enumerate(head_list[1:]):
            scores, exprs = self.__evaluateLabel(
                sentence, head, modifier + 1)
            sentence[modifier + 1].pred_relation = self.rel_list[max(
                enumerate(scores), key=itemgetter(1))[0]]

    def forward(self, sentence, errs, lerrs):
        for entry in sentence:
            c = float(self.wordsCount.get(entry.norm, 0))
            # dropFlag = (random.random() < (c / (0.33 + c)))
            dropFlag = (random.random() < (c / (0.25 + c)))
            wordvec = self.wlookup(scalar(
                int(self.vocab.get(entry.norm, 0)) if dropFlag else 0)) if self.wdims > 0 else None
            ontovec = self.olookup(scalar(int(self.onto[entry.onto]) if random.random(
            ) < 0.9 else 0)) if self.odims > 0 else None
            cposvec = self.clookup(scalar(int(self.cpos[entry.cpos]) if random.random(
            ) < 0.9 else 0)) if self.cdims > 0 else None
            posvec = self.plookup(
                scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None

            evec = None
            entry.vec = concatenate_tensors([wordvec, posvec, ontovec, cposvec, evec])
            entry.lstms = [entry.vec, entry.vec]
            entry.headfov = None
            entry.modfov = None

            entry.rheadfov = None
            entry.rmodfov = None

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

        scores, exprs = self.__evaluate(sentence, True)
        gold = [entry.parent_id for entry in sentence]
        heads = decoder.parse_proj(scores, gold)

        for modifier, head in enumerate(gold[1:]):
            rscores, rexprs = self.__evaluateLabel(sentence, head, modifier + 1)
            goldLabelInd = self.rels[sentence[modifier + 1].relation]
            wrongLabelInd = \
                max(((l, scr) for l, scr in enumerate(rscores)
                     if l != goldLabelInd), key=itemgetter(1))[0]
            if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                lerrs += [rexprs[wrongLabelInd] - rexprs[goldLabelInd]]

        e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
        if e > 0:
            errs += [(exprs[h][i] - exprs[g][i])[0]
                     for i, (h, g) in enumerate(zip(heads, gold)) if h != g]
        return e


def get_optim(opt, parameters):
    if opt.optim == 'sgd':
        return optim.SGD(parameters, lr=opt.learning_rate)
    elif opt.optim == 'adam':
        return optim.Adam(parameters)


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
            errs = []
            lerrs = []
            for iSentence, sentence in enumerate(shuffledData):
                # print("Initializing all values to 0")
                self.model.hid_for_1, self.model.hid_back_1, self.model.hid_for_2, self.model.hid_back_2 = [
                    self.model.init_hidden(self.model.ldims) for _ in range(4)]
                if iSentence % 100 == 0 and iSentence != 0:
                    print('Processing sentence number:', iSentence,
                          'Loss:', eloss / etotal,
                          'Errors:', (float(eerrors)) / etotal,
                          'Time', time.time() - start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(
                    entry, utils.ConllEntry)]
                e_output = self.model.forward(conll_sentence, errs, lerrs)
                # here the errs and lerrs variables have tensor values after the forward
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
                        errs = []
                        lerrs = []
                self.trainer.zero_grad()
        if len(errs) > 0:
            eerrs_sum = (torch.sum(errs + lerrs))
            eerrs_sum.backward()
            self.trainer.step()
        self.trainer.zero_grad()
        print("Loss: ", mloss / iSentence)

    def print_model_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
