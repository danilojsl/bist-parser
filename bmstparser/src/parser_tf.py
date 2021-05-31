import os
import shutil
from optparse import OptionParser
from os import path

import tensorflow as tf

import utils
import mstlstm_tf
import mstlstm_predict_tf


def evaluate_model():
    print("Starting Validation...")
    conllu = (os.path.splitext(dev_file.lower())[1] == '.conllu')
    devpath = os.path.join(output_path, 'dev_epoch_' + str(epoch + 1) + '.conllu')
    utils.write_conll(devpath, mstlstm_predict_tf.predict(dev_file, options.wembedding_dims, weights_bi_lstm,
                                                          heads_variables, relations_variables))

    if not conllu:
        perl_command = 'perl ' + utils_path + '/eval.pl -g ' + dev_file + ' -s ' + devpath + ' > ' \
                       + devpath + '.txt'
        os.system(perl_command)
        with open(devpath + '.txt', 'r') as f:
            for i in range(0, 3):
                print(f.readline())
    else:
        python_command = 'python3 ' + utils_path + 'evaluation_script/conll17_ud_eval.py -v -w ' + \
                         utils_path + 'evaluation_script/weights.clas ' + dev_file + ' ' + devpath + ' > ' \
                         + devpath + '.txt'
        os.system(python_command)
        with open(devpath + '.txt', 'r') as f:
            for l in f:
                if l.startswith('UAS'):
                    print('UAS:%s' % l.strip().split()[-1])
                elif l.startswith('LAS'):
                    print('LAS:%s' % l.strip().split()[-1])


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("--outdir", type="string", dest="output", default="/model-small-tf")

    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                      default="/corpus/en-small-ud-train.conllu")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE",
                      default="/corpus/en-small-ud-dev.conllu")

    # multi-task has been deleted for bloated code

    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)

    parser.add_option("--epochs", type="int", dest="epochs", default=40)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="lr", default=1e-3)
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=126)

    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="dp-parser.model")

    (options, args) = parser.parse_args()

    # TODO: Check if we can add operation parallelism on CPU with Tensorflow
    # Added to run from IntelliJ
    os.chdir("../../")
    print('Current directory: ' + os.getcwd())
    output_path = os.getcwd() + options.output
    model_name = options.model
    utils_path = os.getcwd() + '/bmstparser/src/utils/'  # 'src/utils/'
    # Added to run from IntelliJ
    if path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    # Training classifier
    print(f'Training with file {options.conll_train}')
    # Added to run from IntelliJ
    train_file = os.getcwd() + options.conll_train
    dev_file = os.getcwd() + options.conll_dev
    # Added to run from IntelliJ

    print('Preparing vocabulary table')
    words, enum_word, pos, rels, onto, cpos = list(utils.vocab(train_file))
    print('Finished collecting vocabulary')

    print('Initializing mst-parser with Stateless LSTM:')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    parser = mstlstm_tf.MSTParserLSTM(words, rels, enum_word, options)
    for epoch in range(options.epochs):
        print('Starting epoch', epoch)
        parser.train(train_file)

        print('Saving model...')
        base_name = output_path + '/' + model_name
        parser.save_light(base_name, str(epoch + 1))
        weights_bi_lstm, heads_variables, relations_variables = parser.get_model_variables()
        evaluate_model()
