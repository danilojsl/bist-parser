import os
import shutil
from optparse import OptionParser
from os import path

import tensorflow as tf

import utils
import mstlstm_keras
# import mstlstm_tf

if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("--outdir", type="string", dest="output", default="/model-light-small-tf")

    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                      default="/corpus/en-small-ud-train.conllu")

    # multi-task has been deleted for bloated code

    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)

    parser.add_option("--epochs", type="int", dest="epochs", default=3)
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
    # Added to run from IntelliJ

    print('Preparing vocabulary table')
    words, enum_word, pos, rels, onto, cpos = list(utils.vocab(train_file))
    print('Finished collecting vocabulary')

    print('Initializing mst-parser with Stateless LSTM:')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    parser = mstlstm_keras.MSTParserLSTM(words, rels, enum_word, options)
    for epoch in range(options.epochs):
        print('Starting epoch', epoch)
        parser.train(train_file)

        print('Saving model...')
        base_name = output_path + '/' + model_name
        # parser.save(base_name, str(epoch + 1))
        # parser.save_light(base_name, str(epoch + 1))
