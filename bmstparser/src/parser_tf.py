import os
from optparse import OptionParser

import tensorflow as tf

import mstlstm_tf
import utils

if __name__ == '__main__':
    parser = OptionParser()

    training_phase = True  # False implies prediction phase

    parser.add_option("--outdir", type="string", dest="output", default="/model-tf")
    parser.add_option("--numthread", type="int", dest="numthread", default=8)

    if training_phase:
        parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
        parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                          default="/corpus/en-small-ud-train.conllu")
        parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE",
                          default="/corpus/en-small-ud-dev.conllu")

        parser.add_option("--multi", dest="multi", help="Annotated CONLL multi-train file", metavar="FILE",
                          default=False)
        # multi-task has been deleted for bloated code

        parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
        parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=0)
        parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)

        parser.add_option("--oembedding", type="int", dest="oembedding_dims", default=0) #ontology
        parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=0) #cpos

        parser.add_option("--epochs", type="int", dest="epochs", default=5)
        parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
        parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
        parser.add_option("--optim", type="string", dest="optim", default='adam')
        parser.add_option("--lr", type="float", dest="lr", default=1e-3)
        parser.add_option("--activation", type="string", dest="activation", default="tanh")
        parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
        parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)

        parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
        parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                          default="/model/neuralfirstorder.model")

    else:
        parser.add_option("--predict", action="store_true", dest="predictFlag", default=True)

        parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE",
                          default="/corpus/en-ud-test.conllu")

        parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE",
                          default="/model/params.pickle")
        parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                          default="/model/neuralfirstorder.model3")

    (options, args) = parser.parse_args()

    # TODO: Check if we can add operation parallelism on CPU with Tensorflow
    # Added to run from IntelliJ
    os.chdir("../../")
    print('Current directory: ' + os.getcwd())
    output_file = os.getcwd() + options.output
    model_path = os.getcwd() + options.model
    utils_path = os.getcwd() + '/bmstparser/src/utils/'  # 'src/utils/'
    # Added to run from IntelliJ

    # Training classifier
    # Added to run from IntelliJ
    train_file = os.getcwd() + options.conll_train
    dev_file = os.getcwd() + options.conll_dev
    # Added to run from IntelliJ

    print('Preparing vocabulary table')
    words, enum_word, pos, rels, onto, cpos = list(utils.vocab(train_file))
    # TODO: Check if pickle serialization is required
    print('Finished collecting vocabulary')

    print('Initializing mst-parser:')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.enable_eager_execution()
    parser = mstlstm_tf.MSTParserLSTM(words, pos, rels, enum_word, options, onto, cpos)
    for epoch in range(options.epochs):
        print('Starting epoch', epoch)
        parser.train(train_file)
        # parser.save(os.path.join(output_file, os.path.basename(model_path) + str(epoch + 1)))
        # parser.save("/home/dburbano/IdeaProjects/JSL/bist-parser-tensorflow/model-tf")
