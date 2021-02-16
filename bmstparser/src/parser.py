import os.path
import pickle
import time
from optparse import OptionParser

import mstlstm
import utils


def evaluate_model():
    conllu = (os.path.splitext(dev_file.lower())[1] == '.conllu')
    devpath = os.path.join(output_file,
                           'dev_epoch_' + str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
    utils.write_conll(devpath, parser.predict(dev_file))

    if not conllu:
        perl_command = 'perl ' + utils_path + '/eval.pl -g ' + dev_file + ' -s ' + devpath + ' > ' \
                       + devpath + '.txt'
        print(perl_command)
        os.system(perl_command)
        with open(devpath + '.txt', 'r') as f:
            for i in range(0, 3):
                print(f.readline())
    else:
        python_command = 'python3 ' + utils_path + 'evaluation_script/conll17_ud_eval.py -v -w ' + \
                         utils_path + 'evaluation_script/weights.clas ' + dev_file + ' ' + devpath + ' > ' \
                         + devpath + '.txt'
        print(python_command)
        os.system(python_command)
        # time.sleep(60)
        # with open(devpath + '.txt', 'r') as f:
        #     for l in f:
        #         if l.startswith('UAS'):
        #             print('UAS:%s' % l.strip().split()[-1])
        #         elif l.startswith('LAS'):
        #             print('LAS:%s' % l.strip().split()[-1])


if __name__ == '__main__':
    parser = OptionParser()

    training_phase = True  # False implies prediction phase

    parser.add_option("--outdir", type="string", dest="output", default="/model")

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
        parser.add_option("--lr", type="float", dest="lr", default=0.1)
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
    # TODO: Uncomment multiprocess when not debugging
    # max_thread = multiprocessing.cpu_count()
    # active_thread = options.numthread if max_thread > options.numthread else max_thread
    # torch.set_num_threads(active_thread)
    # print(active_thread, "threads are in use")

    # Added to run from IntelliJ
    os.chdir("../../")
    print('Current directory: ' + os.getcwd())
    output_file = os.getcwd() + options.output
    model_path = os.getcwd() + options.model
    utils_path = os.getcwd() + '/bmstparser/src/utils/'  # 'src/utils/'
    # Added to run from IntelliJ

    if options.predictFlag:
        # Added to run from IntelliJ
        test_file = os.getcwd() + options.conll_test
        params_file = os.getcwd() + options.params
        # Added to run from IntelliJ

        with open(params_file, 'rb') as paramsfp:
            words, enum_word, pos, rels, onto, cpos, stored_opt = pickle.load(paramsfp)

        print('Initializing lstm mstparser:')
        parser = mstlstm.MSTParserLSTM(words, pos, rels, enum_word, stored_opt, onto, cpos)
        parser.load(model_path)
        conllu = (os.path.splitext(test_file.lower())[1] == '.conllu')
        testpath = os.path.join(output_file, 'test_pred.conll' if not conllu else 'test_pred.conllu')

        ts = time.time()
        test_res = list(parser.predict(test_file))
        te = time.time()
        print('Finished predicting test.', te - ts, 'seconds.')
        utils.write_conll(testpath, test_res)

        if not conllu:
            os.system('perl ' + utils_path + 'eval.pl -g ' + test_file + ' -s ' + testpath + ' > ' + testpath + '.txt')
        else:
            python_command = 'python3 ' + utils_path + 'evaluation_script/conll17_ud_eval.py -v -w ' + utils_path + \
                             'evaluation_script/weights.clas ' + test_file + ' ' + testpath + ' > ' + testpath + '.txt'
            print(python_command)
            os.system(python_command)
            with open(testpath + '.txt', 'r') as f:
                for l in f:
                    if l.startswith('UAS'):
                        print('UAS:%s' % l.strip().split()[-1])
                    elif l.startswith('LAS'):
                        print('LAS:%s' % l.strip().split()[-1])
    else:
        # Training classifier
        # Added to run from IntelliJ
        train_file = os.getcwd() + options.conll_train
        dev_file = os.getcwd() + options.conll_dev
        # Added to run from IntelliJ

        print('Preparing vocabulary table')

        words, enum_word, pos, rels, onto, cpos = list(utils.vocab(train_file))
        with open(os.path.join(output_file, options.params), 'wb') as paramsfp:
            pickle.dump((words, enum_word, pos, rels, onto, cpos, options), paramsfp)
        print('Finished collecting vocabulary')

        print('Initializing mst-parser:')
        parser = mstlstm.MSTParserLSTM(words, pos, rels, enum_word, options, onto, cpos)
        for epoch in range(options.epochs):
            print('Starting epoch', epoch)
            parser.train(train_file)
            parser.save(os.path.join(output_file, os.path.basename(model_path) + str(epoch + 1)))
            # evaluate_model()


