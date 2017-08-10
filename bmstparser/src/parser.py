from optparse import OptionParser
import pickle
import utils
import mstlstm
import os
import os.path
import time
import torch
import multiprocessing


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--outdir", type="string",
                      dest="output", default="model")

    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                      default="corpus/train.conll")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE",
                      default="corpus/dev.conll")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE",
                      default="corpus/test.conll")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file",
                      metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="model/neuralfirstorder.model")

    parser.add_option("--multi", dest="multi", help="Annotated CONLL multi-train file", metavar="FILE",
                      default=False)
                      # multi-task has been deleted for bloated code

    parser.add_option("--wembedding", type="int",
                      dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int",
                      dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int",
                      dest="rembedding_dims", default=25)
    parser.add_option("--oembedding", type="int", dest="oembedding_dims", default=45) #ontology
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=30) #cpos
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--numthread", type="int", dest="numthread", default=8)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="lr", default=0.1)
    parser.add_option("--activation", type="string",
                      dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int",
                      dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)
    parser.add_option("--predict", action="store_true",
                      dest="predictFlag", default=False)

    (options, args) = parser.parse_args()
    max_thread = multiprocessing.cpu_count()
    active_thread = options.numthread if max_thread>options.numthread else max_thread
    torch.set_num_threads(active_thread)
    print(active_thread, "threads are in use")
    print('Using external embedding:', options.external_embedding)

    if options.predictFlag:
        with open(options.params, 'rb') as paramsfp:
            words, enum_word, pos, rels, onto, cpos, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = options.external_embedding

        print('Initializing lstm mstparser:')
        parser = mstlstm.MSTParserLSTM(words, pos, rels, enum_word, stored_opt, onto, cpos)
        parser.load(options.model)
        conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
        testpath = os.path.join(
            options.output, 'test_pred.conll' if not conllu else 'test_pred.conllu')

        ts = time.time()
        test_res = list(parser.predict(options.conll_test))
        te = time.time()
        print('Finished predicting test.', te - ts, 'seconds.')
        utils.write_conll(testpath, test_res)

        if not conllu:
            os.system('perl src/utils/eval.pl -g ' + options.conll_test +
                      ' -s ' + testpath + ' > ' + testpath + '.txt')
        else:
            os.system(
                'python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + options.conll_test + ' ' + testpath + ' > ' + testpath + '.txt')
            with open(testpath + '.txt', 'r') as f:
                for l in f:
                    if l.startswith('UAS'):
                        print('UAS:%s' % l.strip().split()[-1])
                    elif l.startswith('LAS'):
                        print('LAS:%s' % l.strip().split()[-1])
    else:
        print('Preparing vocabulary table')
        words, enum_word, pos, rels, onto, cpos = list(utils.vocab(options.conll_train))
        with open(os.path.join(options.output, options.params), 'wb') as paramsfp:
            pickle.dump((words, enum_word, pos, rels, onto, cpos, options), paramsfp)
        print('Finished collecting vocabulary')

        print('Initializing mst-parser:')
        parser = mstlstm.MSTParserLSTM(words, pos, rels, enum_word, options, onto, cpos)

        for epoch in range(options.epochs):
            print('Starting epoch', epoch)
            parser.train(options.conll_train)
            conllu = (os.path.splitext(
                options.conll_dev.lower())[1] == '.conllu')
            devpath = os.path.join(options.output,
                                   'dev_epoch_' + str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
            utils.write_conll(devpath, parser.predict(options.conll_dev))
            parser.save(os.path.join(options.output, os.path.basename(
                options.model) + str(epoch + 1)))

            if not conllu:
                os.system(
                    'perl src/utils/eval.pl -g ' + options.conll_dev + ' -s ' + devpath + ' > ' + devpath + '.txt')
                with open(devpath + '.txt', 'r') as f:
                    for i in range(0, 3):
                        print(f.readline())
            else:
                os.system(
                    'python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
                with open(devpath + '.txt', 'r') as f:
                    for l in f:
                        if l.startswith('UAS'):
                            print('UAS:%s' % l.strip().split()[-1])
                        elif l.startswith('LAS'):
                            print('LAS:%s' % l.strip().split()[-1])
