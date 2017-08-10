# A Pytorch implementation of the BIST Parsers (for graph based parser only)
To be more accurate, this implementation is just a line-by-line translation from the DyNet implementation that can be found [here](https://github.com/elikip/bist-parser). The techniques behind the parser are described in the paper [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198).

#### Required software

 * Python 3.x interpreter
 * [Pytorch library](http://pytorch.org/)


#### Data format:
The software requires having a `training.conll` and `development.conll` files formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat), or a `training.conllu` and `development.conllu` files formatted according to the [CoNLLU data format](http://universaldependencies.org/format.html).

#### Train a parsing model

    python src/parser.py --outdir [results directory] --train training.conll --dev development.conll --epochs 30 --lstmdims 125 --lstmlayers 2 [--extrn extrn.vectors] --bibi-lstm

#### Parse data with your parsing model

The command for parsing a `test.conll` file formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat) with a previously trained model is:

    python src/parser.py --predict --outdir [results directory] --test test.conll [--extrn extrn.vectors] --model [trained model file] --params [param file generate during training]

The parser will store the resulting conll file in the out directory (`--outdir`).

#### Difference from the DyNet implementation

1. The multiple roots checking of the evaluation script is turned off (See [here](https://github.com/wddabc/bist-parser/blob/pytorch/bmstparser/src/utils/evaluation_script/conll17_ud_eval.py#L168-L172)) as it might generate trees with multiple roots. (See the discussion [here](https://github.com/elikip/bist-parser/issues/10)) 
2. This version hasn't yet supported deep LSTM as the DyNet version does, which means `--lstmlayer` is no larger than 1.
