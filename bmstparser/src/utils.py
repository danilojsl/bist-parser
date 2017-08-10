from collections import Counter
import re

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

class ConllEntry:
    def __init__(self, id, form, lemma, pos, cpos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.pos = pos
        self.cpos = cpos
        self.parent_id = parent_id
        self.relation = relation

        self.onto = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.onto, self.pos, self.cpos, self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def read_conll(conllFP):
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS',
                      'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    tokens = [root]
    for line in conllFP:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1:
                yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(
                    tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9]))
    if len(tokens) > 1:
        yield tokens


def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()
    ontoCount = Counter()
    cposCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            wordsCount.update(
                [node.norm for node in sentence if isinstance(node, ConllEntry)])
            posCount.update(
                [node.pos for node in sentence if isinstance(node, ConllEntry)])
            relCount.update(
                [node.relation for node in sentence if isinstance(node, ConllEntry)])
            ontoCount.update(
                [node.onto for node in sentence if isinstance(node, ConllEntry)])
            cposCount.update(
                [node.cpos for node in sentence if isinstance(node, ConllEntry)])

    print('the amount of kind of words, pos-tag, relations, ontology, cpos_tag:',
          len(wordsCount), len(posCount), len(relCount), len(ontoCount), len(cposCount))
    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, list(posCount.keys()), list(relCount.keys()), list(ontoCount.keys()), list(cposCount.keys()))


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')
