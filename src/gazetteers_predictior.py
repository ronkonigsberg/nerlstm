import os
import random
from itertools import chain
from functools import partial
from collections import defaultdict

import numpy as np

from pycnn import (Model, AdamTrainer, LSTMBuilder, renew_cg, lookup, dropout, parameter, concatenate,
                   softmax, pickneglogsoftmax, esum, log, exp, tanh, squared_distance, vecInput)

from bilstm.indexer import Indexer
from bilstm.parse import parse_words
from bilstm.gazetteers import create_gazetteers_annotator


BASE_DIR = os.environ.get('LSTM_BASE_DIR')

CONLL_DIR = os.path.join(BASE_DIR, 'conll')
TRAIN_FILE_PATH = os.path.join(CONLL_DIR, 'eng.train')
DEV_FILE_PATH = os.path.join(CONLL_DIR, 'eng.testa')
TEST_FILE_PATH = os.path.join(CONLL_DIR, 'eng.testb')

EMBEDDINGS_FILE_PATH = os.path.join(BASE_DIR, 'glove', 'glove.6B.100d.txt')
GAZETTEERS_DIR_PATH = '/Users/konix/Workspace/nertagger/resources/gazetteers'


class GazetteerClassifier(object):
    WORD_DIM = 100
    HIDDEN_DIM = 50

    def __init__(self, word_indexer, gazetteer_indexer, external_word_embeddings=None):
        self.word_indexer = word_indexer
        self.gazetteer_indexer = gazetteer_indexer
        self.external_word_embeddings = external_word_embeddings

        model = Model()
        model.add_lookup_parameters("word_lookup", (len(word_indexer), self.WORD_DIM))

        if external_word_embeddings:
            word_lookup = model["word_lookup"]
            for idx in xrange(len(word_indexer)):
                word = word_indexer.get_object(idx)
                if word in external_word_embeddings:
                    word_lookup.init_row(idx, external_word_embeddings[word])

        self.param_hidden = model.add_parameters("HIDDEN", (self.HIDDEN_DIM, self.WORD_DIM))
        self.param_out = model.add_parameters("OUT", (len(gazetteer_indexer), self.HIDDEN_DIM))

        self.model = model
        self.trainer = AdamTrainer(model)

    def train(self, train_word_to_gazetteers, test_word_to_gazetteers=None, iterations=50):
        training_examples = self.build_example_vectors(train_word_to_gazetteers)

        for iteration_idx in xrange(1, iterations+1):
            print "Starting training iteration %d/%d" % (iteration_idx, iterations)
            random.shuffle(training_examples)
            loss = 0

            for example_index, (word_index, expected_output) in enumerate(training_examples, 1):
                out_expression = self.build_expression(word_index)

                expected_output_expr = vecInput(len(self.gazetteer_indexer))
                expected_output_expr.set(expected_output)
                sentence_error = squared_distance(out_expression, expected_output_expr)

                loss += sentence_error.scalar_value()
                sentence_error.backward()
                self.trainer.update()

            # Trainer Status
            self.trainer.status()
            print loss / float(len(training_examples))

    def build_example_vectors(self, word_to_gazetteers):
        examples = []
        for word, word_gazetteers in word_to_gazetteers.iteritems():
            word_index = self.word_indexer.get_index(word)
            word_gazetteers_indices = [1 if self.gazetteer_indexer.get_object(gazetteer_idx) in word_gazetteers else 0
                                       for gazetteer_idx in xrange(len(self.gazetteer_indexer))]
            examples.append((word_index, np.asarray(word_gazetteers_indices)))
        return examples

    def build_expression(self, word_index):
        renew_cg()

        H = parameter(self.param_hidden)
        O = parameter(self.param_out)

        word_vector = lookup(self.model["word_lookup"], word_index, False)
        return O * tanh(H * word_vector)


def parse_gazetteers_directory(gazetteers_dir_path):
    gazetteers = dict()
    lowercase_gazetteers = dict()

    for gazetteer_name in os.listdir(gazetteers_dir_path):
        with open(os.path.join(gazetteers_dir_path, gazetteer_name), 'rb') as gazetteer_file:
            gazetteer_data = gazetteer_file.read()

        gazetteer_expressions = set()
        gazetteer_expressions_lowercase = set()
        for record in gazetteer_data.splitlines():
            gazetteer_expressions.add(record)
            gazetteer_expressions_lowercase.add(record.lower())

        gazetteers[gazetteer_name] = gazetteer_expressions
        lowercase_gazetteers[gazetteer_name] = gazetteer_expressions_lowercase
    return gazetteers, lowercase_gazetteers


def find_gazetteer(my_clf, word_indexer, gazetteer_indexer, word_text):
    word_index = word_indexer.get_index(word_text)
    gazetteer_vector = my_clf.build_expression(word_index).npvalue()
    min_index, min_diff = None, None
    for gazetteer_index, gazetter_value in enumerate(gazetteer_vector):
        cur_diff = abs(gazetter_value - 1)
        if min_diff is None or cur_diff< min_diff:
            min_diff = cur_diff
            min_index = gazetteer_index
    return gazetteer_indexer.get_object(min_index)


def main():
    train_words = parse_words(open(TRAIN_FILE_PATH, 'rb'))
    dev_words = parse_words(open(DEV_FILE_PATH, 'rb'))
    test_words = parse_words(open(TEST_FILE_PATH, 'rb'))

    train_word_set = set([word_.text.lower() for word_ in train_words])
    dev_and_test_word_set = set([word_.text.lower() for word_ in chain(dev_words, test_words)])
    dataset_words_set = train_word_set.union(dev_and_test_word_set)

    external_word_embeddings = {}
    for line in open(EMBEDDINGS_FILE_PATH, 'rb').readlines():
        word, embedding_str = line.rstrip().split(' ', 1)
        word = word.lower()
        embedding = np.asarray([float(value_str) for value_str in embedding_str.split()])
        external_word_embeddings[word] = embedding

    word_indexer = Indexer()
    word_indexer.index_object_list(dataset_words_set)
    word_indexer.index_object_list(external_word_embeddings.keys())

    gazetteers_annotator = create_gazetteers_annotator(GAZETTEERS_DIR_PATH)
    gazetteers_annotator.annotate_data(train_words)
    gazetteers_annotator.annotate_data(dev_words)
    gazetteers_annotator.annotate_data(test_words)

    gazetteers_names = set()
    word_to_gazetteers = defaultdict(set)
    for word in chain(train_words, dev_words, test_words):
        if word.gazetteers:
            word_gazetteer_names = [gazetteer_part[2:] for gazetteer_part in word.gazetteers]
            word_to_gazetteers[word.text.lower()].update(word_gazetteer_names)
            gazetteers_names.update(word_gazetteer_names)

    gazetteer_indexer = Indexer()
    gazetteer_indexer.index_object_list(gazetteers_names)

    num_words = len(word_to_gazetteers)
    train_to_gazetteer = dict(word_to_gazetteers.items()[:int(0.9*num_words)])
    test_to_gazetteer = dict(word_to_gazetteers.items()[int(0.9*num_words):])

    my_clf = GazetteerClassifier(word_indexer, gazetteer_indexer, external_word_embeddings)
    my_clf.train(word_to_gazetteers, iterations=500)

    my_find_gazetteer = partial(find_gazetteer, my_clf, word_indexer, gazetteer_indexer)
    import IPython;IPython.embed()


if __name__ == '__main__':
    main()
