import os
import random
from itertools import chain
from functools import partial
from collections import Counter, defaultdict

import json
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

CLASS_TO_GAZETTEERS = {
    'nationality': {
        'KnownNationalities.txt',
        'known_nationalities.lst'
    },

    'job': {
        'known_jobs.lst',
        'known_title.lst',
        'Occupations.txt',
        'VincentNgPeopleTitles.txt'
    },

    'company': {
        'known_corporations.lst',
        'WikiOrganizations.lst',
        'WikiOrganizationsRedirects.lst'
    },

    'place': {
        'known_place.lst',
        'known_state.lst',
        'known_country.lst',
        'WikiLocations.lst',
        'WikiLocationsRedirects.lst'
    },

    'person': {
        'known_name.lst',
        'known_names.big.lst',
        'WikiPeople.lst',
        'WikiPeopleRedirects.lst'
    },

    'object_name': {
        'WikiArtWork.lst',
        'WikiArtWorkRedirects.lst',
        'WikiManMadeObjectNames.lst',
        'WikiManMadeObjectNamesRedirects.lst',
        'WikiCompetitionsBattlesEvents.lst',
        'WikiCompetitionsBattlesEventsRedirects.lst'
    },

    'ext': {
        'WikiFilms.lst',
        'WikiFilmsRedirects.lst',
        'WikiSongs.lst',
        'WikiSongsRedirects.lst',

        'measurments.txt',
        'ordinalNumber.txt',
        'temporal_words.txt',
        'cardinalNumber.txt',
        'currencyFinal.txt'
    }
}




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

    gazetteer_to_class = {}
    for gazetteer_class, class_gazetteer_set in CLASS_TO_GAZETTEERS.iteritems():
        for gazetteer in class_gazetteer_set:
            gazetteer_to_class[gazetteer] = gazetteer_class

    word_to_gazetteers = defaultdict(set)
    for word in chain(train_words, dev_words, test_words):
        if word.gazetteers:
            word_gazetteer_names = [gazetteer_part[2:] for gazetteer_part in word.gazetteers]
            word_gazetteer_classes = set([gazetteer_to_class[gazetteer_] for gazetteer_ in word_gazetteer_names])
            word_to_gazetteers[word.text.lower()].update(word_gazetteer_classes)

    gazetteer_indexer = Indexer()
    gazetteer_indexer.index_object_list(CLASS_TO_GAZETTEERS.keys())

    train_word_count = Counter([y_.text.lower() for y_ in train_words])
    common_train_words = [word_text_ for word_text_, word_count_ in train_word_count.iteritems() if word_count_ >= 10]
    ext_train_words = [word_text_ for word_text_ in common_train_words if word_text_ not in word_to_gazetteers]
    for word_text_ in ext_train_words:
        word_to_gazetteers[word_text_] = {'ext'}

    num_words = len(word_to_gazetteers)
    train_to_gazetteer = dict(word_to_gazetteers.items()[:int(0.9*num_words)])
    test_to_gazetteer = dict(word_to_gazetteers.items()[int(0.9*num_words):])

    my_clf = GazetteerClassifier(word_indexer, gazetteer_indexer, external_word_embeddings)
    my_clf.train(train_to_gazetteer, iterations=1000)

    my_find_gazetteer = partial(find_gazetteer, my_clf, word_indexer, gazetteer_indexer)

    print "Calculating class for each word"
    word_to_class = {}
    for word in external_word_embeddings.iterkeys():
        if word in dataset_words_set:
            word_to_class[word] = my_find_gazetteer(word)

    print "Saving results to file"
    with open('/tmp/word_to_class.json', 'wb') as result_file:
        json.dump(word_to_class, result_file)


if __name__ == '__main__':
    main()