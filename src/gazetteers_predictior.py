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


BASE_DIR = os. environ.get('LSTM_BASE_DIR')

CONLL_DIR = os.path.join(BASE_DIR, 'conll')
TRAIN_FILE_PATH = os.path.join(CONLL_DIR, 'eng.train')
DEV_FILE_PATH = os.path.join(CONLL_DIR, 'eng.testa')
TEST_FILE_PATH = os.path.join(CONLL_DIR, 'eng.testb')

# EMBEDDINGS_FILE_PATH = os.path.join(BASE_DIR, 'glove', 'glove.6B.100d.txt')
# WORDS_FILE_PATH = os.path.join(BASE_DIR, 'glove', 'glove.100d.words')
# VECTORS_FILE_PATH = os.path.join(BASE_DIR, 'glove', 'glove.100d.vectors')

# EMBEDDINGS_FILE_PATH = '/Users/konix/Documents/pos_data/glove.twitter.27B/glove.twitter.27B.100d.txt'

EMBEDDINGS_FILE_PATH = '/Users/konix/Workspace/GloVe-1.2/yelp_vectors.txt'
WORDS_FILE_PATH = "/Users/konix/Workspace/GloVe-1.2/yelp.50d.words"
VECTORS_FILE_PATH = "/Users/konix/Workspace/GloVe-1.2/yelp.50d.vectors"
COMMON_WORDS_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/yelp_common"
CLASSIFICATION_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/yelp_classification"

# EMBEDDINGS_FILE_PATH = '/Users/konix/Workspace/GloVe-1.2/amazon_vectors.txt'
# WORDS_FILE_PATH = "/Users/konix/Workspace/GloVe-1.2/amazon.50d.words"
# VECTORS_FILE_PATH = "/Users/konix/Workspace/GloVe-1.2/amazon.50d.vectors"
# COMMON_WORDS_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/amazon_common"
# CLASSIFICATION_FILE_PATH = "/Users/konix/Workspace/nerlstm/glove/amazon_classification"


GAZETTEERS_DIR_PATH = '/Users/konix/Workspace/nertagger/resources/gazetteers'

CLASS_TO_GAZETTEERS = {
    'nationality': {
        'KnownNationalities.txt',
        'known_nationalities.lst'
    },

    'job': {
        'known_jobs.lst',
        'Occupations.txt',
        'known_title.lst',
        'VincentNgPeopleTitles.txt'
    },

    'organization': {
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


NLTK_STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                   'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                   'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                   'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                   'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                   'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                   'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                   'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                   'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                   'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                   'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                   'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


class GazetteerClassifier(object):
    WORD_DIM = 50
    HIDDEN_DIM = 25

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
    for gazetteer_index, gazetteer_value in enumerate(gazetteer_vector):
        cur_diff = abs(gazetteer_value - 1)
        if min_diff is None or cur_diff< min_diff:
            min_diff = cur_diff
            min_index = gazetteer_index
    return gazetteer_indexer.get_object(min_index)


def get_gazetteer_scores(my_clf, word_indexer, gazetteer_indexer, word_text):
    word_index = word_indexer.get_index(word_text)
    gazetteer_vector = my_clf.build_expression(word_index).npvalue()
    score_by_gazetteer = {}
    for gazetteer_index, gazetteer_value in enumerate(gazetteer_vector):
        gazetteer_name = gazetteer_indexer.get_object(gazetteer_index)
        score_by_gazetteer[gazetteer_name] = gazetteer_value
    return score_by_gazetteer


def predict_all_words(my_clf, E):
    H = np.tanh(np.dot(E, np.transpose(my_clf.param_hidden.as_array())))
    return np.dot(H, np.transpose(my_clf.param_out.as_array()))


def main():
    # train_words = parse_words(open(TRAIN_FILE_PATH, 'rb'))
    # dev_words = parse_words(open(DEV_FILE_PATH, 'rb'))
    # test_words = parse_words(open(TEST_FILE_PATH, 'rb'))
    #
    # train_word_set = set([word_.text.lower() for word_ in train_words])
    # dev_and_test_word_set = set([word_.text.lower() for word_ in chain(dev_words, test_words)])
    # dataset_words_set = train_word_set.union(dev_and_test_word_set)

    external_word_embeddings = {}
    for line in open(EMBEDDINGS_FILE_PATH, 'rb').readlines():
        word, embedding_str = line.rstrip().split(' ', 1)
        word = word.lower()
        embedding = np.asarray([float(value_str) for value_str in embedding_str.split()])
        external_word_embeddings[word] = embedding

    pos_words = open('/Users/konix/positive.txt', 'rb').read().split('\n')[:-1]
    pos_words = [word_text_ for word_text_ in pos_words if word_text_ in external_word_embeddings]
    random.shuffle(pos_words)
    cufoff = int(len(pos_words) * 1)
    pos_words_train, pos_words_test = pos_words[:cufoff], pos_words[cufoff:]

    neg_words = open('/Users/konix/negative.txt', 'rb').read().split('\n')[:-1]
    neg_words = [word_text_ for word_text_ in neg_words if word_text_ in external_word_embeddings]
    random.shuffle(neg_words)
    cufoff = int(len(neg_words) * 1)
    neg_words_train, neg_words_test = neg_words[:cufoff], neg_words[cufoff:]

    word_to_sentiment = {}
    # pos_also_neg = set()
    for word in pos_words_train:
        word_to_sentiment[word] = ['positive']
        # if random.random() < 0:
        #     word_to_sentiment[word].append('negative')
        #     pos_also_neg.add(word)
    # neg_also_pos = set()
    for word in neg_words_train:
        word_to_sentiment[word] = ['negative']
        # if random.random() < 0:
        #     word_to_sentiment[word].append('positive')
        #     neg_also_pos.add(word)

    for word_text in NLTK_STOP_WORDS:
        if word_text in external_word_embeddings:
            word_to_sentiment[word_text] = []

    common_words = open(COMMON_WORDS_FILE_PATH, 'rb').read().split('\n')
    for word_text in common_words:
        if word_text in external_word_embeddings and word_text not in word_to_sentiment:
            word_to_sentiment[word_text] = []

    sentiment_indexer = Indexer()
    sentiment_indexer.index_object_list(['positive', 'negative'])

    word_indexer = Indexer()
    # word_indexer.index_object_list(dataset_words_set)
    word_indexer.index_object_list(external_word_embeddings.keys())

    # gazetteers_annotator = create_gazetteers_annotator(GAZETTEERS_DIR_PATH)
    # gazetteers_annotator.annotate_data(train_words)
    # gazetteers_annotator.annotate_data(dev_words)
    # gazetteers_annotator.annotate_data(test_words)
    #
    # gazetteer_to_class = {}
    # for gazetteer_class, class_gazetteer_set in CLASS_TO_GAZETTEERS.iteritems():
    #     for gazetteer in class_gazetteer_set:
    #         gazetteer_to_class[gazetteer] = gazetteer_class
    #
    # word_to_gazetteers = defaultdict(set)
    # for word in chain(train_words, dev_words, test_words):
    #     if word.gazetteers:
    #         word_gazetteer_names = [gazetteer_part[2:] for gazetteer_part in word.gazetteers]
    #         word_gazetteer_classes = set([gazetteer_to_class[gazetteer_] for gazetteer_ in word_gazetteer_names])
    #         word_to_gazetteers[word.text.lower()].update(word_gazetteer_classes)
    #
    # gazetteer_indexer = Indexer()
    # gazetteer_indexer.index_object_list(CLASS_TO_GAZETTEERS.keys())
    #
    # train_word_count = Counter([y_.text.lower() for y_ in train_words])
    # common_train_words = [word_text_ for word_text_, word_count_ in train_word_count.iteritems() if word_count_ >= 10]
    # ext_train_words = [word_text_ for word_text_ in common_train_words if word_text_ not in word_to_gazetteers]
    # for word_text_ in ext_train_words:
    #     word_to_gazetteers[word_text_] = {'ext'}
    #
    # # make sure ext is used only for N/A
    # for word_text, word_gazetteers in word_to_gazetteers.iteritems():
    #     if 'ext' in word_gazetteers and len(word_gazetteers) > 1:
    #         word_gazetteers.remove('ext')
    #
    # for word_text in NLTK_STOP_WORDS:
    #     if word_text in external_word_embeddings:
    #         word_to_gazetteers[word_text] = {'ext'}
    #
    # num_words = len(word_to_gazetteers)
    # train_to_gazetteer = dict(word_to_gazetteers.items()[:int(0.9*num_words)])
    # test_to_gazetteer = dict(word_to_gazetteers.items()[int(0.9*num_words):])
    #
    # my_clf = GazetteerClassifier(word_indexer, gazetteer_indexer, external_word_embeddings)
    # my_clf.train(word_to_gazetteers, iterations=1500)

    my_clf = GazetteerClassifier(word_indexer, sentiment_indexer, external_word_embeddings)
    my_clf.train(word_to_sentiment, iterations=1000)

    # my_find_gazetteer = partial(find_gazetteer, my_clf, word_indexer, gazetteer_indexer)
    # my_get_gazetteer_scores = partial(get_gazetteer_scores, my_clf, word_indexer, gazetteer_indexer)
    my_find_sentiment = partial(find_gazetteer, my_clf, word_indexer, sentiment_indexer)
    my_get_sentiment_scores = partial(get_gazetteer_scores, my_clf, word_indexer, sentiment_indexer)

    W = np.array(file(WORDS_FILE_PATH).read().strip().split())
    w2i = {w_: i_ for (i_, w_) in enumerate(W)}
    E = np.loadtxt(VECTORS_FILE_PATH)

    prediction_matrix = predict_all_words(my_clf, E)

    word_by_sentiment = {'positive': [], 'negative': []}
    for w, i in w2i.iteritems():
        pos_score, neg_score = prediction_matrix[i]
        if neg_score >= 0.5 and (pos_score < 0.1 or ((neg_score - pos_score) > 0.5)):
            word_by_sentiment['negative'].append(w)
        elif pos_score >= 0.5 and (neg_score < 0.1 or ((pos_score - neg_score) > 0.5)):
            word_by_sentiment['positive'].append(w)

    with open(CLASSIFICATION_FILE_PATH, 'wb') as classification_file:
        json.dump(word_by_sentiment, classification_file)

    import IPython;IPython.embed()

    # print "Calculating class and vector for each word"
    # word_to_class = {}
    # word_to_class_scores = {}
    # for word in external_word_embeddings.iterkeys():
    #     if word in dataset_words_set:
    #         word_to_class[word] = my_find_gazetteer(word)
    #         word_to_class_scores[word] = my_get_gazetteer_scores(word)
    #
    # print "Saving results to file"
    # with open('/tmp/word_to_class.json', 'wb') as result_file:
    #     json.dump(word_to_class, result_file)
    # with open('/tmp/word_to_class_scores.json', 'wb') as result_file:
    #     json.dump(word_to_class_scores, result_file)


if __name__ == '__main__':
    main()
