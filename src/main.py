import os
import gc
import random
from collections import Counter
from tempfile import NamedTemporaryFile

import numpy as np

from bilstm.parse import parse_words, split_words_to_sentences, format_words
from bilstm.indexer import Indexer
from bilstm.tag_scheme import BILOU
from bilstm.tagger import BiLstmNerTagger


random.seed(1)


EVAL_NER_CMD = '/Users/konix/Workspace/nertagger/src/conlleval < {test_file}'

TRAIN_FILE_PATH = '/Users/konix/Workspace/nertagger/data/eng.train'
DEV_FILE_PATH = '/Users/konix/Workspace/nertagger/data/eng.testa'
TEST_FILE_PATH = '/Users/konix/Workspace/nertagger/data/eng.testb'


TAG_SCHEME = BILOU


def eval_ner(test_sentence_list):
    test_words = []
    for sentence in test_sentence_list:
        test_words.extend(sentence)

    with NamedTemporaryFile(mode='wb') as temp_file:
        format_words(temp_file, test_words, tag_scheme=TAG_SCHEME)
        temp_file.flush()
        os.system(EVAL_NER_CMD.format(test_file=temp_file.name))


def main():
    train_words = parse_words(open(TRAIN_FILE_PATH, 'rb'), tag_scheme=TAG_SCHEME)
    train_sentences = split_words_to_sentences(train_words)
    dev_words = parse_words(open(DEV_FILE_PATH, 'rb'), tag_scheme=TAG_SCHEME)
    dev_sentences = split_words_to_sentences(dev_words)
    test_words = parse_words(open(TEST_FILE_PATH, 'rb'), tag_scheme=TAG_SCHEME)
    test_sentences = split_words_to_sentences(test_words)

    external_word_embeddings = {}
    for line in open('/Users/konix/Documents/pos_data/glove.6B/glove.6B.100d.txt', 'rb').readlines():
        word, embedding_str = line.split(' ', 1)
        embedding = np.asarray([float(value_str) for value_str in embedding_str.split()])
        external_word_embeddings[word] = embedding

    word_list = []
    char_list = []
    tag_list = []
    for sentence_ in train_sentences:
        for word_ in sentence_:
            word_list.append(word_.text.lower())
            tag_list.append(word_.gold_label)
            char_list.extend(word_.text)

    word_counter = Counter(word_list)
    word_indexer = Indexer()
    word_indexer.index_object_list(
        [word_text for (word_text, word_count) in word_counter.iteritems() if word_count >= 4]
    )
    word_indexer.index_object_list(external_word_embeddings.keys())
    word_indexer.index_object('_UNK_')

    char_counter = Counter(char_list)
    char_indexer = Indexer()
    char_indexer.index_object_list(char_counter.keys())

    tag_counter = Counter(tag_list)
    tag_indexer = Indexer()
    tag_indexer.index_object_list(tag_counter.keys())
    tag_indexer.index_object('-START-')
    tag_indexer.index_object('-END-')

    tagger = BiLstmNerTagger(word_indexer, char_indexer, tag_indexer, external_word_embeddings)

    del word_list
    del char_list
    del tag_list
    del external_word_embeddings
    gc.collect()

    tagger.train(train_sentences, dev_sentences, test_sentences, eval_func=eval_ner, iterations=50)

    word_index = 0
    while word_index < len(dev_words):
        sentence = dev_words[word_index].sentence
        tagger.tag_sentence_viterbi(sentence)
        word_index += len(sentence)
    format_words(open('/tmp/dev_ner', 'wb'), dev_words, tag_scheme=TAG_SCHEME)

    word_index = 0
    while word_index < len(test_words):
        sentence = test_words[word_index].sentence
        tagger.tag_sentence_viterbi(sentence)
        word_index += len(sentence)
    format_words(open('/tmp/test_ner', 'wb'), test_words, tag_scheme=TAG_SCHEME)


if __name__ == '__main__':
    main()
