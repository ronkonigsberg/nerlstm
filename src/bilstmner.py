import random
import gc
from collections import Counter

import numpy as np
from pycnn import *

from nertagger.tag_scheme import BILOU, BIO

from parse import parse_words, split_words_to_sentences, format_words
from indexer import Indexer


TRAIN_FILE_PATH = '/Users/konix/Workspace/nertagger/data/eng.train'
DEV_FILE_PATH = '/Users/konix/Workspace/nertagger/data/eng.testa'
TEST_FILE_PATH = '/Users/konix/Workspace/nertagger/data/eng.testb'


TAG_SCHEME = BILOU


train_words = parse_words(open(TRAIN_FILE_PATH, 'rb'), tag_scheme=TAG_SCHEME)
train_sentences = split_words_to_sentences(train_words)
dev_words = parse_words(open(DEV_FILE_PATH, 'rb'), tag_scheme=TAG_SCHEME)
dev_sentences = split_words_to_sentences(dev_words)
test_words = parse_words(open(TEST_FILE_PATH, 'rb'), tag_scheme=TAG_SCHEME)
test_sentences = split_words_to_sentences(test_words)


word_list = []
tag_list = []
for sentence in train_sentences:
    for (word_text, ner_tag) in sentence:
        word_list.append(word_text)
        tag_list.append(ner_tag)


embedding_by_word = {}
for line in open('/Users/konix/Documents/pos_data/glove.6B/glove.6B.300d.txt', 'rb').readlines():
    word, embedding_str = line.split(' ', 1)
    embedding = np.asarray([float(value_str) for value_str in embedding_str.split()])
    embedding_by_word[word] = embedding


word_counter = Counter(word_list)
word_indexer = Indexer()
word_indexer.index_object_list([word_text for (word_text, word_count) in word_counter.iteritems() if word_count >= 5])
word_indexer.index_object_list(embedding_by_word.keys())
unk_word_index = word_indexer.index_object('_UNK_')

tag_counter = Counter(tag_list)
tag_indexer = Indexer()
tag_indexer.index_object_list(tag_counter.keys())
tag_indexer.index_object('_START_')


model = Model()
sgd = SimpleSGDTrainer(model)

model.add_lookup_parameters("lookup", (len(word_indexer), 300))
model.add_lookup_parameters("tl", (len(tag_indexer), 8))

# # My Code: initialize word lookup based on pre-trained embeddings
# embedding_by_word = {}
# for line in open('/Users/konix/Documents/pos_data/glove.6B/glove.6B.100d.txt', 'rb').readlines():
#     word, embedding_str = line.split(' ', 1)
#     embedding = np.asarray([float(value_str) for value_str in embedding_str.split()])
#     embedding_by_word[word] = embedding
word_lookup = model["lookup"]
for idx in xrange(len(word_indexer)):
    word = word_indexer.get_object(idx)
    if word in embedding_by_word:
        word_lookup.init_row(idx, embedding_by_word[word])
del embedding_by_word
gc.collect()

pH = model.add_parameters("HID", (32, 50*2))
pO = model.add_parameters("OUT", (len(tag_indexer), 32))

builders=[
        LSTMBuilder(1, 300, 50, model),
        LSTMBuilder(1, 300, 50, model),
        ]

def build_tagging_graph(words, tags, model, builders):
    renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]

    wembs = [lookup(model["lookup"], w) for w in words]
    wembs = [noise(we,0.1) for we in wembs]

    fw = [x.output() for x in f_init.add_inputs(wembs)]
    bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

    H = parameter(pH)
    O = parameter(pO)

    errs = []
    for f,b,t in zip(fw, reversed(bw), tags):
        f_b = concatenate([f,b])
        r_t = O*(tanh(H * f_b))
        err = pickneglogsoftmax(r_t, t)
        errs.append(err)
    return esum(errs)


def tag_sent(sent, model, builders):
    renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]
    wembs = [lookup(model["lookup"], (word_indexer.get_index(w) or unk_word_index)) for w, t in sent]

    fw = [x.output() for x in f_init.add_inputs(wembs)]
    bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

    H = parameter(pH)
    O = parameter(pO)

    tags=[]
    for f, b, (w, t) in zip(fw,reversed(bw),sent):
        r_t = O*(tanh(H * concatenate([f, b])))
        out = softmax(r_t)
        chosen = np.argmax(out.npvalue())
        tags.append(tag_indexer.get_object(chosen))
    return tags


tagged = loss = 0
for ITER in xrange(5):
    random.shuffle(train_sentences)
    for i,s in enumerate(train_sentences, 1):
        if i % 5000 == 0:
            sgd.status()
            print loss / tagged
            loss = 0
            tagged = 0
        if i % 10000 == 0:
            good = bad = 0.0
            for sent in dev_sentences:
                tags = tag_sent(sent, model, builders)
                golds = [t for w,t in sent]
                for go,gu in zip(golds,tags):
                    if go == gu: good +=1
                    else: bad+=1
            print good/(good+bad)
        ws = [(word_indexer.get_index(w) or unk_word_index) for w,p in s]
        ps = [tag_indexer.get_index(p) for w,p in s]
        sum_errs = build_tagging_graph(ws,ps,model,builders)
        squared = -sum_errs# * sum_errs
        loss += sum_errs.scalar_value()
        tagged += len(ps)
        sum_errs.backward()
        sgd.update()


word_index = 0
while word_index < len(dev_words):
    sentence = dev_words[word_index].sentence
    sentence_for_tagging = [(word_.text, None) for word_ in sentence]
    tags = tag_sent(sentence_for_tagging, model, builders)

    for word, tag in zip(sentence, tags):
        word.tag = tag

    word_index += len(sentence)
format_words(open('/tmp/dev_ner', 'wb'), dev_words, tag_scheme=TAG_SCHEME)

word_index = 0
while word_index < len(test_words):
    sentence = test_words[word_index].sentence
    sentence_for_tagging = [(word_.text, None) for word_ in sentence]
    tags = tag_sent(sentence_for_tagging, model, builders)

    for word, tag in zip(sentence, tags):
        word.tag = tag

    word_index += len(sentence)
format_words(open('/tmp/test_ner', 'wb'), test_words, tag_scheme=TAG_SCHEME)
