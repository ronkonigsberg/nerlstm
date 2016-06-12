from pycnn import *
from collections import Counter
import random
import gc
from nertagger.parser import parse_conll_train, format_conll_tagged
from nertagger.word import parsed_documents_to_words, words_to_parsed_documents
import numpy as np

import util


MLP = True


# format of files: each line is "word<TAB>tag<newline>", blank line is new sentence.
train_file_path = '/Users/konix/Workspace/nertagger/data/eng.train'
train_words = parsed_documents_to_words(parse_conll_train(open(train_file_path, 'rb').read()))
train_sentences = []
word_index = 0
while word_index < len(train_words):
    sentence = train_words[word_index].sentence
    train_sentences.append([(word_.text, word_.gold_label) for word_ in sentence])
    word_index += len(sentence)

dev_file_path = '/Users/konix/Workspace/nertagger/data/eng.testa'
dev_words = parsed_documents_to_words(parse_conll_train(open(dev_file_path, 'rb').read()))
dev_sentences = []
word_index = 0
while word_index < len(dev_words):
    sentence = dev_words[word_index].sentence
    dev_sentences.append([(word_.text, word_.gold_label) for word_ in sentence])
    word_index += len(sentence)

train = train_sentences
test = dev_sentences

words=[]
tags=[]
wc=Counter()
for s in train:
    for w,p in s:
        words.append(w)
        tags.append(p)
        wc[w]+=1
words.append("_UNK_")
#words=[w if wc[w] > 1 else "_UNK_" for w in words]
tags.append("_START_")

for s in test:
    for w,p in s:
        words.append(w)

vw = util.Vocab.from_corpus([words])
vt = util.Vocab.from_corpus([tags])
UNK = vw.w2i["_UNK_"]

nwords = vw.size()
ntags  = vt.size()

model = Model()
sgd = SimpleSGDTrainer(model)

model.add_lookup_parameters("lookup", (nwords, 200))
model.add_lookup_parameters("tl", (ntags, 5))

# My Code: initialize word lookup based on pre-trained embeddings
embedding_by_word = {}
for line in open('/Users/konix/Documents/pos_data/glove.6B/glove.6B.200d.txt', 'rb').readlines():
    word, embedding_str = line.split(' ', 1)
    embedding = np.asarray([float(value_str) for value_str in embedding_str.split()])
    embedding_by_word[word] = embedding
word_lookup = model["lookup"]
for idx in xrange(vw.size()):
    word = vw.i2w[idx]
    if word in embedding_by_word:
        word_lookup.init_row(idx, embedding_by_word[word])
del embedding_by_word
gc.collect()

if MLP:
    pH = model.add_parameters("HID", (32, 50*2))
    pO = model.add_parameters("OUT", (ntags, 32))
else:
    pO = model.add_parameters("OUT", (ntags, 50*2))

builders=[
        LSTMBuilder(1, 200, 50, model),
        LSTMBuilder(1, 200, 50, model),
        ]

def build_tagging_graph(words, tags, model, builders):
    renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]

    wembs = [lookup(model["lookup"], w) for w in words]
    wembs = [noise(we,0.1) for we in wembs]

    fw = [x.output() for x in f_init.add_inputs(wembs)]
    bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

    if MLP:
        H = parameter(pH)
        O = parameter(pO)
    else:
        O = parameter(pO)
    errs = []
    for f,b,t in zip(fw, reversed(bw), tags):
        f_b = concatenate([f,b])
        if MLP:
            r_t = O*(tanh(H * f_b))
        else:
            r_t = O * f_b
        err = pickneglogsoftmax(r_t, t)
        errs.append(err)
    return esum(errs)

def tag_sent(sent, model, builders):
    renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]
    wembs = [lookup(model["lookup"], vw.w2i.get(w, UNK)) for w,t in sent]

    fw = [x.output() for x in f_init.add_inputs(wembs)]
    bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

    if MLP:
        H = parameter(pH)
        O = parameter(pO)
    else:
        O = parameter(pO)
    tags=[]
    for f,b,(w,t) in zip(fw,reversed(bw),sent):
        if MLP:
            r_t = O*(tanh(H * concatenate([f,b])))
        else:
            r_t = O*concatenate([f,b])
        out = softmax(r_t)
        chosen = np.argmax(out.npvalue())
        tags.append(vt.i2w[chosen])
    return tags


tagged = loss = 0
for ITER in xrange(6):
    random.shuffle(train)
    for i,s in enumerate(train,1):
        if i % 5000 == 0:
            sgd.status()
            print loss / tagged
            loss = 0
            tagged = 0
        if i % 10000 == 0:
            good = bad = 0.0
            for sent in test:
                tags = tag_sent(sent, model, builders)
                golds = [t for w,t in sent]
                for go,gu in zip(golds,tags):
                    if go == gu: good +=1
                    else: bad+=1
            print good/(good+bad)
        ws = [vw.w2i.get(w, UNK) for w,p in s]
        ps = [vt.w2i[p] for w,p in s]
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

dev_text = format_conll_tagged(words_to_parsed_documents(dev_words))
open('/tmp/dev_ner', 'wb').write(dev_text)
