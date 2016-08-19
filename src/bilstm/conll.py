import re
from itertools import chain


DOC_START_MARK = '-DOCSTART-'
SENTENCE_END_MARK = '\r \n'
WORD_END_MARK = '\n'

_WORD_DATA_SEP = ['\r ', ' ']
WORD_DATA_SEP_PATTERN = '|'.join(map(re.escape, _WORD_DATA_SEP))


def parse_conll_train(text, separate_to_documents=True):
    """
    Parses text according to CoNLL-2003 NER train\test file format
    :param text: the raw text of the train\test file
    :return: a list containing a dict of attributes for each word in the given file
    """
    document_list = list()
    for raw_document in text.split(DOC_START_MARK)[1:]:
        document = list()
        for raw_sentence in raw_document.split(SENTENCE_END_MARK)[1:-1]:
            sentence = list()
            for raw_word in raw_sentence.split(WORD_END_MARK)[:-1]:
                text, pos, chunk, gold_label = re.split(WORD_DATA_SEP_PATTERN, raw_word)
                word_attrs = {'text': text, 'pos': pos, 'chunk': chunk, 'gold_label': gold_label}
                sentence.append(word_attrs)
            document.append(sentence)
        document_list.append(document)

    if not separate_to_documents:
        # let's unite all documents to one big document
        united_document = list(chain(*document_list))
        document_list = [united_document]

    return document_list


def format_conll_train(document_list):
    conll_text = ''
    for document in document_list:
        document_text = DOC_START_MARK

        first_sentence_mock = '\r -X- O O O' + WORD_END_MARK + SENTENCE_END_MARK
        document_text += first_sentence_mock

        for sentence in document:
            sentence_text = ""
            for word_attrs in sentence:
                word_text = '%s\r %s %s %s' % (word_attrs['text'], word_attrs['pos'], word_attrs['chunk'],
                                               word_attrs['gold_label'])
                word_text += WORD_END_MARK

                sentence_text += word_text
            sentence_text += SENTENCE_END_MARK

            document_text += sentence_text
        conll_text += document_text
    return conll_text


def parse_conll_tagged(text, separate_to_documents=True):
    document_list = list()
    for raw_document in text.split(DOC_START_MARK)[1:]:
        document = list()
        for raw_sentence in raw_document.split('\n\n')[1:-1]:
            sentence = list()
            for raw_word in raw_sentence.split(WORD_END_MARK):
                text, pos, chunk, gold_label, tag = re.split(WORD_DATA_SEP_PATTERN, raw_word)
                word_attrs = {'text': text, 'pos': pos, 'chunk': chunk, 'gold_label': gold_label, 'tag': tag}
                sentence.append(word_attrs)
            document.append(sentence)
        document_list.append(document)

    if not separate_to_documents:
        # let's unite all documents to one big document
        united_document = list(chain(*document_list))
        document_list = [united_document]

    return document_list


def format_conll_tagged(document_list):
    conll_text = ''
    for document in document_list:
        document_text = DOC_START_MARK

        first_sentence_mock = '\r -X- O O O\n\n'
        document_text += first_sentence_mock

        for sentence in document:
            sentence_text = ""
            for word_attrs in sentence:
                word_text = '%s\r %s %s %s %s' % (word_attrs['text'], word_attrs['pos'], word_attrs['chunk'],
                                                  word_attrs['gold_label'], word_attrs['tag'])
                word_text += WORD_END_MARK

                sentence_text += word_text
            sentence_text += '\n'

            document_text += sentence_text
        conll_text += document_text
    return conll_text
