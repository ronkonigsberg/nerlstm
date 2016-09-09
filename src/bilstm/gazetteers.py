import re

from bilstm.word import Word


def gazetteer_file_to_sentences(gazetteer_file_path, entity_type, tag_scheme=None):
    gazetteer_words = []
    gazetteer_sentences = []
    for line in open(gazetteer_file_path, 'rb').readlines():
        if '(' in line:
            line = line[:line.find('(')]
        line = re.sub(" +", ' ', line.replace(',', ' , '))
        line = line.strip('\n\t ')

        line_words = []
        for word_idx, word_text in enumerate(line.split()):
            current_word = Word(sentence=line_words, sentence_index=word_idx,
                                document=line_words, document_index=word_idx,
                                text=word_text, gold_label='I-%s' % entity_type)
            line_words.append(current_word)
        gazetteer_words.extend(line_words)
        gazetteer_sentences.append(line_words)

    if tag_scheme is not None:
        tag_scheme.encode(gazetteer_words, 'gold_label')
    return gazetteer_sentences
