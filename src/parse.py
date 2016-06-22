from nertagger.parser import parse_conll_train, format_conll_tagged
from nertagger.word import parsed_documents_to_words, words_to_parsed_documents


def parse_words(file_obj, tag_scheme=None):
    word_list = parsed_documents_to_words(parse_conll_train(file_obj.read()))

    if tag_scheme is not None:
        tag_scheme.encode(word_list, 'gold_label')

    return word_list


def split_words_to_sentences(word_list):
    sentence_list = []

    word_index = 0
    while word_index < len(word_list):
        current_word = word_list[word_index]
        current_sentence = current_word.sentence

        # sentence_list.append([(word_.text, word_.gold_label) for word_ in current_sentence])
        sentence_list.append(current_sentence)
        word_index += len(current_sentence)
    return sentence_list


def format_words(file_obj, word_list, tag_scheme=None):
    if tag_scheme is not None:
        tag_scheme.decode(word_list, 'gold_label')
        tag_scheme.decode(word_list, 'tag')

    file_data = format_conll_tagged(words_to_parsed_documents(word_list))
    file_obj.write(file_data)
