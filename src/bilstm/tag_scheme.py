class BIO(object):
    """
    Converter which supports encoding from IOB(=CoNLL format) to BIO and decoding vice versa.
    """

    @staticmethod
    def encode(word_list, tag_attr='tag'):
        current_index = 0
        while current_index < len(word_list):
            current_sentence_words = word_list[current_index].sentence
            BIO._encode_sentence(current_sentence_words, tag_attr)
            current_index += len(current_sentence_words)

    @staticmethod
    def _encode_sentence(sentence_words, tag_attr):
        prev_type = None
        for word in sentence_words:
            word_type = _get_word_type(word, tag_attr)
            if word_type != prev_type and word_type:
                new_tag = 'B-' + word_type
                setattr(word, tag_attr, new_tag)

            prev_type = word_type

    @staticmethod
    def decode(word_list, tag_attr='tag'):
        current_index = 0
        while current_index < len(word_list):
            current_sentence_words = word_list[current_index].sentence
            BIO._decode_sentence(current_sentence_words, tag_attr)
            current_index += len(current_sentence_words)

    @staticmethod
    def _decode_sentence(sentence_words, tag_attr='tag'):
        prev_type = None
        for word in sentence_words:
            word_type = _get_word_type(word, tag_attr)
            if word_type and word_type != prev_type:
                new_tag = 'I-' + word_type
                setattr(word, tag_attr, new_tag)

            prev_type = word_type


class BILOU(object):
    """
    Converter which supports encoding from IOB(=CoNLL format) to BILOU and decoding vice versa.
    """

    @staticmethod
    def encode(word_list, tag_attr='tag'):
        current_index = 0
        while current_index < len(word_list):
            current_sentence_words = word_list[current_index].sentence
            BILOU._encode_sentence(current_sentence_words, tag_attr)
            current_index += len(current_sentence_words)

    @staticmethod
    def _encode_sentence(sentence_words, tag_attr):
        sentence_next_tag = map(lambda word_: getattr(word_, tag_attr), sentence_words[1:]) + ['O']

        prev_type = None
        for (word, next_tag) in zip(sentence_words, sentence_next_tag):
            word_type = _get_word_type(word, tag_attr)
            if word_type:
                word_tag = getattr(word, tag_attr)
                if prev_type == word_type and word_tag[0] == 'I':
                    if word_tag == next_tag:
                        new_tag = 'I-' + word_type
                    else:
                        new_tag = 'L-' + word_type
                else:
                    if next_tag[0] == 'I' and next_tag[1:] == word_tag[1:]:
                        new_tag = 'B-' + word_type
                    else:
                        new_tag = 'U-' + word_type

                setattr(word, tag_attr, new_tag)

            prev_type = word_type

    @staticmethod
    def decode(word_list, tag_attr='tag'):
        current_index = 0
        while current_index < len(word_list):
            current_sentence_words = word_list[current_index].sentence
            BILOU._decode_sentence(current_sentence_words, tag_attr)
            current_index += len(current_sentence_words)

    @staticmethod
    def _decode_sentence(sentence, tag_attr='tag'):
        prev_type = None
        for word in sentence:
            word_type = _get_word_type(word, tag_attr)
            if word_type:
                word_tag_prefix = getattr(word, tag_attr)[0]
                if word_type == prev_type and word_tag_prefix in ['B', 'U']:
                    new_tag = 'B-' + word_type
                else:
                    new_tag = 'I-' + word_type
                setattr(word, tag_attr, new_tag)

            prev_type = word_type


def _get_word_type(word, tag_attr):
    tag = getattr(word, tag_attr)
    tag_parts = tag.split('-')
    return tag_parts[1] if len(tag_parts) == 2 else None
