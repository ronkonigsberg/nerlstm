class Word(object):
    def __init__(self, sentence, sentence_index, document, document_index, **word_attr_dict):
        self.sentence = sentence
        self.sentence_index = sentence_index
        self.document = document
        self.document_index = document_index

        for (attr, value) in word_attr_dict.iteritems():
            setattr(self, attr, value)

    def __repr__(self):
        return 'Word(text=%s)' % self.text

    def __str__(self):
        return self.text


def parsed_documents_to_words(parsed_documents_list):
    all_words = list()
    for parsed_document in parsed_documents_list:
        document_words = list()
        for parsed_sentence in parsed_document:
            sentence_words = list()
            for word_attr_dict in parsed_sentence:
                current_word = Word(sentence=sentence_words, sentence_index=len(sentence_words),
                                    document=document_words, document_index=len(document_words),
                                    **word_attr_dict)
                sentence_words.append(current_word)
                document_words.append(current_word)
                all_words.append(current_word)
    return all_words


def words_to_parsed_documents(all_words, word_attributes=('text', 'pos', 'chunk', 'gold_label', 'tag')):
    prev_document = None
    prev_sentence = None

    parsed_documents_list = list()
    for word in all_words:
        if word.document != prev_document:
            current_document = list()
            parsed_documents_list.append(current_document)
        if word.sentence != prev_sentence:
            current_sentence = list()
            current_document.append(current_sentence)
        word_attr_dict = {attr_name: getattr(word, attr_name) for attr_name in word_attributes}
        current_sentence.append(word_attr_dict)

        prev_document = word.document
        prev_sentence = word.sentence
    return parsed_documents_list
