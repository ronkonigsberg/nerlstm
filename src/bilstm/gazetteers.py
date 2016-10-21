import os


class GazetteersAnnotator(object):
    MAX_EXPRESSION_LEN = 5
    ANNOTATION_ATTRIBUTE = 'gazetteers'

    def __init__(self, gazetteers, gazetteers_lowercase):
        self._gazetteers = gazetteers
        self._gazetteers_lowercase = gazetteers_lowercase

    def annotate_data(self, word_list):
        for word in word_list:
            setattr(word, self.ANNOTATION_ATTRIBUTE, list())

        for word in word_list:
            self._annotate_from_anchor(word)

    def _annotate_from_anchor(self, anchor_word):
        """
        annotates all gazetteers expression starting from a given word.
        :param anchor_word: the given anchor (starting) word
        """
        sentence = anchor_word.sentence
        anchor_index = anchor_word.sentence_index
        max_expression_len = min(len(sentence)-anchor_index, self.MAX_EXPRESSION_LEN)

        for expression_len in xrange(1, max_expression_len+1):
            if expression_len == 1:
                expression_labels = ['U-']
            else:
                expression_labels = ['B-'] + ['I-']*(expression_len-2) + ['L-']

            expression_words = sentence[anchor_index: anchor_index+expression_len]
            expression = ' '.join(map(lambda w: w.text, expression_words))

            for gazetteer_name, gazetteer in self._gazetteers.iteritems():
                if expression in gazetteer:
                    for word, label in zip(expression_words, expression_labels):
                        word_gazetteer_list = getattr(word, self.ANNOTATION_ATTRIBUTE)
                        word_gazetteer_list .append(label+gazetteer_name)

            for gazetteer_name, gazetteer_lowercase in self._gazetteers_lowercase.iteritems():
                if expression.lower() in gazetteer_lowercase:
                    for word, label in zip(expression_words, expression_labels):
                        word_gazetteer_list = getattr(word, self.ANNOTATION_ATTRIBUTE)
                        word_gazetteer_list.append(label+gazetteer_name+"(lower)")


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


def create_gazetteers_annotator(gazetteers_dir_path):
    gazetteers, lowercase_gazetteers = parse_gazetteers_directory(gazetteers_dir_path)
    return GazetteersAnnotator(gazetteers, lowercase_gazetteers)
