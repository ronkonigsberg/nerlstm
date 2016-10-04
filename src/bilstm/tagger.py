import random

import numpy as np
from pycnn import (Model, AdamTrainer, LSTMBuilder, renew_cg, lookup, dropout, parameter, concatenate,
                   softmax, pickneglogsoftmax, esum, log, exp)


class BiLstmNerTagger(object):
    WORD_DIM = 100
    CHAR_DIM = 25
    LSTM_DIM = 100

    HIDDEN_DIM = 150

    def __init__(self, word_indexer, char_indexer, tag_indexer, external_word_embeddings=None, model_file_path=None):
        self.word_indexer = word_indexer
        self._unk_word_index = word_indexer.index_object('_UNK_')
        self.char_indexer = char_indexer

        self.tag_indexer = tag_indexer
        self.start_tag_index = tag_indexer.get_index('-START-')
        self.end_tag_index = tag_indexer.get_index('-END-')
        self.word_tag_count = len(self.tag_indexer) - 2

        model = Model()
        model.add_lookup_parameters("word_lookup", (len(word_indexer), self.WORD_DIM))
        model.add_lookup_parameters("char_lookup", (len(char_indexer), self.CHAR_DIM))

        if external_word_embeddings:
            word_lookup = model["word_lookup"]
            for idx in xrange(len(word_indexer)):
                word = word_indexer.get_object(idx)
                if word in external_word_embeddings:
                    word_lookup.init_row(idx, external_word_embeddings[word])

        self.param_transition = model.add_parameters("TRANSITION", (len(tag_indexer)**2,  1))
        self.param_out = model.add_parameters("OUT", (self.word_tag_count, self.LSTM_DIM*2))

        self.char_builders = [
            LSTMBuilder(1, self.CHAR_DIM, self.CHAR_DIM, model),
            LSTMBuilder(1, self.CHAR_DIM, self.CHAR_DIM, model)
        ]

        self.word_builders = [
            LSTMBuilder(1, self.WORD_DIM + self.CHAR_DIM*2, self.LSTM_DIM, model),
            LSTMBuilder(1, self.WORD_DIM + self.CHAR_DIM*2, self.LSTM_DIM, model)
        ]

        if model_file_path is not None:
            model.load(model_file_path)

        self.model = model
        self.trainer = AdamTrainer(model)

    def _build_word_expression_list(self, sentence, is_train=False):
        renew_cg()

        sentence_word_vectors = []
        for word in sentence:
            sentence_word_vectors.append(self._get_word_vector(word, use_dropout=is_train))

        lstm_forward = self.word_builders[0].initial_state()
        lstm_backward = self.word_builders[1].initial_state()

        embeddings_forward = []
        embeddings_backward = []
        for word_vector, reverse_word_vector in zip(sentence_word_vectors, reversed(sentence_word_vectors)):
            lstm_forward = lstm_forward.add_input(word_vector)
            lstm_backward = lstm_backward.add_input(reverse_word_vector)

            embeddings_forward.append(lstm_forward.output())
            embeddings_backward.append(lstm_backward.output())

        O = parameter(self.param_out)

        sentence_word_expressions = []
        for word_f_embedding, word_b_embedding in zip(embeddings_forward, reversed(embeddings_backward)):
            word_concat_embedding = concatenate([word_f_embedding, word_b_embedding])

            word_expression = O * word_concat_embedding
            sentence_word_expressions.append(word_expression)
        return sentence_word_expressions

    def calc_sentence_error(self, sentence):
        word_expression_list = self._build_word_expression_list(sentence, is_train=True)

        sentence_errors = []
        for word, word_expression in zip(sentence, word_expression_list):
            gold_label_index = self.tag_indexer.get_index(word.gold_label)
            word_error = pickneglogsoftmax(word_expression, gold_label_index)
            sentence_errors.append(word_error)
        return esum(sentence_errors)

    def calc_sentence_error_viterbi(self, sentence):
        word_expression_list = self._build_word_expression_list(sentence, is_train=True)
        transition_matrix = parameter(self.param_transition)

        gold_expr = self._get_gold_expression(sentence, word_expression_list, transition_matrix)
        all_sequence_expr = self._get_all_sequence_expr(word_expression_list, transition_matrix)
        return all_sequence_expr - gold_expr

    def _get_all_sequence_expr(self, word_expression_list, transition_matrix):
        cur_viterbi_dict = {self.start_tag_index: None}
        for word_index, word_expression in enumerate(word_expression_list):
            prev_viterbi_dict = cur_viterbi_dict
            cur_viterbi_dict = {}
            for prev_tag_index, prev_tag_expr in prev_viterbi_dict.iteritems():
                for cur_tag_index in xrange(self.word_tag_count):
                    cur_tag_expr = word_expression[cur_tag_index]
                    transition_expr = self._get_transition_expr(transition_matrix, prev_tag_index, cur_tag_index)

                    bigram_expr = ((prev_tag_expr + (transition_expr + cur_tag_expr)) if prev_tag_expr is not None
                                   else (transition_expr + cur_tag_expr))

                    if cur_tag_index not in cur_viterbi_dict:
                        cur_viterbi_dict[cur_tag_index] = bigram_expr
                    else:
                        cur_viterbi_dict[cur_tag_index] = self.logadd_expr(cur_viterbi_dict[cur_tag_index], bigram_expr)

        end_tag_index = self.end_tag_index
        all_sequence_expr = None
        for last_tag_index, last_tag_expr in cur_viterbi_dict.iteritems():
            transition_expr = self._get_transition_expr(transition_matrix, last_tag_index, end_tag_index)
            final_expr = (last_tag_expr + transition_expr) if last_tag_expr is not None else transition_expr

            if all_sequence_expr is None:
                all_sequence_expr = final_expr
            else:
                all_sequence_expr = self.logadd_expr(all_sequence_expr, final_expr)
        return all_sequence_expr

    def calc_sentence_error_semi_viterbi(self, sentence):
        word_expression_list = self._build_word_expression_list(sentence, is_train=True)
        transition_matrix = parameter(self.param_transition)

        gold_expr = self._get_gold_expression(sentence, word_expression_list, transition_matrix)
        output_expr, _ = self.decode_sentence_tags_by_viterbi(word_expression_list, transition_matrix)
        return exp(output_expr - gold_expr)

    def _get_gold_expression(self, sentence, word_expression_list, transition_matrix):
        prev_gold_tag_index = self.start_tag_index
        sentence_expr = None
        for word, word_expression in zip(sentence, word_expression_list):
            cur_gold_tag_index = self.tag_indexer.get_index(word.gold_label)
            transition_expr = self._get_transition_expr(transition_matrix, prev_gold_tag_index, cur_gold_tag_index)
            cur_tag_expr = word_expression[cur_gold_tag_index]

            sentence_expr = ((sentence_expr + transition_expr + cur_tag_expr) if sentence_expr is not None else
                             (transition_expr + cur_tag_expr))

            prev_gold_tag_index = cur_gold_tag_index

        final_transition_expr = self._get_transition_expr(transition_matrix, prev_gold_tag_index, self.end_tag_index)
        sentence_expr = (sentence_expr + final_transition_expr) if sentence_expr is not None else final_transition_expr
        return sentence_expr

    def tag_sentence_viterbi(self, sentence):
        word_expression_list = self._build_word_expression_list(sentence, is_train=False)
        transition_matrix = parameter(self.param_transition)

        best_score_expression, sentence_tags = self.decode_sentence_tags_by_viterbi(word_expression_list,
                                                                                    transition_matrix)
        for word, word_tag in zip(sentence, sentence_tags):
            word.tag = word_tag

    def decode_sentence_tags_by_viterbi(self, word_expression_list, transition_matrix):
        cur_viterbi_dict = {self.start_tag_index: (None, None)}
        bp = {}
        for word_index, word_expression in enumerate(word_expression_list):
            prev_viterbi_dict = cur_viterbi_dict
            cur_viterbi_dict = {}
            for prev_tag_index, (prev_tag_expr, prev_tag_score) in prev_viterbi_dict.iteritems():
                for cur_tag_index in xrange(self.word_tag_count):
                    cur_tag_expr = word_expression[cur_tag_index]
                    transition_expr = self._get_transition_expr(transition_matrix, prev_tag_index, cur_tag_index)

                    bigram_expr = ((prev_tag_expr + transition_expr + cur_tag_expr) if prev_tag_expr is not None else
                                   (transition_expr + cur_tag_expr))
                    bigram_score = bigram_expr.npvalue()

                    if cur_tag_index not in cur_viterbi_dict or cur_viterbi_dict[cur_tag_index][1] < bigram_score:
                        cur_viterbi_dict[cur_tag_index] = (bigram_expr, bigram_score)
                        bp[(word_index, cur_tag_index)] = prev_tag_index

        end_tag_index = self.end_tag_index
        best_viterbi_score = None
        best_score_expression = None
        for last_tag_index, (last_tag_expr, last_tag_score) in cur_viterbi_dict.iteritems():
            transition_expr = self._get_transition_expr(transition_matrix, last_tag_index, end_tag_index)
            final_expr = (last_tag_expr + transition_expr) if last_tag_expr is not None else transition_expr
            final_score = final_expr.npvalue()

            if best_viterbi_score is None or best_viterbi_score < final_score:
                best_viterbi_score = final_score
                best_score_expression = final_expr
                bp[(len(word_expression_list), end_tag_index)] = last_tag_index

        best_expression_sentence_tags = []
        cur_tag_index = end_tag_index
        for idx in xrange(len(word_expression_list), 0, -1):
            prev_tag_index = bp[(idx, cur_tag_index)]
            best_expression_sentence_tags.insert(0, self.tag_indexer.get_object(prev_tag_index))
            cur_tag_index = prev_tag_index

        return best_score_expression, best_expression_sentence_tags

    def _get_transition_expr(self, transition_matrix, prev_tag_index, cur_tag_index):
        transition_index = prev_tag_index * len(self.tag_indexer) + cur_tag_index
        return transition_matrix[transition_index]

    def logadd_expr(self, expr1, expr2):
        if (expr1.npvalue() - expr2.npvalue()) > 0:
            return expr1 + log(exp(expr2 - expr1) + 1)
        else:
            return expr2 + log(exp(expr1 - expr2) + 1)

    def tag_sentence(self, sentence):
        word_expression_list = self._build_word_expression_list(sentence, is_train=False)
        for word, word_expression in zip(sentence, word_expression_list):
            out = softmax(word_expression)
            tag_index = np.argmax(out.npvalue())
            word.tag = self.tag_indexer.get_object(tag_index)

    def train(self, train_sentence_list, dev_sentence_list=None, test_sentence_list=None, eval_func=None, iterations=5,
              model_save_dir=None):
        train_sentence_list = list(train_sentence_list)

        loss = tagged = 0
        for iteration_idx in xrange(1, iterations+1):
            print "Starting training iteration %d/%d" % (iteration_idx, iterations)
            random.shuffle(train_sentence_list)
            for sentence_index, sentence in enumerate(train_sentence_list, 1):
                sentence_error = self.calc_sentence_error_viterbi(sentence)
                loss += sentence_error.scalar_value()
                tagged += len(sentence)
                sentence_error.backward()
                self.trainer.update()

            # Trainer Status
            self.trainer.status()
            print loss / tagged
            loss = tagged = 0

            if eval_func and dev_sentence_list:
                # Dev Evaluation
                for dev_sentence in dev_sentence_list:
                    self.tag_sentence_viterbi(dev_sentence)
                eval_func(dev_sentence_list)

            if eval_func and test_sentence_list:
                # Test Evaluation
                for test_sentence in test_sentence_list:
                    self.tag_sentence_viterbi(test_sentence)
                eval_func(test_sentence_list)

            if model_save_dir is not None:
                iteration_save_file = os.path.join(model_save_dir, 'iteration_%d' % iteration_idx)
                self.model.save(iteration_save_file)

    def _get_word_vector(self, word, use_dropout=False):
        word_embedding = self._get_word_embedding(word)
        char_representation = self._get_char_representation(word)

        word_vector = concatenate([word_embedding, char_representation])
        if use_dropout:
            word_vector = dropout(word_vector, 0.5)
        return word_vector

    def _get_word_embedding(self, word):
        word_index = self.word_indexer.get_index(word.text.lower()) or self._unk_word_index
        return lookup(self.model["word_lookup"], word_index)

    def _get_char_representation(self, word):
        word_char_vectors = []
        for char in word.text:
            char_index = self.char_indexer.get_index(char)
            if char_index is None:
                print "Warning: Unexpected char '%s' (word='%s')" % (char, word.text)
                continue
            char_vector = lookup(self.model["char_lookup"], char_index)
            word_char_vectors.append(char_vector)

        lstm_forward = self.char_builders[0].initial_state()
        lstm_backward = self.char_builders[1].initial_state()

        for char_vector, reverse_char_vector in zip(word_char_vectors, reversed(word_char_vectors)):
            lstm_forward = lstm_forward.add_input(char_vector)
            lstm_backward = lstm_backward.add_input(reverse_char_vector)
        return concatenate([lstm_forward.output(), lstm_backward.output()])
