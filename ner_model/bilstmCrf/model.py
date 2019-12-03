# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/11/28 11:34
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf


class NerCore:
    def __init__(self, io_sequence_size, vocab_size, class_size=6, keep_prob=0.5, learning_rate=0.001, trainable=False):
        self.is_training = trainable
        self.vocab_size = vocab_size
        self.io_sequence_size = io_sequence_size
        self.learning_rate = learning_rate
        self.embedding_size = 256
        self.hidden_size = 128
        self.output_class_size = class_size
        self.keep_prob = keep_prob
        self.num_layers = 1
        self.warmup_steps = 4000
        self.clip = 5
        with tf.name_scope("ner_declare"):
            self.inputs = tf.placeholder(tf.int32, [None, self.io_sequence_size], name="char_inputs")
            self.targets = tf.placeholder(tf.int32, [None, self.io_sequence_size], name="targets")
            self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.create_declare()
        self.create_embedding()
        self.build_model()
        self.create_loss()

    def create_declare(self):
        with tf.name_scope("ner_declare"):
            self.weight_variable = tf.get_variable("weight_variable",
                                                   shape=[self.hidden_size * 2, self.output_class_size],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.bias_variable = tf.get_variable("bias_variable", shape=[self.output_class_size])

    def create_embedding(self):
        with tf.variable_scope('embedding', reuse=None):
            self.embedding_variable = tf.get_variable("embeddings",
                                                      shape=[self.vocab_size, self.embedding_size],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            self.embedded_layer = tf.nn.embedding_lookup(self.embedding_variable, self.inputs,
                                                         name="embedding_layer")

    def build_model(self):
        with tf.name_scope("ner_layer"):
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size),
                                                         output_keep_prob=self.keep_prob)
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size),
                                                         output_keep_prob=self.keep_prob)
            with tf.variable_scope("ner_layer", reuse=tf.AUTO_REUSE):
                for i in range(self.num_layers):
                    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw,
                                                                                lstm_cell_bw,
                                                                                self.embedded_layer,
                                                                                sequence_length=self.sequence_lengths,
                                                                                dtype=tf.float32)
                    outputs = tf.concat((output_fw, output_bw), 2)
                self.outputs = tf.concat(outputs, axis=-1)
                self.outputs = tf.nn.dropout(self.outputs, self.keep_prob)
                self.outputs = tf.reshape(self.outputs, [-1, 2 * self.hidden_size])
                self.logits = tf.matmul(self.outputs, self.weight_variable) + self.bias_variable
                self.logits = tf.reshape(self.logits, [-1, self.io_sequence_size, self.output_class_size])

    def create_loss(self):
        with tf.name_scope("ner_loss"):
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                                                       tag_indices=self.targets,
                                                                                       sequence_lengths=self.sequence_lengths)
            if self.is_training == True:
                self.cost_func = tf.reduce_mean(-log_likelihood)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_func)

    def decode(self, terms, taggs):
        char_item = []
        tag_item = []
        raw_content = {}
        for i in range(len(terms)):
            if taggs[i][0] == 'B':
                if len(char_item) > 0 and len(tag_item) > 0:
                    content = ''.join(char_item)
                    key = tag_item[0][2:]
                    position = (i - len(content), len(content))
                    if key in raw_content.keys():
                        raw_content[key].append((content, position))
                    else:
                        raw_content[key] = [(content, position)]
                    char_item = []
                    tag_item = []
                char_item.append(terms[i])
                tag_item.append(taggs[i])
            elif taggs[i][0] == 'O':
                if len(char_item) > 0 and len(tag_item) > 0:
                    content = ''.join(char_item)
                    position = (i - len(content), len(content))
                    key = tag_item[0][2:]
                    if key in raw_content.keys():
                        raw_content[key].append((content, position))
                    else:
                        raw_content[key] = [(content, position)]
                    char_item = []
                    tag_item = []
            else:
                char_item.append(terms[i])
                tag_item.append(taggs[i])
        if len(char_item) > 0 and len(tag_item) > 0:
            content = ''.join(char_item)
            key = tag_item[0][2:]
            position = (len(terms) - len(content), len(content))
            if key in raw_content.keys():
                raw_content[key].append((content, position))
            else:
                raw_content[key] = [(content, position)]
        return raw_content