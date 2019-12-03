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
        self.filters = [2, 3, 4]
        self.pooling_size = 5
        self.filter_size = 3
        self.num_filters = 30
        self.half_kernel_size = 3
        self.clip = 5
        with tf.name_scope("ner_declare"):
            self.inputs = tf.placeholder(tf.int32, [None, self.io_sequence_size], name="char_inputs")
            self.targets = tf.placeholder(tf.int32, [None, self.io_sequence_size], name="targets")
            self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.create_declare()
        self.build_model()
        self.create_loss()

    def create_declare(self):
        with tf.name_scope("ner_declare"):
            self.weight_variable = tf.get_variable("weight_variable",
                                                   shape=[self.hidden_size * 2, self.output_class_size],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.bias_variable = tf.get_variable("bias_variable", shape=[self.output_class_size])

    def build_model(self):
        with tf.device('/cpu:0'), tf.name_scope("word_embedding"):
            # 加1是因为第0个字表示任意的word
            self.W_word = tf.Variable(tf.random_uniform([self.vocab_size + 1, self.embedding_size], -1, 1),
                                      name="W_word")
            self.embedded_layer = tf.nn.embedding_lookup(self.W_word, self.inputs, name="embedded_words")

        with tf.name_scope("conv_maxPool"):
            """
            filters的格式为：[filter_width, in_channels, out_channels]。
            filter_width可以看作每次与value进行卷积的行数，
            in_channels表示value一共有多少列（与value中的in_channels相对应）也就是词向量的维度
            out_channels表示输出通道，可以理解为一共有多少个卷积核，即卷积核的数目
            """
            filter_shape = [self.filter_size, self.embedding_size, self.num_filters]
            W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
            b_conv = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b_conv")
            # 一维卷积（conv1d）
            # self.embedded_layer要被卷积的矩阵[batch_size, self.io_sequence_size, self.embedding_size]
            # stride：一个整数，表示步长，每次（向下）移动的距离
            conv = tf.nn.conv1d(self.embedded_layer,
                                W_conv,
                                stride=1,
                                padding="SAME",
                                name="conv")
            # conv: [batch, out_width, out_channels]
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv, name="add_bias"))
            # 补齐channels
            h_expand = tf.expand_dims(h, -1)
            pooled = tf.nn.max_pool(
                h_expand,
                ksize=[1, self.io_sequence_size, 1, 1],
                strides=[1, 1, 1, 1],
                padding='SAME',
                name="pooled")
            self.word_pool_flat = tf.reshape(pooled, [-1, self.io_sequence_size, self.num_filters],
                                             name="word_pool_flat")
            self.embedded_layer = tf.nn.dropout(self.word_pool_flat, self.keep_prob,
                                                name="word_features_dropout")

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
                # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
                # mask = tf.sequence_mask(self.sequence_lengths)
                # self.cost_func = tf.boolean_mask(losses, mask)
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
