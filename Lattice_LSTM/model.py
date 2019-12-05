# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/4 19:12
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from utils.data import Data
from Latttice_Lstm import LatticeLSTMCell
import tensorflow as tf
from tensorflow.python import debug as tf_debug


class Model_Lattice:

    def __init__(self, max_char_len, emb_size, max_lexicon_words_num, num_units, num_tags, learning_rate):
        self.batch_size = 64
        self.max_char_len = max_char_len
        self.emb_size = emb_size
        self.max_lexicon_words_num = max_lexicon_words_num
        self.num_units = num_units
        self.num_tags = num_tags
        self.learning_rate = learning_rate
        self.optimizer = 'adam'
        self.clip = 5
        self.data = Data()
        self.data.build_word_pretrain_emb(
            'D:\\mygit\\NER_MODEL\\data\\data\\gigaword_chn.all.a2b.uni.ite50.vec')
        self.data.build_gaz_pretrain_emb('D:\\mygit\\NER_MODEL\\data\\data\\ctb.50d.vec')

        def my_filter_callable(tensor):
            # A filter that detects zero-valued scalars.
            return len(tensor.shape) == 0 and tensor == 0.0

        self.sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session())
        self.sess.add_tensor_filter('my_filter', my_filter_callable)

        self.sess = tf.Session()
        self.placeholders = {}
        self.epoch = 0
        self.global_step = tf.Variable(0, trainable=False)

        self.char_ids = tf.placeholder(tf.int32, [None, self.max_char_len])
        self.lexicon_word_ids = tf.placeholder(tf.int32, [None, self.max_char_len,
                                                          self.max_lexicon_words_num])
        self.word_length_tensor = tf.placeholder(tf.float32, [None, self.max_char_len,
                                                              self.max_lexicon_words_num])
        self.labels = tf.placeholder(tf.int32, [None, self.max_char_len])

        self.lexicon_word_ids_reshape = tf.reshape(self.lexicon_word_ids,
                                                   [-1, self.max_char_len * self.max_lexicon_words_num])
        self.seq_length = tf.convert_to_tensor(self.batch_size * [self.max_char_len], dtype=tf.int32)
        self.placeholders["char_ids"] = self.char_ids
        self.placeholders["lexicon_word_ids"] = self.lexicon_word_ids
        self.placeholders["word_length_tensor"] = self.word_length_tensor
        self.placeholders["labels"] = self.labels
        self.create_embedding()
        self.create_declare()
        self.create_model()
        self.create_loss()

    def create_embedding(self):
        self.char_embeddings = tf.Variable(self.data.pretrain_word_embedding,
                                           dtype=tf.float32, name="char_embeddings")
        self.word_embeddings = tf.Variable(self.data.pretrain_gaz_embedding,
                                           dtype=tf.float32, name="word_embeddings")

        self.char_embed = tf.nn.embedding_lookup(self.char_embeddings, self.char_ids)
        self.lexicon_word_embed_reshape = tf.nn.embedding_lookup(self.word_embeddings, self.lexicon_word_ids_reshape)
        self.lexicon_word_embed = tf.reshape(self.lexicon_word_embed_reshape,
                                             [-1, self.max_char_len, self.max_lexicon_words_num, self.emb_size])

    def create_declare(self):
        # projection:
        self.W = tf.get_variable("projection_w", [self.num_units, self.num_tags],
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
        self.b = tf.get_variable("projection_b", [self.num_tags])

    def create_model(self):
        lattice_lstm = LatticeLSTMCell(self.num_units,
                                       self.num_units,
                                       batch_size=self.batch_size,
                                       seq_len=self.max_char_len,
                                       max_lexicon_words_num=self.max_lexicon_words_num,
                                       word_length_tensor=self.word_length_tensor,
                                       dtype=tf.float32)

        initial_state = lattice_lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell=lattice_lstm,
                                           inputs=[self.char_embed, self.lexicon_word_embed],
                                           initial_state=initial_state,
                                           dtype=tf.float32)
        x_reshape = tf.reshape(outputs, [-1, self.num_units])
        projection = tf.matmul(x_reshape, self.W) + self.b

        # -1 to timestep
        self.logits = tf.reshape(projection, [self.batch_size, -1, self.num_tags])

    def create_loss(self):
        self.loss = self.loss_layer(self.logits, self.seq_length, self.labels)
        with tf.variable_scope("optimizer"):
            self.opt = tf.train.AdamOptimizer(self.learning_rate)

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = map(
                lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -self.clip, self.clip), gv[1]],
                grads_vars)
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

    def loss_layer(self, project_logits, lengths, labels, name=None):
        """ calculate crf loss
        :param project_logits: [batch_size, num_steps, num_tags]
        :param lengths: [batch_size, num_steps]
        :param labels: [batch_size, num_steps]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                 tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)

            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.max_char_len, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)

            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), labels], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=tf.random_uniform_initializer(
                    0.008, 0.15, seed=1311, dtype=tf.float32))

            log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths + 1)

            return tf.reduce_sum(-log_likelihood)
