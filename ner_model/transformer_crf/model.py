# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/11/27 16:32
@Author  : zwt
@git   : https://github.com/phychaos/transformer_crf
@Software: PyCharm
"""
from ner_model.transformer_crf.model_utils import *


class TransformerCRFModel(object):

    def __init__(self, vocab_size, num_tags, sequence_length, num_units, dropout_rate, num_heads, is_training=True):
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.is_training = is_training
        self.sequence_length = sequence_length
        self.num_units = num_units
        self.num_blocks = 6
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.lr = 0.00001
        self.clip = 5
        self.embedding_size = 512
        self.warmup_steps = 4000
        self.num_layers = 2
        self.x = tf.placeholder(tf.int32, shape=[None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, shape=[None, self.sequence_length])
        self.seq_lens = tf.placeholder(tf.int32, shape=[None])
        self.global_step = tf.train.create_global_step()
        self.create_declare()
        self.create_embedding()
        self.build_model()
        self.create_loss()

    def create_embedding(self):
        with tf.variable_scope('embedding', reuse=None):
            lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[self.vocab_size, self.embedding_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
            lookup_table = tf.concat((tf.zeros(shape=[1, self.embedding_size]), lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, self.x)
            self.embedding = outputs * (self.num_units ** 0.5)

    def create_declare(self):
        self.w = tf.get_variable(name='w', dtype=tf.float32, shape=[self.num_units * 2, self.num_tags])
        self.b = tf.get_variable(name='b', dtype=tf.float32, shape=[self.num_tags])

    def build_model(self):
        embeddings = self.encoder(self.embedding)
        fw_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
        bw_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
        with tf.variable_scope("ner_layer", reuse=tf.AUTO_REUSE):
            for i in range(self.num_layers):
                (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, embeddings,
                                                                            sequence_length=self.seq_lens,
                                                                            dtype=tf.float32)

                outputs = tf.concat((fw_output, bw_output), 2)
            # [batch_size, sequence_lenght, 2 * self.num_units]
            self.outputs = tf.concat(outputs, axis=-1)
            # [batch_size, sequence_lenght, 2 * self.num_units]
            self.outputs = tf.nn.dropout(self.outputs, self.dropout_rate)
            # [batch_size * sequence_lenght, 2 * self.num_units]
            self.outputs = tf.reshape(self.outputs, [-1, 2 * self.num_units])
            # [batch_size * sequence_length, num_tags]
            self.logits = tf.matmul(self.outputs, self.w) + self.b
            # [batch_size, sequence_length, num_tags]
            self.logits = tf.reshape(self.logits, [-1, self.sequence_length, self.num_tags])

    def create_loss(self):
        """
        self.transition: 形状为[num_tags, num_tags] 的转移矩阵
        log_likelihood: 包含给定序列标签索引的对数似然的标量
        """
        log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(self.logits, self.y, self.seq_lens)
        self.loss = tf.reduce_mean(-log_likelihood)

        # l2正则化
        l2 = sum(1e-5 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.loss += l2
        """
        global_step在滑动平均、优化器、指数衰减学习率等方面都有用到,
        代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表
        """
        global_step = tf.train.get_or_create_global_step()
        lr = self.noam_scheme(self.lr, global_step, self.warmup_steps)
        """
        tf.clip_by_global_norm:对梯度进行裁剪、通过控制梯度的最大范式，防止梯度爆炸问题
        1：设置clip
        2：在前向传播与后向传播后，会得到每个权重的梯度，和不裁剪的区别在于，需要先求得所有权重梯度的平方和a，如果a大于clip，
        则计算缩放因子clip/a， 缩放因子在0-1之间，如果权重梯度的平方和a越大，则缩放因子越小
        3：最后将所有的权重梯度乘以这个缩放因子，这就得到了最后的梯度信息。
        这样可以保证在一次迭代更新中，所有权重的平方和在一个设定范围内，这个范围的上限就是clip
        """
        params = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, params), self.clip)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98,
                                                epsilon=1e-8).apply_gradients(zip(grads, params))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        # self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def noam_scheme(self, init_lr, global_step, warmup_steps=4000.):
        """
        学习率预热：在训练的轮数达到warmup_steps过程中，学习率会逐渐增加到init_lr，
        训练轮数超过warmup_steps之后学习率会从init_lr开始逐步下降。
        :param init_lr:
        :param global_step:
        :param warmup_steps:
        :return:
        """
        # tf.cast:数据类型转换
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    def predict(self, logits, transition, seq_lens):
        pre_seqs = []
        for score, seq_len in zip(logits, seq_lens):
            pre_seq, pre_score = tf.contrib.crf.viterbi_decode(score[:seq_len], transition)
            pre_seqs.append(pre_seq)
        return pre_seqs

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
        if len(char_item) > 0 and len(tag_item) > 0:  # 超出后循环外处理
            content = ''.join(char_item)
            key = tag_item[0][2:]
            position = (len(terms) - len(content), len(content))
            if key in raw_content.keys():
                raw_content[key].append((content, position))
            else:
                raw_content[key] = [(content, position)]
        return raw_content

    def encoder(self, embed):
        with tf.variable_scope("Transformer_Encoder"):
            # Positional Encoding
            embed += positional_encoding(self.x, num_units=self.num_units, zero_pad=False, scale=False,
                                         scope="enc_pe")
            # Dropout
            embed = tf.layers.dropout(embed, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            output = self.multi_head_block(embed, embed)
            return output

    def multi_head_block(self, query, key, decoding=False, causality=False):
        """
        多头注意力机制
        :param query:
        :param key:
        :param decoding:
        :param causality:
        :return:
        """
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                # multi head Attention ( self-attention)
                query = multihead_attention(
                    queries=query, keys=key, num_units=self.num_units, num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate, is_training=self.is_training, causality=causality,
                    scope="self_attention")
                if decoding:
                    # multi head Attention ( vanilla attention)
                    query = multihead_attention(
                        queries=query, keys=key, num_units=self.num_units, num_heads=self.num_heads,
                        dropout_rate=self.dropout_rate, is_training=self.is_training, causality=False,
                        scope="vanilla_attention")
                # Feed Forward
                query = feedforward(query, num_units=[4 * self.num_units, self.num_units])
        return query
