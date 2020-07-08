# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from albert_bisltm_crf.al_bert import modeling
from data_process import bio_to_json
# import tensorflow_addons as tfa


class Model(object):
    def __init__(self, init_checkpoint_file, bert_config_dir):
        self.lr = 0.001
        self.hidden_size = 256
        self.num_tags = 15
        self.dropout_keep = 0.5
        self.optimizer = 'adam'
        self.global_step = tf.Variable(0, trainable=False)
        self.initializer = initializers.xavier_initializer()
        self.init_checkpoint = init_checkpoint_file
        self.bert_config = bert_config_dir

        # add placeholders for the model
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

        used = tf.sign(tf.abs(self.input_ids))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.input_ids)[0]
        self.num_steps = tf.shape(self.input_ids)[-1]

        # embeddings for chinese character and segmentation representation
        embedding = self.bert_embedding()

        # apply dropout before feed to lstm layer
        inputs = tf.nn.dropout(embedding, self.dropout)

        # logits for tags
        self.logits = self.project_layer(inputs)

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        # bert模型参数初始化的地方
        # init_checkpoint = "chinese_L-12_H-768_A-12/bert_model.ckpt"
        init_checkpoint = self.init_checkpoint
        # 获取模型中所有的训练参数。
        tvars = tf.trainable_variables()
        # 加载BERT模型
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        # print("**** Trainable Variables ****")
        # 打印加载模型的参数
        train_vars = []
        for var in tvars:
            if var.name not in initialized_variable_names:
                train_vars.append(var)
        with tf.variable_scope("optimizer"):
            optimizer = self.optimizer
            if optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            else:
                raise KeyError

            grads = tf.gradients(self.loss, train_vars)
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            self.train_op = self.opt.apply_gradients(
                zip(grads, train_vars), global_step=self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def bert_embedding(self):
        # load bert embedding
        bert_config = modeling.BertConfig.from_json_file(self.bert_config)  # 配置文件地址。
        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        # 这个获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
        embedding = model.get_sequence_output()
        return embedding

    def project_layer(self, outputs, name=None):
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[786, self.hidden_size],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.hidden_size], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(outputs, shape=[-1, 786])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_size, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        with tf.variable_scope("crf_loss" if not name else name):
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=project_logits,
                tag_indices=self.targets,
                sequence_lengths=lengths)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):

        _, segment_ids, chars, mask, tags = batch
        feed_dict = {
            self.input_ids: np.asarray(chars),
            self.input_mask: np.asarray(mask),
            self.segment_ids: np.asarray(segment_ids),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.dropout_keep
        return feed_dict

    def run_step(self, sess, is_train, batch):
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        results = []
        trans = self.trans.eval(sess)
        for batch in data_manager.iter_batch():
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for batch_path in batch_paths:
                tags = [id_to_tag[idx] for idx in batch_path]
                results.append(tags)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return bio_to_json(inputs[0], tags[1:-1])