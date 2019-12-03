# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2019/12/2 15:54
@Author  : zwt
@git   : https://github.com/crownpku/Information-Extraction-Chinese
@Software: PyCharm
"""
import numpy as np
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from IDCNN_CRF.utils import *


class IDCNN_MODEL:

    def __init__(self, lr, char_dim, seg_dim, num_tags, num_chars):
        self.lr = lr
        self.char_dim = char_dim
        # 增加的维度
        self.seg_dim = seg_dim
        # 标签数量
        self.num_tags = num_tags
        # 字典维度
        self.num_chars = num_chars
        # 0，1，2，3，0是不需要的字，1是第一个，2是中间的，3是最后一个
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)

        # add placeholders for the model
        # 字输入
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ChatInputs")
        # 上下文输入
        self.seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")
        # 目标
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")
        # tf.sign符号函数
        used = tf.sign(tf.abs(self.char_inputs))
        # reduction_indices=1， 按照列求和
        length = tf.reduce_sum(used, reduction_indices=1)
        # 句子实际长度
        self.lengths = tf.cast(length, tf.int32)
        # 批次大小
        self.batch_size = tf.shape(self.char_inputs)[0]
        # 句子最大长度
        self.sequence_length = tf.shape(self.char_inputs)[-1]

        # parameters for idcnn
        # 每次有3次卷积操作，前两次卷积膨胀系数为1，后一次膨胀系数为2
        # 膨胀卷积 膨胀卷积核尺寸 = 膨胀系数 *（原始卷积核尺寸-1）+1
        self.layers = [{'dilation': 1}, {'dilation': 1}, {'dilation': 2}]
        #
        self.num_filter = 100
        self.filter_width = 3
        self.keep_out = 0.5
        # 总的嵌入维度
        self.embedding_dim = self.char_dim + self.seg_dim
        # 重复4次增强特征提取能力
        self.repeat_times = 4
        # 梯度裁剪
        self.clip = 5
        # self.cnn_output_width = 0
        self.initializer = initializers.xavier_initializer()
        self.create_declare()
        self.create_model()
        self.create_loss()
        # saver of the model
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables, max_to_keep=5)

    def create_declare(self):
        """
         高：3 血：22 糖：23 和：24 高：3 血：22 压：25 char_inputs=[3,22,23,24,3,22,25]
         seg_num:个人理解为类似于上下文的意思：
         高血糖 和 高血压 seg_inputs 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3] seg_inputs=[1,2,3,0,1,2,3]
        :return:
        """
        embedding = []
        with tf.variable_scope("char_embedding"), tf.device('/cpu:0'):
            # [vocab_size, embedding]
            self.char_lookup = tf.get_variable(
                name="char_embedding",
                shape=[self.num_chars, self.char_dim],
                initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, self.char_inputs))
            if self.seg_dim:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, self.seg_inputs))
            self.embed = tf.concat(embedding, axis=-1)
            # [None, sequence_length, embedding_size]
            self.model_inputs = tf.nn.dropout(self.embed, self.dropout)

    def create_model(self):
        # tf.expand_dims增加一个维度
        """
        由于conv2d的参数列表： [batch, in_height, in_width, in_channels]
        就差一个in_channels
        针对当前的计算，嵌入维度作为通道，差了in_height，所以在第二个维度增加1
        """
        # [batch_size, 1, sequence_lenght, embedding_size]
        model_inputs = tf.expand_dims(self.model_inputs, 1)
        # reuse=False时，函数get_variable（）表示创建变量
        # reuse=True时，函数get_variable（）表示获取变量
        reuse = False
        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn"):
            # [卷积核的高度，卷积核的宽度，通道数，卷积核个数]
            shape = [1, self.filter_width, self.embedding_dim,
                     self.num_filter]
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=shape,
                initializer=self.initializer)
            # input : 输入的要做卷积的矩阵，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]
            # filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，
            # 其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，
            # in_channel 是通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
            # strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
            # padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，
            # 不足的时候用0去填充周围，"VALID"则不考虑
            # 返回feature map：[batch, height, width, channels]
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            # 多次卷积，就会将膨胀的时候单次没有卷到的数据在下次卷到
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        w = tf.get_variable("filterW", shape=[1, self.filter_width, self.num_filter, self.num_filter],
                                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        # 输入要做卷积的矩阵：[batch, height, width, channels]
                        # w:[batch, height, width, out_channels]
                        # rate=1时表示普通的卷积
                        """
                        填充方式为“VALID”时，返回[batch,height-2*(filter_width-1),width-2*(filter_height-1),out_channels]的Tensor，
                        填充方式为“SAME”时，返回[batch, height, width, out_channels]的Tensor
                        """
                        # 空洞卷积的时候padding一定要注意，因为卷积核可能比输入还要打，所以尽量使用padding=‘SAME’
                        conv = tf.nn.atrous_conv2d(layerInput, w, rate=dilation, padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            # 该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果
            # 删除掉height
            finalOut = tf.squeeze(finalOut, [1])
            self.finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.totalWidthForLastDim = totalWidthForLastDim
        with tf.variable_scope("project"):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.totalWidthForLastDim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[self.num_tags]))
                pred = tf.nn.xw_plus_b(self.finalOut, W, b)
                # [batch_size, sequence_length, num_tags]
            self.logits = tf.reshape(pred, [-1, self.sequence_length, self.num_tags])

    def create_loss(self):
        with tf.variable_scope("crf_loss"):
            small = -1000.0
            # sequence_length是句子长度；pad_logits是特征提取并全连接后的输出
            # [batch_size,1,num_tags+1]
            """
            start_logits作用：增加了句子开头的标志和结束的标志，并把score赋值为很小的值
            这样就多了开头标签到第一个字标签的转移score
            """
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1)
            # [batch_size, sequence_length, 1]
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.sequence_length, 1]), tf.float32)
            # [batch_size,sequence_length,num_tags+1]
            logits = tf.concat([self.logits, pad_logits], axis=-1)
            # [batch_size,sequence_length+1,num_tags+1]
            logits = tf.concat([start_logits, logits], axis=1)
            # tf.cast类型转换
            targets = tf.concat(
                [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            # transition_params:形状为[num_tags,num_tags]的转移矩阵
            # inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor
            # tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签.
            # sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度.
            # 这是一个样本真实的序列长度，因为为了对齐长度会做些padding，但是可以把真实的长度放到这个参数里
            """
            log_likelihood: 标量,log-likelihood 
            transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            """
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=self.lengths + 1)
            self.loss = tf.reduce_mean(-log_likelihood)
            self.opt = tf.train.AdagradOptimizer(self.lr)
            grads_vars = self.opt.compute_gradients(self.loss)
            # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
            # 小于min的让它等于min，大于max的元素的值等于max。
            capped_grads_vars = [[tf.clip_by_value(g, -self.clip, self.clip), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.keep_out
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss, feed_dict
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits, feed_dict

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, sequence_length, num_tags]float32, logits
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
            """
            viterbi_decode(score,transition_params):
            score: 一个形状为[seq_len, num_tags] matrix of unary potentials. 
            transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            返回：
            viterbi: 一个形状为[seq_len] 显示了最高分的标签索引的列表. 
            viterbi_score: A float containing the score for the Viterbi sequence.
            """
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores, _ = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
