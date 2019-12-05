# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/12/4 19:54
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from model import Model_Lattice
import tensorflow as tf
import random
import numpy as np
import os
from utils.data import Data


class Train_Model:

    def __init__(self):

        self.gaz_file = 'D:\\mygit\\NER_MODEL\\data\\data\\ctb.50d.vec'
        self.char_emb = 'D:\\mygit\\NER_MODEL\\data\\data\\gigaword_chn.all.a2b.uni.ite50.vec'
        self.train_file = 'D:\\mygit\\NER_MODEL\\data\\data\\demo.train.char'
        self.dev_file = 'D:\\mygit\\NER_MODEL\\data\\data\\demo.dev.char'
        self.test_file = 'D:\\mygit\\NER_MODEL\\data\\data\\demo.test.char'
        self.model_save_path = 'D:\\mygit\\NER_MODEL\\models\\ckpt'

        self.batch_size = 64
        self.max_char_len = 100
        self.emb_size = 50
        self.max_lexicon_words_num = 5
        self.num_units = 128
        self.num_tags = 18
        self.learning_rate = 0.005
        self.optimizer = 'adam'
        self.epoch = 0
        self.bichar_emb = None
        self.data = Data()
        self.load_data_and_embedding()
        self.model = Model_Lattice(self.max_char_len, self.emb_size, self.max_lexicon_words_num, self.num_units, self.num_tags, self.learning_rate)
        self.saver = tf.train.Saver()

    def train(self, epochs=10):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        with tf.Session(config=config) as sess:
            sess.run(init)
            for iter in range(epochs):
                loss = []
                print('iter: ', iter)
                random.shuffle(self.data.train_Ids)
                train_num = len(self.data.train_Ids)
                total_batch = train_num // self.batch_size
                for batch_id in range(total_batch):
                    start = batch_id * self.batch_size
                    end = (batch_id + 1) * self.batch_size

                    if end > train_num:
                        end = train_num

                    instance = self.data.train_Ids[start:end]
                    if not instance:
                        continue

                    self.epoch += 1
                    _, char_ids, lexicon_word_ids, word_length_tensor, _, labels = self.batch_with_label(instance)

                    # run模型
                    feed_dict = {
                        self.model.placeholders["char_ids"]: char_ids,
                        self.model.placeholders["lexicon_word_ids"]: lexicon_word_ids,
                        self.model.placeholders["word_length_tensor"]: word_length_tensor,
                        self.model.placeholders["labels"]: labels,
                    }

                    _, losses, step = sess.run([self.model.train_op, self.model.loss, self.model.global_step], feed_dict = feed_dict)
                    loss.append(losses)
                    # print(loss)
                    self.ls = sum(loss) / len(loss)
                if self.epoch % 1 == 0:
                    print('*' * 100)
                    print(self.epoch, 'loss', self.ls)

                    # self.evaluate(sess, data)
                    self.evaluate_line(sess,
                                       ['习', '近', '平', '在', '北', '京', '中', '南', '海', '呼', '吁',
                                        '美', '国', '加', '强', '合', '作', '共', '创', '美', '好', '生', '活'])
                self.saver.save(sess, os.path.join(self.model_save_path, "ner.dat"))

    def batch_with_label(self, input_batch_list, is_train=True):
        """
        input: list of words, chars and labels, various length.
            [[words,biwords,chars,gaz,labels], [words,biwords,chars,gaz,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for one sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            char_ids: (batch_size, )
            lexicon_word_ids: (batch_size, )
            word_length_tensor: (batch_size, )
            labels: (batch_size, )
        """
        # batch_size = len(input_batch_list)
        lengths = [len(sent[0][0:self.max_char_len]) for sent in input_batch_list]
        chars_ids = [sent[0][0:self.max_char_len] for sent in input_batch_list]
        biwords = [sent[1][0:self.max_char_len] for sent in input_batch_list]
        # chars_ids_split = [sent[2][0:self.max_char_len] for sent in input_batch_list]
        # lexicon_words = [sent[3][0:self.max_char_len] for sent in input_batch_list]

        if is_train:
            target = [sent[4][0:self.max_char_len] for sent in input_batch_list]

        chars_ids = list(map(lambda l: l + [0] * (self.max_char_len - len(l)), chars_ids))
        # biwords = list(map(lambda l: l + [0] * (self.max_char_len - len(l)), biwords))

        if is_train:
            labels = list(map(lambda l: l + [0] * (self.max_char_len - len(l)), target))

        lexicon_word_ids = []
        word_length_tensor = []
        for sent in input_batch_list:
            lexicon_word_ids_sent = []
            word_length_tensor_sent = []

            for word_lexicon in sent[3][0:self.max_char_len]:
                word_lexicon_pad = list(map(lambda l:
                                            l + [0] * (self.max_lexicon_words_num - len(l)),
                                            word_lexicon))
                lexicon_word_ids_sent.append(word_lexicon_pad[0][0:self.max_lexicon_words_num])    # id
                word_length_tensor_sent.append(word_lexicon_pad[1][0:self.max_lexicon_words_num])  # length

            lexicon_word_ids.append(lexicon_word_ids_sent)
            word_length_tensor.append(word_length_tensor_sent)

        lexicon_word_ids = list(map(lambda l:
                                    l + [[0] * self.max_lexicon_words_num] * (self.max_char_len - len(l)),
                                    lexicon_word_ids))
        word_length_tensor = list(map(lambda l:
                                      l + [[0] * self.max_lexicon_words_num] * (self.max_char_len - len(l)),
                                      word_length_tensor))

        if is_train:
            return lengths, chars_ids, lexicon_word_ids, word_length_tensor, target, labels

        return lengths, chars_ids, lexicon_word_ids, word_length_tensor

    def evaluate_line(self, sess, sentence, ):
        '''
        因LatticeLSTM内部参数受batch_size限制，数据会转为批处理
        :param sess: 会话
        :param sentence: 带处理文本
        :param self.data: 含词库等处理的数据集
        :return: 返回标注结果
        '''
        _, Ids = self.data.generate_sentence_instance_with_gaz(sentence)
        lengths, char_ids, lexicon_word_ids, word_length_tensor = self.batch_with_label(Ids, False)

        lengths = lengths * self.batch_size
        char_ids = char_ids * self.batch_size
        lexicon_word_ids = lexicon_word_ids * self.batch_size
        word_length_tensor = word_length_tensor * self.batch_size

        # run模型
        feed_dict = {
            self.model.placeholders["char_ids"]: char_ids,
            self.model.placeholders["lexicon_word_ids"]: lexicon_word_ids,
            self.model.placeholders["word_length_tensor"]: word_length_tensor,
        }

        logits = sess.run(self.model.logits, feed_dict=feed_dict)
        paths = self.decode(logits, lengths, self.model.trans.eval(session=sess))
        tags = [self.data.label_alphabet.get_instance(idx) for idx in paths[0]]
        print("tags: ", tags)

        return tags

    def decode(self, logits, lengths, transition_matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param transition_matrix: transaction matrix for inference
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
            path, _ = tf.contrib.crf.viterbi_decode(logits, transition_matrix)

            paths.append(path[1:])

        return paths
    
    def load_data_and_embedding(self):
        self.data.HP_use_char = False
        self.data.HP_batch_size = 1
        self.data.use_bigram = False
        self.data.gaz_dropout = 0.5
        self.data.norm_gaz_emb = False
        self.data.HP_fix_gaz_emb = False

        self.data_initialization()

        self.data.generate_instance_with_gaz(self.train_file, 'train')
        self.data.generate_instance_with_gaz(self.dev_file, 'dev')
        self.data.generate_instance_with_gaz(self.test_file, 'test')

        self.data.build_word_pretrain_emb(self.char_emb)
        self.data.build_biword_pretrain_emb(self.bichar_emb)
        self.data.build_gaz_pretrain_emb(self.gaz_file)

    def data_initialization(self):
        self.data.build_alphabet(self.train_file)
        self.data.build_alphabet(self.dev_file)
        self.data.build_alphabet(self.test_file)

        self.data.build_gaz_file(self.gaz_file)

        self.data.build_gaz_alphabet(self.train_file)
        self.data.build_gaz_alphabet(self.dev_file)
        self.data.build_gaz_alphabet(self.test_file)
        self.data.fix_alphabet()


if __name__ == '__main__':
    demo_train = Train_Model()
    demo_train.load_data_and_embedding()
    demo_train.train()
