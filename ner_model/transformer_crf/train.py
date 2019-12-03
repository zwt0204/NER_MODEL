# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/11/27 17:24
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from model import TransformerCRFModel
import json
import tensorflow as tf
import numpy as np
import os


class Train_model():

    def __init__(self):
        self.vocab_file = 'D:\zwt\work_test\qa_robot_test\\resource\\vocab\dictionary.json'
        self.model_dir = 'D:\mygit\\tf1.0\models\Transformer'
        self.batch_size = 128
        self.lr = 0.0001
        self.sequence_length = 50
        self.num_units = 512
        self.num_blocks = 6
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.char_index = {' ': 0}
        self.unknow_char_id = len(self.char_index)
        self.load_dict()
        self.vocab_size = len(self.char_index)
        self.classnames = [{"O": 0, "M-ORG": 1, "M-TITLE": 2, "B-TITLE": 3, "E-TITLE": 4, "B-ORG": 5, "E-ORG": 6,
                           "M-EDU": 7, "B-NAME": 8, "E-NAME": 9, "B-EDU": 10, "E-EDU": 11, "M-NAME": 12, "M-PRO": 13,
                           "M-CONT": 14, "B-PRO": 15, "E-PRO": 16, "B-CONT": 17, "E-CONT": 18, "M-LOC": 19,
                           "B-RACE": 20, "E-RACE": 21, "S-NAME": 22, "B-LOC": 23, "E-LOC": 24, "M-RACE": 25,
                           "S-RACE": 26, "S-ORG": 27}]
        self.id2tag = {str(_id): tag for tag, _id in self.classnames[0].items()}
        self.num_tags = len(self.id2tag)
        self.model = TransformerCRFModel(self.vocab_size, self.num_tags, self.sequence_length, self.num_units,
                                         self.dropout_rate, self.num_heads, is_training=True)

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def train(self, epochs):
        x_train, y_train, seq_lens, _, _ = self.generate_data('D:\model\\transformer_crf-master\data\\train.char.bmes', self.char_index, self.classnames[0],
                                                              max_len=self.sequence_length)
        x_dev, y_dev, dev_seq_lens, _, source_tag = self.generate_data('D:\model\\transformer_crf-master\data\dev.char.bmes', self.char_index, self.classnames[0],
                                                                       max_len=self.sequence_length)
        initer = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(initer)
            saver = tf.train.Saver()
            for epoch in range(1, epochs + 1):
                train_loss = []
                for x_batch, y_batch, len_batch in self.batch_data(x_train, y_train, seq_lens, self.batch_size):
                    feed_dict = {self.model.x: x_batch, self.model.y: y_batch, self.model.seq_lens: len_batch}
                    loss = sess.run([self.model.loss], feed_dict=feed_dict)
                    train_loss.append(loss)
                dev_loss = []
                predict_lists = []
                for x_batch, y_batch, len_batch in self.batch_data(x_dev, y_dev, dev_seq_lens, self.batch_size):
                    feed_dict = {self.model.x: x_batch, self.model.y: y_batch, self.model.seq_lens: len_batch}
                    loss, logits = sess.run([self.model.loss, self.model.logits], feed_dict)
                    dev_loss.append(loss)

                    transition = self.model.transition.eval(session=sess)
                    pre_seq = self.model.predict(logits, transition, len_batch)
                    pre_label = self.recover_label(pre_seq, len_batch, self.id2tag)
                    predict_lists.extend(pre_label)
                train_loss_v = np.round(float(np.mean(train_loss)), 4)
                dev_loss_v = np.round(float(np.mean(dev_loss)), 4)
                print('****************************************************')
                acc, p, r, f = self.get_ner_fmeasure(source_tag, predict_lists)
                print('epoch:\t{}\ttrain loss:\t{}\tdev loss:\t{}'.format(epoch, train_loss_v, dev_loss_v))
                print('acc:\t{}\tp:\t{}\tr:\t{}\tf:\t{}'.format(acc, p, r, f))
                print('****************************************************\n\n')
                saver.save(sess, os.path.join(self.model_dir, "ner.dat"))

    def generate_data(self, filename, char_index, classnames, max_len=50):
        """
        pad 补全<max_len 数据
        :param filename:
        :param self.char_index:
        :param self.classnames:
        :param max_len:
        :return:
        """
        _sentences, _tags, _seq_lens = self.read_data(filename)

        sentences = []
        tags = []
        seq_lens = []
        source_sentences = []
        source_tags = []
        for _sentence, seq_len in zip(_sentences, _seq_lens):
            sentence = [char_index.get(word, 0) for word in _sentence]
            if seq_len <= max_len:
                sentence += [0] * (max_len - seq_len)
                seq_lens.append(seq_len)
                sentences.append(sentence)
                source_sentences.append(_sentence)
        default = 0
        for _tag, seq_len in zip(_tags, _seq_lens):
            tag = [classnames.get(label, default) for label in _tag]
            if seq_len <= max_len:
                tag += [default] * (max_len - seq_len)
                tags.append(tag)
                source_tags.append(_tag)
        return sentences, tags, seq_lens, source_sentences, source_tags

    def batch_data(self, x, y, seq_lens, batch_size):
        """
        生成小批量数据
        :param x:
        :param y:
        :param seq_lens:
        :param batch_size:
        :return:
        """
        total_batch = len(seq_lens) // batch_size + 1

        for ii in range(total_batch):
            start, end = ii * batch_size, (ii + 1) * batch_size

            x_batch = np.array(x[start:end], dtype=np.int32)
            y_batch = np.array(y[start:end], dtype=np.int32)
            len_batch = np.array(seq_lens[start:end], dtype=np.int32)
            yield x_batch, y_batch, len_batch

    def read_data(self, filename):
        """
        读取标注数据
        :param filename:
        :return:
        """
        sentences = []
        tags = []
        seq_lens = []
        with open(filename, 'r', encoding='utf8') as fp:
            sentence = []
            tag = []
            for line in fp.readlines():
                token = line.strip().split()
                if len(token) == 2:
                    sentence.append(token[0])
                    tag.append(token[1])
                elif len(token) == 0 and len(sentence) > 0:
                    assert len(sentence) == len(tag)
                    seq_lens.append(len(sentence))
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence = []
                    tag = []
            if sentence:
                assert len(sentence) == len(tag)
                seq_lens.append(len(sentence))
                sentences.append(sentence)
                tags.append(tag)
        return sentences, tags, seq_lens

    def recover_label(self, tags, seq_lens, id2tag):
        """
        恢复标签 ID->tag
        :param tags:
        :param seq_lens:
        :param id2tag:
        :return:
        """
        labels = []
        for tag, seq_len in zip(tags, seq_lens):
            pre = [id2tag.get(str(_id), 'O') for _id in tag[:seq_len]]
            labels.append(pre)
        return labels

    def get_ner_fmeasure(self, golden_lists, predict_lists, label_type="BMES"):
        golden_full = []
        predict_full = []
        right_full = []
        right_tag = 0
        all_tag = 0
        for idx, (golden_list, predict_list) in enumerate(zip(golden_lists, predict_lists)):
            for golden_tag, predict_tag in zip(golden_list, predict_list):
                if golden_tag == predict_tag:
                    right_tag += 1
            all_tag += len(golden_list)
            if label_type == "BMES":
                gold_matrix = self.get_ner_BMES(golden_list)
                pred_matrix = self.get_ner_BMES(predict_list)
            else:
                gold_matrix = self.get_ner_BIO(golden_list)
                pred_matrix = self.get_ner_BIO(predict_list)
            # 交集
            right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
            golden_full += gold_matrix
            predict_full += pred_matrix
            right_full += right_ner
        right_num = len(right_full)
        golden_num = len(golden_full)
        predict_num = len(predict_full)
        if predict_num == 0:
            precision = -1
        else:
            precision = (right_num + 0.0) / predict_num
        if golden_num == 0:
            recall = -1
        else:
            recall = (right_num + 0.0) / golden_num
        if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
            f_measure = -1
        else:
            f_measure = 2 * precision * recall / (precision + recall)
        accuracy = (right_tag + 0.0) / all_tag
        print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
        return round(accuracy, 4), round(precision, 4), round(recall, 4), round(f_measure, 4)

    def get_ner_BMES(self, label_list):
        list_len = len(label_list)
        begin_label = 'B'
        end_label = 'E'
        single_label = 'S'
        whole_tag = ''
        index_tag = ''
        tag_list = []
        stand_matrix = []
        for i in range(list_len):
            current_label = label_list[i].upper()
            tags = current_label.split('-')
            if begin_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = tags[-1] + '[' + str(i)
                index_tag = tags[-1]

            elif single_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = tags[-1] + '[' + str(i)
                tag_list.append(whole_tag)
                whole_tag = ""
                index_tag = ""
            elif end_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i))
                whole_tag = ''
                index_tag = ''
            else:
                continue
        if (whole_tag != '') & (index_tag != ''):
            tag_list.append(whole_tag)
        tag_list_len = len(tag_list)

        for i in range(0, tag_list_len):
            if len(tag_list[i]) > 0:
                tag_list[i] = tag_list[i] + ']'
                insert_list = self.reverse_style(tag_list[i])
                stand_matrix.append(insert_list)
        return stand_matrix

    def get_ner_BIO(self, label_list):
        list_len = len(label_list)
        begin_label = 'B-'
        inside_label = 'I-'
        whole_tag = ''
        index_tag = ''
        tag_list = []
        stand_matrix = []
        for i in range(0, list_len):
            current_label = label_list[i].upper()
            if begin_label in current_label:
                if index_tag == '':
                    whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                    index_tag = current_label.replace(begin_label, "", 1)
                else:
                    tag_list.append(whole_tag + ',' + str(i - 1))
                    whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                    index_tag = current_label.replace(begin_label, "", 1)

            elif inside_label in current_label:
                if current_label.replace(inside_label, "", 1) == index_tag:
                    whole_tag = whole_tag
                else:
                    if (whole_tag != '') & (index_tag != ''):
                        tag_list.append(whole_tag + ',' + str(i - 1))
                    whole_tag = ''
                    index_tag = ''
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''

        if (whole_tag != '') & (index_tag != ''):
            tag_list.append(whole_tag)
        tag_list_len = len(tag_list)

        for i in range(0, tag_list_len):
            if len(tag_list[i]) > 0:
                tag_list[i] = tag_list[i] + ']'
                insert_list = self.reverse_style(tag_list[i])
                stand_matrix.append(insert_list)
        return stand_matrix

    def reverse_style(self, input_string):
        target_position = input_string.index('[')
        input_len = len(input_string)
        output_string = input_string[target_position:input_len] + input_string[0:target_position]
        return output_string


if __name__ == '__main__':
    train_temp = Train_model()
    train_temp.train(10)