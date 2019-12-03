# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2019/11/27 18:12
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from model import TransformerCRFModel
import json
import tensorflow as tf
import numpy as np


class NerPredict():

    def __init__(self):
        self.vocab_file = 'vocab.json'
        self.model_dir = 'models\Transformer'
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
        self.id2tag = {_id: tag for tag, _id in self.classnames[0].items()}
        self.num_tags = len(self.id2tag)
        self.model = TransformerCRFModel(self.vocab_size, self.num_tags, self.sequence_length, self.num_units,
                                         self.dropout_rate, self.num_heads, is_training=False)
        self.session = None
        self.load()

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def load(self):
        saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt is not None and ckpt.model_checkpoint_path:
            saver.restore(self.session, ckpt.model_checkpoint_path)

    def close(self):
        if self.session is not None:
            self.session.close()

    def predict(self, input_text):
        input_text = input_text.strip().lower()
        char_vector = self.convert_xrow(input_text.strip().lower())
        seq_len_list = np.array([len(input_text)], dtype=np.int32)
        feed_dict = {self.model.x: np.array([char_vector], dtype=np.int32), self.model.seq_lens: seq_len_list}
        logits, transition_params = self.session.run([self.model.logits, self.model.transition], feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        taggs = []
        for i in range(seq_len_list[0]):
            taggs.append(self.id2tag[label_list[0][i]])
        output_labels = self.model.decode(list(input_text), taggs)
        print(output_labels)
        data_items = []
        if output_labels is not None and len(output_labels) > 0:
            for key in output_labels.keys():
                terms = [record[0] for record in output_labels[key]]
                value = ' '.join(terms)
                print(key)
                data_items.append(value)
        return " , ".join(data_items)

    def convert_xrow(self, input_text):
        char_vector = np.zeros((self.sequence_length), dtype=np.int32)
        for i in range(len(input_text)):
            char_value = input_text[i]
            if char_value in self.char_index.keys():
                char_vector[i] = self.char_index[char_value]
        return char_vector


if __name__ == '__main__':
    predict_temp = NerPredict()
    predict_temp.predict('高勇，男，中国国籍， 无境外居留权')