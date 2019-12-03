# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2019/11/28 11:49
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import numpy as np
import json
import tensorflow as tf
from .model import NerCore


class NerPredicter:
    def __init__(self, vocab_file="vocab.json"):
        self.model_dir = "model/ner"
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.io_sequence_size = 70
        vocab_size = len(self.char_index) + 1
        keep_prob = 1.0
        learning_rate = 0.001
        trainable = False
        self.batch_size = 64
        self.classnames = {'O': 0, 'B-BRD': 1, 'I-BRD': 2, 'B-KWD': 3, 'I-KWD': 4, 'B-POP': 5, 'I-POP': 6, 'B-PRC': 7,
                           'I-PRC': 8, 'B-FLR': 9, 'I-FLR': 10}
        class_size = len(self.classnames)
        self.classids = {}
        for key in self.classnames.keys():
            self.classids[self.classnames[key]] = key
        with tf.variable_scope('ner_query'):
            self.model = NerCore(self.io_sequence_size, vocab_size, class_size, keep_prob, learning_rate,
                                         trainable)
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

    def convert_xrow(self, input_text):
        char_vector = np.zeros((self.io_sequence_size), dtype=np.int32)
        for i in range(len(input_text)):
            char_value = input_text[i]
            if char_value in self.char_index.keys():
                char_vector[i] = self.char_index[char_value]
        return char_vector

    def predict(self, input_text):
        input_text = input_text.strip().lower()
        char_vector = self.convert_xrow(input_text.strip().lower())
        seq_len_list = np.array([len(input_text)], dtype=np.int32)
        feed_dict = {self.model.inputs: np.array([char_vector], dtype=np.float32),
                     self.model.sequence_lengths: seq_len_list}
        logits, transition_params = self.session.run([self.model.logits, self.model.transition_params], feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        taggs = []
        for i in range(seq_len_list[0]):
            taggs.append(self.classids[label_list[0][i]])
        print(taggs)
        output_labels = self.model.decode(list(input_text), taggs)
        print(output_labels)
        data_items = []
        if output_labels is not None and len(output_labels) > 0:
            for key in output_labels.keys():
                terms = [record[0] for record in output_labels[key]]
                value = ' '.join(terms)
                data_items.append(value)
        return " , ".join(data_items)


if __name__ == "__main__":
    predicter = NerPredicter()
    sents = ["123456"]
    for input_text in sents:
        line = predicter.predict(input_text)
        print("-> " + input_text)
        print("--> " + line)
