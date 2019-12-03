# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/11/28 11:45
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import os
import numpy as np
import json
import tensorflow as tf
from ner_model.bilstm_cnn_crf.model import NerCore


class NerTrainner:
    def __init__(self, vocab_file="vocab.json"):
        self.model_dir = "ner"
        self.vocab_file = vocab_file
        self.char_index = {' ': 0}
        self.load_dict()
        self.unknow_char_id = len(self.char_index)
        self.io_sequence_size = 70
        vocab_size = len(self.char_index) + 1
        self.classnames = {'O': 0, 'B-BRD': 1, 'I-BRD': 2, 'B-KWD': 3, 'I-KWD': 4, 'B-POP': 5, 'I-POP': 6, 'B-PRO': 7,
                           'I-PRO': 8, 'B-PRC': 9, 'I-PRC': 10, 'B-FLR': 11, 'I-FLR': 12}
        class_size = len(self.classnames)
        keep_prob = 0.5
        learning_rate = 0.0005
        trainable = True
        self.batch_size = 1

        with tf.variable_scope('ner_query'):
            self.model = NerCore(self.io_sequence_size, vocab_size, class_size, keep_prob, learning_rate,
                                         trainable)

    def load_dict(self):
        i = 0
        with open(self.vocab_file, "r+", encoding="utf-8") as reader:
            items = json.load(reader)
            for charvalue in items:
                self.char_index[charvalue.strip()] = i + 1
                i += 1

    def train(self, epochs):
        records = self.load_samples()
        batch_count = int(len(records) / self.batch_size)
        initer = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(initer)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            saver = tf.train.Saver()
            if ckpt is not None and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            for epoch in range(epochs):
                train_loss_value = 0.
                for i in range(batch_count):
                    batch_records = records[i * self.batch_size:(i + 1) * self.batch_size]
                    xrows, xlens, yrows = self.convert_batch(batch_records)
                    feed_dict = {self.model.inputs: xrows, self.model.targets: yrows,
                                 self.model.sequence_lengths: xlens}
                    batch_loss_value, _ = session.run([self.model.cost_func, self.model.optimizer], feed_dict)
                    train_loss_value += batch_loss_value / batch_count
                    if i % 100 == 0:
                        batch_buffer = "Progress {0}/{1} , cost : {2}".format(i + 1, batch_count, batch_loss_value)
                        print(batch_buffer)
                print("Epoch: %d/%d , train cost=%f " % ((epoch + 1), epochs, train_loss_value))
                saver.save(session, os.path.join(self.model_dir, "ner.dat"))

    def convert_batch(self, records):
        xrows = np.zeros((self.batch_size, self.io_sequence_size), dtype=np.float32)
        xlens = np.zeros((self.batch_size), dtype=np.int32)
        yrows = np.zeros((self.batch_size, self.io_sequence_size), dtype=np.int32)
        count = len(records)
        for i in range(count):
            sent_text = records[i]["text"]
            tags = records[i]["label"].split(" ")
            xlen = len(records[i]["text"])
            if xlen > self.io_sequence_size:
                print(xlen)
            xlens[i] = xlen
            xrows[i] = self.convert_xrow(sent_text)
            yrows[i] = self.convert_classids(tags)
        return xrows, xlens, yrows

    def convert_classids(self, tags):
        yrow = np.zeros(self.io_sequence_size, dtype=np.int32)
        for i in range(len(tags)):
            yrow[i] = self.classnames[tags[i]]
        return yrow

    def convert_xrow(self, input_text):
        char_vector = np.zeros((self.io_sequence_size), dtype=np.int32)
        for i in range(len(input_text)):
            char_value = input_text[i]
            if char_value in self.char_index.keys():
                char_vector[i] = self.char_index[char_value]
        return char_vector

    def load_samples(self, dstfile="train.json"):
        data_items = []
        with open(dstfile, "r", encoding="utf-8") as reader:
            for line in reader:
                record = json.loads(line.strip(), encoding="utf-8")
                data_items.append(record)
        return data_items


if __name__ == "__main__":
    trainner = NerTrainner()
    trainner.train(10)