# -*- encoding: utf-8 -*-
"""
@File    : model_train.py
@Time    : 2019/10/14 13:18
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import codecs
import tensorflow as tf
import numpy as np
import os
import json
from bert_lstm_model import Model
from al_bert import tokenization
import logging
from data_process import BatchManager, convert_samples

_logger = logging.getLogger()


class ModelTrain:

    def __init__(self):
        self.lstm_dim = 128
        self.batch_size = 1
        self.max_seq_len = 70
        self.clip = 5.0
        self.dropout_keep = 0.5
        self.optimizer = 'adam'
        self.lr = 0.001
        self.tag_schema = 'iob'
        self.ckpt_path = '..\\models'
        self.steps_check = 10
        self.zeros = False
        self.lower = True
        self.max_epoch = 2
        self.num_tags = len(convert_samples.tag_to_id)
        self.model = Model(init_checkpoint_file='D:\models\\albert_base_zh\\albert_model.ckpt'
                           , bert_config_dir='D:\models\\albert_base_zh\\albert_config_base.json')
        self.saver = tf.train.Saver()

        self.tokenizer = tokenization.FullTokenizer(vocab_file='D:\models\\albert_base_zh\\vocab.txt',
                                                    do_lower_case=True)

    def train(self):
        path = '..\data\\train.json'
        train_sentences = self.load_sentences(path)
        train_data = self.prepare_dataset(
            train_sentences, self.max_seq_len, self.lower)
        train_manager = BatchManager(train_data, self.batch_size)
        init = tf.global_variables_initializer()
        steps_per_epoch = train_manager.len_data
        with tf.Session() as sess:
            loss = []
            sess.run(init)
            for i in range(self.max_epoch):
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss = self.model.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if step % self.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        print("iteration:{} step:{}/{}, "
                              "NER loss:{:>9.6f}".format(
                            iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []
                    self.save_model(sess, self.model, self.ckpt_path, global_steps=step)

    def load_sentences(self, path):
        """
        Load sentences. A line must contain at least a word and its tag.
        Sentences are separated by empty lines.
        """
        sentences = []
        num = 0
        for j, line in enumerate(codecs.open(path, 'r', 'utf8')):
            sentence = []
            num += 1
            data = json.loads(line)
            list_lable = str(data['label']).split(' ')
            for i, value in enumerate(list(data['text'])):
                temp = []
                temp.append(value)
                temp.append(list_lable[i])
                sentence.append(temp)
            sentences.append(sentence)
        return sentences

    def save_model(self, sess, model, path, global_steps):
        checkpoint_path = os.path.join(path, "ner.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=global_steps)

    def prepare_dataset(self, sentences, max_seq_length, lower=None, train=True):
        """
        Prepare the dataset. Return a list of lists of dictionaries containing:
            - word indexes
            - word char indexes
            - tag indexes
        """
        data = []
        for s in sentences:
            if lower:
                string = [w[0].strip().lower() for w in s]
            else:
                string = [w[0].strip() for w in s]
            char_line = ' '.join(string)
            text = tokenization.convert_to_unicode(char_line)

            if train:
                tags = [w[-1] for w in s]
            else:
                tags = ['O' for _ in string]

            labels = ' '.join(tags)
            labels = tokenization.convert_to_unicode(labels)

            ids, mask, segment_ids, label_ids = self.convert_single_example(char_line=text,
                                                                            max_seq_length=max_seq_length,
                                                                            tokenizer=self.tokenizer,
                                                                            label_line=labels)
            data.append([string, segment_ids, ids, mask, label_ids])

        return data

    def convert_single_example(self, char_line, max_seq_length, tokenizer, label_line):
        """
        将一个样本进行分析，然后将字转化为id, 标签转化为lb
        """
        text_list = char_line.split(' ')
        label_list = label_line.split(' ')

        tokens = []
        labels = []
        for i, word in enumerate(text_list):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        # 序列截断
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        label_ids.append(convert_samples.tag_to_id["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(convert_samples.tag_to_id[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(convert_samples.tag_to_id["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        # padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")

        return input_ids, input_mask, segment_ids, label_ids


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = ModelTrain()
    model.train()
