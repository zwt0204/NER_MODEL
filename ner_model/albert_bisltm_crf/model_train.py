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
import random
import math
import os
import json
from bert_lstm_model import Model
from al_bert import tokenization
import logging
_logger = logging.getLogger()


class modelTrain:
        
    def __init__(self):
        self.lstm_dim = 128
        self.batch_size = 128
        self.max_seq_len = 70
        self.clip = 5.0
        self.dropout_keep = 0.5
        self.optimizer = 'adam'
        self.lr = 0.001
        self.tag_schema = 'iob'
        self.ckpt_path = 'model'
        self.steps_check = 10
        self.zeros = False
        self.lower = True
        self.max_epoch = 10
        self.classnames = {'O': 0, 'I-BRD': 1, 'I-PRO': 2, 'B-PRO': 3, 'I-KWD': 4, 'B-BRD': 5, 'I-POP': 6, 'B-KWD': 7, 'B-POP': 8, 'I-PRC': 9, 'I-FLR': 10, 'B-FLR': 11, 'B-PRC': 12, '[CLS]': 13, '[SEP]': 14}
        self.num_tags = len(self.classnames)
        self.model = Model(init_checkpoint_file='albert_model.ckpt'
                        ,bert_config_dir='albert_config_base.json')
        self.saver = tf.train.Saver()

        self.tokenizer = tokenization.FullTokenizer(vocab_file='vocab.txt',
                                               do_lower_case=True)

    def train(self):
        path = 'samples.json'
        train_sentences = self.load_sentences(path)
        self.update_tag_scheme(train_sentences, self.tag_schema)
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

    def iob2(self, tags):
        """
        Check that tags have a valid IOB format.
        Tags in IOB1 format are converted to IOB2.
        """
        for i, tag in enumerate(tags):
            if tag == 'O':
                continue
            split = tag.split('-')
            if len(split) != 2 or split[0] not in ['I', 'B']:
                return False
            if split[0] == 'B':
                continue
            elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
                tags[i] = 'B' + tag[1:]
            elif tags[i - 1][1:] == tag[1:]:
                continue
            else:  # conversion IOB1 to IOB2
                tags[i] = 'B' + tag[1:]
        return True

    def update_tag_scheme(self, sentences, tag_scheme):
        """
        Check and update sentences tagging scheme to IOB2.
        Only IOB1 and IOB2 schemes are accepted.
        """
        for i, s in enumerate(sentences):
            tags = [w[-1] for w in s]
            # Check that tags are given in the IOB format
            if not self.iob2(tags):
                s_str = '\n'.join(' '.join(w) for w in s)
                raise Exception('Sentences should be given in IOB format! ' +
                                'Please check sentence %i:\n%s' % (i, s_str))
            if tag_scheme == 'iob':
                # If format was IOB1, we convert to IOB2
                for word, new_tag in zip(s, tags):
                    word[-1] = new_tag
            else:
                raise Exception('Unknown tagging scheme!')

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

    def prepare_dataset(self, sentences, max_seq_length, lower=False, train=True):
        """
        Prepare the dataset. Return a list of lists of dictionaries containing:
            - word indexes
            - word char indexes
            - tag indexes
        """
        data = []
        for s in sentences:
            string = [w[0].strip() for w in s]
            char_line = ' '.join(string)  # 使用空格把汉字拼起来
            text = tokenization.convert_to_unicode(char_line)

            if train:
                tags = [w[-1] for w in s]
            else:
                tags = ['O' for _ in string]

            labels = ' '.join(tags)  # 使用空格把标签拼起来
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
        label_ids.append(self.classnames["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(self.classnames[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(self.classnames["[SEP]"])
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


class BatchManager(object):
    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.arrange_batch(sorted_data[int(i*batch_size) : int((i+1)*batch_size)]))
        return batch_data

    @staticmethod
    def arrange_batch(batch):
        '''
        把batch整理为一个[5, ]的数组
        :param batch:
        :return:
        '''
        strings = []
        segment_ids = []
        chars = []
        mask = []
        targets = []
        for string, seg_ids, char, msk, target in batch:
            strings.append(string)
            segment_ids.append(seg_ids)
            chars.append(char)
            mask.append(msk)
            targets.append(target)
        return [strings, segment_ids, chars, mask, targets]

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, segment_ids, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = modelTrain()
    model.train()
