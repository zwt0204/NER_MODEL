# -*- encoding: utf-8 -*-
"""
@File    : run.py
@Time    : 2019/11/28 17:17
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from ner_model.crf.CRF import CRF
from ner_model.crf.utils import *


def train_crf():
    max_len = 70
    word2id, id2word = load_data('vocab.txt')
    tag2id, id2tag = load_data('tag.txt')
    _, _, train_, x_train, y_train = generate_data('train.txt', word2id, tag2id, max_len=max_len)
    _, _, dev_seq_lens, x_dev, y_dev = generate_data('dev.txt', word2id, tag2id, max_len=max_len)
    model_file = "model_crf"
    model = CRF()
    model.fit(x_train, y_train, template_file='templates.txt', model_file=model_file, max_iter=20)
    pre_seq = model.predict(x_dev, model_file=model_file)
    acc, p, r, f = get_ner_fmeasure(y_dev, pre_seq)
    print('acc:\t{}\tp:\t{}\tr:\t{}\tf:\t{}\n'.format(acc, p, r, f))
