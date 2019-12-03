import pickle
import tensorflow as tf
from bert_lstm_model import Model
from al_bert import tokenization
import numpy as np


class bert_predict:
    def __init__(self, **kwargs):
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=kwargs['vocab_file'],
            do_lower_case=True)
        self.max_seq_len = 70
        self.map_file = kwargs['map_file']
        self.ckpt_path = kwargs['model_dir']
        self.init_checkpoint = kwargs['init_checkpoint_file']
        self.bert_config = kwargs['bert_config_dir']

        self.graph = kwargs["graph"]
        with self.graph.as_default():
            self.model = Model(init_checkpoint_file=self.init_checkpoint, bert_config_dir=self.bert_config)
            self.saver = tf.train.Saver()
        config = tf.ConfigProto(log_device_placement=False)
        self.session = tf.Session(graph=self.graph, config=config)
        self.load()

    def convert_single_example(self, char_line, tag_to_id, max_seq_length, tokenizer, label_line):
        """
        将一个样本进行分析，然后将字转化为id, 标签转化为id
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
        label_ids.append(tag_to_id["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(tag_to_id[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(tag_to_id["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        # padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("**NULL**")

        return input_ids, input_mask, segment_ids, label_ids

    def input_from_line(self, line, max_seq_length, tag_to_id):
        """
        Take sentence data and return an input for
        the training or the evaluation function.
        """
        string = [w[0].strip() for w in line]
        char_line = ' '.join(string)
        text = tokenization.convert_to_unicode(char_line)
        tags = ['O' for _ in string]
        labels = ' '.join(tags)
        labels = tokenization.convert_to_unicode(labels)
        ids, mask, segment_ids, label_ids = self.convert_single_example(char_line=text,
                                                                        tag_to_id=tag_to_id,
                                                                        max_seq_length=max_seq_length,
                                                                        tokenizer=self.tokenizer,
                                                                        label_line=labels)
        segment_ids = np.reshape(segment_ids, (1, max_seq_length))
        ids = np.reshape(ids, (1, max_seq_length))
        mask = np.reshape(mask, (1, max_seq_length))
        label_ids = np.reshape(label_ids, (1, max_seq_length))
        return [string, segment_ids, ids, mask, label_ids]

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt is not None and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            raise Exception("load classification failure...")

    def predict(self, input_text):
        with open(self.map_file, "rb") as f:
            tag_to_id, id_to_tag = pickle.load(f)
        result = self.model.evaluate_line(self.session, self.input_from_line(input_text, self.max_seq_len, tag_to_id),
                                          id_to_tag)
        data_items = {}
        if len(result['entities']) > 0:
            for i in range(len(result['entities'])):
                value = result['entities'][i]['word']
                data_items[result['entities'][i]['type']] = [value]
        return data_items


if __name__ == '__main__':
    class_bert_lstm_graph = tf.Graph()
    test = bert_predict(vocab_file='vocab.txt', map_file='maps.pkl'
                        , model_dir='albert', graph=class_bert_lstm_graph, init_checkpoint_file='albert_model.ckpt'
                        , bert_config_dir='albert_config_base.json')
    test.predict('')
