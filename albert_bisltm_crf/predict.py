import tensorflow as tf
from bert_lstm_model import Model
from al_bert import tokenization
from data_process import BatchManager
from data_process import bio_to_json
from data_process import convert_samples
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)


class bert_predict:
    def __init__(self, **kwargs):
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=kwargs['vocab_file'],
            # vocab_file='D:\\mygit\\NER_MODEL\\albert_tiny_489k\\vocab.txt',
            do_lower_case=True)
        self.max_seq_len = 70
        self.ckpt_path = kwargs['model_dir']
        # self.ckpt_path = 'D:\\mygit\\NER_MODEL\\models'
        self.init_checkpoint = kwargs['init_checkpoint_file']
        # self.init_checkpoint = 'D:\\mygit\\NER_MODEL\\albert_tiny_489k\\albert_model.ckpt'
        self.bert_config = kwargs['bert_config_dir']
        # self.bert_config = 'D:\\mygit\\NER_MODEL\\albert_tiny_489k\\albert_config_tiny.json'

        self.graph = kwargs["graph"]
        with self.graph.as_default():
            self.model = Model(init_checkpoint_file=self.init_checkpoint, bert_config_dir=self.bert_config)
            self.saver = tf.train.Saver()
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.session = tf.Session(graph=self.graph, config=config)
        self.load()

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt is not None and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            raise Exception("load model failure...")

    def predict_batch(self, input_text):
        train_sentences = self.load_samples(input_text)
        train_data = convert_samples.prepare_dataset(
            train_sentences, self.max_seq_len, True)
        train_manager = BatchManager(train_data, len(input_text))
        temp = []

        results = self.model.evaluate(self.session, train_manager,
                                      convert_samples.id_to_tag)
        for i, v in enumerate(results):
            a = bio_to_json(input_text[i], v[1:-1])
            temp.append(a)
        data_items = {}
        s = []
        for j in temp:
            if len(j['entities']) > 0:
                for k in range(len(j['entities'])):
                    value = j['entities'][k]['word']
                    data_items[j['entities'][k]['type']] = [value]
            s.append(data_items)
        return s

    def load_samples(self, datas):
        """
        Load sentences. A line must contain at least a word and its tag.
        Sentences are separated by empty lines.
        """
        sentences = []
        num = 0
        for j, data in enumerate(datas):
            sentence = []
            num += 1
            list_lable = list((len(data)) * 'O')
            for i, value in enumerate(data):
                temp = []
                temp.append(value)
                temp.append(list_lable[i])
                sentence.append(temp)
            sentences.append(sentence)
        return sentences


class_bert_lstm_graph = tf.Graph()
test = bert_predict(vocab_file='D:\\mygit\\NER_MODEL\\albert_tiny_489k\\vocab.txt'
                    , model_dir='D:\\mygit\\NER_MODEL\\models', graph=class_bert_lstm_graph,
                    init_checkpoint_file='D:\\mygit\\NER_MODEL\\albert_tiny_489k\\albert_model.ckpt'
                    , bert_config_dir='D:\\mygit\\NER_MODEL\\albert_tiny_489k\\albert_config_tiny.json')

if __name__ == '__main__':
    from datetime import datetime

    class_bert_lstm_graph = tf.Graph()
    test = bert_predict(vocab_file='D:\models\\albert_base_zh\\vocab.txt', map_file='maps.pkl'
                        , model_dir='..\\models', graph=class_bert_lstm_graph,
                        init_checkpoint_file='D:\迅雷下载\\albert_tiny_489k\\albert_model.ckpt'
                        , bert_config_dir='D:\迅雷下载\\albert_tiny_489k\\albert_config_tiny.json')
    data = ['晚上好，我同妈妈去看看卖手套的门店']
    date = datetime.now()
    a = test.predict_batch(data)
    # b = test.predict('晚上好，我同妈妈去看看卖手套的门店')
    date1 = datetime.now()
    print('=', a)
    # print(b)
    print(date1 - date)
    # 测试时间大概30ms左右
