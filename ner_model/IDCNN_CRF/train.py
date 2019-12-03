# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/12/2 16:10
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import itertools
from ner_model.IDCNN_CRF.model import IDCNN_MODEL
import pickle
from ner_model.IDCNN_CRF.loader import *
from ner_model.IDCNN_CRF.data_utils import *
from ner_model.IDCNN_CRF.utils import *
logger = get_logger(os.path.join('log', 'train.log'))


class Train_Model():

    def __init__(self):
        self.lr = 0.001
        self.char_dim = 100
        self.seg_dim = 20
        self.num_tags = 13
        self.num_chars = 4412
        self.pre_emb = True
        self.batch_size = 20
        self.steps_check = 100
        self.emb_file = 'D:\model\\xf-ner-idcnn-crf-master\config\\vec.txt'
        self.map_file = 'D:\model\\xf-ner-idcnn-crf-master\maps.pkl'
        self.summary_path = 'summary'
        self.ckpt_path = 'ckpt'
        self.model = IDCNN_MODEL(self.lr, self.char_dim, self.seg_dim, self.num_tags, self.num_chars)
        
    def data_process(self):
        train_sentences = load_sentences('D:\model\\xf-ner-idcnn-crf-master\data\example.train', True, False)
        dev_sentences = load_sentences('D:\model\\xf-ner-idcnn-crf-master\data\example.dev', True, False)
        test_sentences = load_sentences('D:\model\\xf-ner-idcnn-crf-master\data\example.test', True, False)

        # Use selected tagging scheme (IOB / IOBES)
        update_tag_scheme(train_sentences, "iobes")
        update_tag_scheme(test_sentences, "iobes")

        # create maps if not exist
        if not os.path.isfile(self.map_file):
            # create dictionary for word
            if self.pre_emb:
                dico_chars_train = char_mapping(train_sentences, True)[0]
                dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                    dico_chars_train.copy(),
                    self.emb_file,
                    list(itertools.chain.from_iterable(
                        [[w[0] for w in s] for s in test_sentences])))
            else:
                _c, char_to_id, id_to_char = char_mapping(train_sentences, True)

            # Create a dictionary and a mapping for tags
            _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
            with open(self.map_file, "wb") as f:
                pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
        else:
            with open(self.map_file, "rb") as f:
                char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

        # prepare data, get a collection of list containing index
        train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, True)
        dev_data = prepare_dataset(dev_sentences, char_to_id, tag_to_id, True)
        test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, True)
        print("%i / %i / %i sentences in train / dev / test." % (
            len(train_data), len(dev_data), len(test_data)))
        train_manager = BatchManager(train_data, self.batch_size)
        dev_manager = BatchManager(dev_data, 100)
        test_manager = BatchManager(test_data, 100)
        # make path for store log and model if not exist
        make_path('result', self.ckpt_path)
        return train_manager, dev_manager, test_manager, id_to_tag, tag_to_id

    def train(self, epochs):
        train_manager, dev_manager, test_manager, id_to_tag, tag_to_id = self.data_process()
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        steps_per_epoch = train_manager.len_batch_data
        init = tf.global_variables_initializer()
        with tf.Session(config=tf_config) as sess:
            sess.run(init)
            logger.info("start training")
            # 配置 tensorboard
            # 跟踪节点
            tf.summary.scalar('loss', self.model.loss)
            # summary节点合并
            merged_summary = tf.summary.merge_all()
            # 构造summary文件写入器
            # 接受一个log的目录作为保存文件的路径。log目录如果不存在，会被程序自动创建。
            # 通常训练集日志和验证集日志分开存放，分别构造各自的summary文件写入器即可
            # 添加一个参数sess.graph,绘制静态的graph（计算图），否则绘制动态数据
            writer = tf.summary.FileWriter(self.summary_path)
            loss = []
            for i in range(epochs):
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss, feed_dict = self.model.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if i % 5 == 0:  # 每5次将训练结果写入tensorboard scalar
                        summary = sess.run(merged_summary, feed_dict=feed_dict)  # 运行summary节点
                        # 向记录器添加，除了接受summary节点的运行输出值，还接受一个global_step参数来作为序列号
                        writer.add_summary(summary, i)
                    if step % self.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{}, "
                                    "NER loss:{:>9.6f}".format(
                            iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []
                best = evaluate(sess, self.model, "dev", dev_manager, id_to_tag, logger)
                if best:
                    save_model(sess, self.model, self.ckpt_path, logger)
                evaluate(sess, self.model, "test", test_manager, id_to_tag, logger)


def evaluate_line():
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open('maps.pkl', "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        Model = IDCNN_MODEL(0.001, 100, 20, 13, 4412)
        model = create_model(sess, Model, 'ckpt', load_word2vec, id_to_char, logger)
        while True:
                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)


if __name__ == '__main__':
    model = Train_Model()
    model.train(1000)