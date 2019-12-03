# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2019/10/21 15:30
@Author  : zwt
@git   : 
@Software: PyCharm
"""
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from zwt.NER.albert_ner.al_bert import tokenization
from zwt.NER.albert_ner.al_bert import modeling
from time import time


# two texts pre-process
def text_process(text1, text2, tokenizer, max_seq_length):
    tokens_1 = tokenizer.tokenize(text1)
    tokens_2 = tokenizer.tokenize(text2)

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_1:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_2:
        for token in tokens_2:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return input_ids, input_mask, segment_ids


def text_process_batch(text1s, text2s, tokenizer, max_seq_length):
    result1 = []
    result2 = []
    result3 = []
    for text1 ,text2 in zip(text1s, text2s):
        tokens_1 = tokenizer.tokenize(text1)
        tokens_2 = tokenizer.tokenize(text2)
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_1:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_2:
            for token in tokens_2:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        result1.append(input_ids)
        result2.append(input_mask)
        result3.append(segment_ids)
    return result1, result2, result3


# restore finetuned model
def init(max_sequence_length, bert_config_file, model_path, vocab_file):
    sess = tf.Session()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    input_ids = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name='segment_ids')

    with sess.as_default():
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [2], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        # saver.restore(sess, model_path)

    return sess, tokenizer


def predict(sess, input_ids, input_mask, segment_ids):
    input_ids_tensor = sess.graph.get_tensor_by_name('input_ids:0')
    input_mask_tensor = sess.graph.get_tensor_by_name('input_mask:0')
    segment_ids_tensor = sess.graph.get_tensor_by_name('segment_ids:0')
    output_tensor = sess.graph.get_tensor_by_name('loss/BiasAdd:0')
    fd = {input_ids_tensor: [input_ids, input_ids], input_mask_tensor: [input_mask, input_mask],
          segment_ids_tensor: [segment_ids, segment_ids]}
    output_result = sess.run([output_tensor], feed_dict=fd)

    return output_result


def predict_batch(text1, text2, tokenizer, max_sequence_length):
    input_ids_tensor = sess.graph.get_tensor_by_name('input_ids:0')
    input_mask_tensor = sess.graph.get_tensor_by_name('input_mask:0')
    segment_ids_tensor = sess.graph.get_tensor_by_name('segment_ids:0')
    output_tensor = sess.graph.get_tensor_by_name('loss/Softmax:0')
    input_ids, input_mask, segment_ids = text_process_batch(text1, text2, tokenizer, max_sequence_length)

    feed_dict = {
        input_ids_tensor: input_ids, input_mask_tensor: input_mask,
        segment_ids_tensor: segment_ids}
    pred_ids_result = sess.run([output_tensor], feed_dict)
    print(pred_ids_result)


if __name__ == "__main__":
    vocab_file = 'D:\gitwork\script\data\\albert_base_zh\\vocab.txt'
    bert_config_file = 'D:\gitwork\script\data\\albert_base_zh\\albert_config_base.json'

    # the model path, you need to put your trained model in the path
    model_path = 'D:\gitwork\script\data\\albert_base_zh\model'
    max_sequence_length = 128
    # #########################################

    # the two strings you use to predict
    text1 = '我爱你'
    text2 = '我恨你'

    # init model
    t1 = time()
    sess, tokenizer = init(max_sequence_length, bert_config_file, model_path, vocab_file)
    print("init time: %.4f s" % (time() - t1))

    # predict
    t2 = time()
    input_ids, input_mask, segment_ids = text_process(text1, text2, tokenizer, max_sequence_length)
    result = predict(sess, input_ids, input_mask, segment_ids)
    print(result[0][0])
    print("predict time: %.4f s" % (time() - t2))

    # text1 = ['你是谁', '你的名字', '会员卡在那里办理','你妈妈是谁','停车费怎么算']
    # text2 = ['脑子有病', '我想吃饭', '哪里可以办会员','你是谁','停车怎么收费']
    text1 = ['会员卡在哪里办理', '会员卡在哪里办理', '会员卡在哪里办理', '会员卡在哪里办理', '会员卡在哪里办理', '我的信息有变更的话要怎么办', '我的信息有变更的话要怎么办', '我的信息有变更的话要怎么办', '我的信息有变更的话要怎么办', '我的信息有变更的话要怎么办', '这个会员卡片如何积分', '这个会员卡片如何积分', '这个会员卡片如何积分', '这个会员卡片如何积分', '这个会员卡片如何积分', '会员卡片有有效期吗', '会员卡片有有效期吗', '会员卡片有有效期吗', '会员卡片有有效期吗', '会员卡片有有效期吗', '积分有效期到什么时候', '积分有效期到什么时候', '积分有效期到什么时候', '积分有效期到什么时候', '积分有效期到什么时候', '积分可以做些什么', '积分可以做些什么', '积分可以做些什么', '积分可以做些什么', '积分可以做些什么', '会员卡如何晋级', '会员卡如何晋级', '会员卡如何晋级', '会员卡如何晋级', '会员卡如何晋级', '会员卡会降级吗', '会员卡会降级吗', '会员卡会降级吗', '会员卡会降级吗', '会员卡会降级吗', '哪些品牌不参与积分', '哪些品牌不参与积分', '哪些品牌不参与积分', '哪些品牌不参与积分', '哪些品牌不参与积分', '会员福利有哪些', '会员福利有哪些', '会员福利有哪些', '会员福利有哪些', '会员福利有哪些', '商场附近的地铁站', '商场附近的地铁站', '商场附近的地铁站', '商场附近的地铁站', '商场附近的地铁站', '母婴室在哪儿', '母婴室在哪儿', '母婴室在哪儿', '母婴室在哪儿', '母婴室在哪儿', '停车场在哪儿', '停车场在哪儿', '停车场在哪儿', '停车场在哪儿', '停车场在哪儿', '餐厅在几楼', '餐厅在几楼', '餐厅在几楼', '餐厅在几楼', '餐厅在几楼', '停车怎么收费', '停车怎么收费', '停车怎么收费', '停车怎么收费', '停车怎么收费', '停车有什么优惠', '停车有什么优惠', '停车有什么优惠', '停车有什么优惠', '停车有什么优惠', '喷泉表演时间', '喷泉表演时间', '喷泉表演时间', '喷泉表演时间', '喷泉表演时间', '哪里可以寄存行李', '哪里可以寄存行李', '哪里可以寄存行李', '哪里可以寄存行李', '哪里可以寄存行李', '顾客在客服台还可以获得那些顾客便利', '顾客在客服台还可以获得那些顾客便利', '顾客在客服台还可以获得那些顾客便利', '顾客在客服台还可以获得那些顾客便利', '顾客在客服台还可以获得那些顾客便利', '唱首歌', '唱首歌', '唱首歌', '唱首歌', '唱首歌']
    text2 = ['微信可以办理电子会员卡吗', '想办会员卡，必须去服务台么', '会员卡电子的哪里可以领', '只想要电子的会员卡，怎么办理', '会员卡的办理渠道', '我手机号换了哪里能改', '个人信息可以在微信上修改么', '我信息有变更，怎么同步给你们', '手机号码换了，会员卡要更新吗', '我之前资料填错了，想改下生日', '会员卡怎么积分', '是要到你们服务台才可以积分么', '我可以上传小票积分么', '积分方式有哪几种', '会员卡办好了，不知道消费了能有积分吗', '这个卡片有效期是几年', '你们的会员卡还有别的地方可以用么', '会员卡有效期是一年么', '会员卡会过期吗', '办了会员卡以后会失效吗', '积分有没有有效期呢', '积分截止日期怎么算', '我今天消费的积分是明年到期么', '积分不用的话会过期吗', '卡里有1000分什么时候失效', '积分有啥用', '积分太麻烦，真的能换东西吗', '存了几万分了，不知道要用来干嘛', '积分可以直接抵扣钱吗', '积分的用处', '银卡怎么升级成金卡', '我已经是金卡了，还有更高级别么', '会员卡怎么样可以进阶', '会员卡级别怎么升', '要达到多少钱才可以是黑卡', '会员卡要是不用会不会降级别', '我如果是黑卡的话还会降级么', '会员卡如果积分增速很慢的话会不会降级', '会员卡如果一直不消费级别会降吗', '金卡会降银卡吗', '你们是所有品牌都可以参加积分吗', '我在超市购物可以享受你们会员卡积分吗', '有没有什么商户是不给积分的', '你们是所有品牌都给积分么', '是不是有些消费类型没有积分', '有啥会员福利呢', '我生日的话会员积分可以翻倍么', '会员有什么福利', '会员的福利有什么', '会员生日月有礼物么', '爱琴海最近的地铁站是哪个', '你们附近是几号线什么站', '哪个地铁站离这里最近', '你们附近是9号线吗', '10号线下来哪个口子最近', '母婴室位置在哪里', '母婴室在几楼', '3楼有母婴室吗', '母婴室和卫生间是在一起吗', '我想找个给娃换尿布的地方', '你们停车场是在地下吗', '停车场有地下几楼', '新能源充电的车位是在哪里', '我是特斯拉，车位去哪里找', '荣威充电桩车位在地下几层', '吃饭的地方在几楼', '3楼有吃饭的地方吗', '餐厅分布', '哪些楼层有吃饭的地方', '每楼都有饭店吗', '停车的费用怎么收', '你们这里停车一个小时多少钱', '停车是一小时收费还是半小时收费', '停在你们这里的话一小时多少钱', '你们停车收费标准是什么', '停车有优惠吗', '停车可以用积分抵扣吗', '停车费有减免吗', '可以有积分换免费停车吗', '会员卡积分可以扣停车费吗', '喷泉表演啥时候开始', '喷泉表演晚上几点开始', '喷泉表演在哪', '喷泉每场是多久', '想看音乐喷泉', '我带了行李箱哪里可以寄放', '你们有寄存行李箱的服务吗', '哪里可以存箱子', '提着两个大箱子逛街不方便', '有行李寄放吗', '服务台有什么服务提供', '客服台提供什么服务', '客服台有提供免费租童车么', '客户服务中心那里有什么服务提供呢', '你们服务台有免费尿布吗', '给我唱首歌吧', '你会唱歌吗', '会放音乐吗', '你会唱什么歌', '会唱周杰伦的歌么']

    t3 = time()
    print(predict_batch(text1, text2, tokenizer, max_sequence_length))
    print("predict time: %.4f s" % (time() - t3))