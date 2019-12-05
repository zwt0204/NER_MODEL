# -*- encoding: utf-8 -*-
"""
@File    : Latttice_Lstm.py
@Time    : 2019/12/4 19:09
@Author  : zwt
@git   : https://github.com/lyssym/NER-toolkits/tree/master/tf_kit/lattice
@Software: PyCharm
"""
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell, RNNCell
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
import numpy as np


'''
本欲想调整LatticeLSTM, 使其不依赖于固定的batch_size，但由于Lattice内部Cell的参数需要指定Shape，
而与batch_size对应的Shape在后续融合相关信息时，具有明显的实际意义，每个词汇对字符的贡献统计，
这也造成了本模型适用于处理批处理任务，对单个任务需进行调整后再处理较合适。

对于引入外部知识，以便对LSTM进行变形来说，应该是一种折中平衡吧。
'''


class CharLSTM(BasicLSTMCell):
    def __init__(self, lexicon_num_units, dtype, batch_size,
                 reuse=None, name=None, **kwargs):
        super(CharLSTM, self).__init__(reuse=reuse, name=name, **kwargs)
        self._lexicon_num_units = lexicon_num_units
        self._dtype = dtype
        self._char_state_tensor = tf.Variable(tf.zeros(shape=[batch_size, self._num_units]),
                                              dtype=self._dtype,
                                              trainable=False)

    def build(self, inputs_shape):
        # inputs_shape should be in the shape of [batch_size, char_embedding_size]
        if inputs_shape[-1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[-1].value
        h_depth = self._num_units
        lexicon_state_depth = self._lexicon_num_units

        self._kernel = self.add_variable(name='multi_input_kernel',
                                         shape=[input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(name='multi_input_bias',
                                       shape=[4 * self._num_units],
                                       initializer=tf.zeros_initializer(dtype=self._dtype))

        self._linking_kernel = self.add_variable(name='linking_kernel',
                                                 shape=[input_depth + lexicon_state_depth,
                                                        self._num_units])
        self._linking_bias = self.add_variable(name='linking_bias',
                                               shape=[self._num_units],
                                               initializer=tf.zeros_initializer(dtype=self._dtype))
        self.built = True

    def call(self, inputs, state):
        char_inputs = inputs[0]   # shape = [batch_size, input_dimension]
        state_inputs = inputs[1]  # shape = [batch_size, max_num_of_lexicon words, lexicon_state_dimension]

        # check whether the last dimension of state_inputs are all zero.
        # check_state_0 should be in the shape of [batch_size, max_num_of_lexicon words]
        check_state_0 = tf.reduce_sum(state_inputs, axis=-1)
        # check_state_1 should be in the shape of [batch_size]
        check_state_1 = tf.reduce_sum(check_state_0, axis=-1)

        # 查找匹配含有词汇的索引，只处理该部分信息，避免较多无词库匹配的信息参与计算消耗资源
        # state_inputs_indices_for_lexicon should be in the shape of [batch_size, 2]
        state_inputs_indices_for_lexicon = tf.where(tf.not_equal(check_state_0, 0))

        # 查找不含有词汇的索引，避免较多无词库匹配的信息参与计算消耗资源
        # tf.where(tf.equal(check_state_1, 0)) should be in the shape of [batch_size, 1]
        # state_inputs_indices_for_not_lexicon should be in the shape of [batch_size]
        state_inputs_indices_for_not_lexicon = tf.squeeze(tf.where(tf.equal(check_state_1, 0)))

        # 对不含词汇的细胞状态进行选择，主要是针对标量数据，因其秩为0，需进行维度扩展
        # in case `[i]` is squeezed to scalar `i`, change it back to 1-dimension tensor `[i]` by `tf.expand_dims()`
        # otherwise, `[]` and `[i, j]` will remain as-is after tf.squeeze() and further conversion on it
        state_inputs_indices_for_not_lexicon = tf.cond(pred=tf.equal(tf.rank(state_inputs_indices_for_not_lexicon), 0),
                                                       true_fn=lambda: tf.expand_dims(
                                                           state_inputs_indices_for_not_lexicon, axis=0),
                                                       false_fn=lambda: state_inputs_indices_for_not_lexicon)

        # 含有词汇匹配的字符索引
        # char_inputs_indices_for_lexicon should be in the shape of [batch_size, 1]
        char_inputs_indices_for_lexicon = tf.where(tf.not_equal(check_state_1, 0))

        # 不含有词汇匹配的字符索引
        # char_inputs_indices_for_not_lexicon should be in the shape of [batch_size, 1]
        char_inputs_indices_for_not_lexicon = tf.where(tf.equal(check_state_1, 0))

        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        # tf.concat([char_inputs, h], 1) should be in the shape of
        # [batch_size, char_embedding_size + state_dimension]
        # h should be in the shape of [batch_size, state_dimension]
        # self._kernel should be in the shape of [char_embedding_size + state_dimension, X]
        # gate_inputs should be in the shape of [batch_size, 4 * state_dimension]
        gate_inputs = tf.matmul(tf.concat([char_inputs, h], 1), self._kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)

        i, j, f, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)

        new_c_without_lexicon = self._new_c_without_lexicon(i=i, f=f, j=j, c=c,
                                                            indices_tensor=state_inputs_indices_for_not_lexicon)
        new_c = tf.scatter_nd_update(self._char_state_tensor,
                                     indices=char_inputs_indices_for_not_lexicon,
                                     updates=new_c_without_lexicon)

        new_c = tf.cond(tf.not_equal(tf.shape(state_inputs_indices_for_not_lexicon)[-1],
                                     tf.shape(state_inputs)[0]),
                        true_fn=lambda: self._if_not_empty_lexicon_state(i, j, char_inputs, state_inputs,
                                                                         char_inputs_indices_for_lexicon,
                                                                         state_inputs_indices_for_lexicon,
                                                                         new_c),
                        false_fn=lambda: new_c)

        # 计算输出隐状态
        new_h = tf.multiply(self._activation(new_c), tf.nn.sigmoid(o))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)

        return new_h, new_state

    def _new_c_without_lexicon(self, i, f, j, c, indices_tensor):
        # indices_tensor should be in the shape of [batch_size]
        f_without_lexicon_state_input = tf.gather(f, indices=indices_tensor)
        i_without_lexicon_state_input = tf.gather(i, indices=indices_tensor)
        j_without_lexicon_state_input = tf.gather(j, indices=indices_tensor)
        # j_without_lexicon_state_input should be in the shape of [batch_size]

        # 运行常规LSTM描述逻辑
        forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)
        new_c_without_lexicon_state = tf.add(
            tf.multiply(c, tf.nn.sigmoid(tf.add(f_without_lexicon_state_input,
                                                forget_bias_tensor))),
            tf.multiply(tf.nn.sigmoid(i_without_lexicon_state_input),
                        self._activation(j_without_lexicon_state_input)))

        return new_c_without_lexicon_state

    def _new_c_with_lexicon(self, i, j, char_inputs, state_inputs, indices_tensor):
        # char_inputs should be in the shape of [batch_size, char_embedding]
        # state_inputs should be in the shape of
        # [batch_size, max_num_of_lexicon words, lexicon_state_dimension]
        # indices_tensor is state_inputs_indices_for_lexicon, should be in the shape of [batch_size]
        char_inputs_with_lexicon_state = tf.gather_nd(char_inputs, indices=[indices_tensor])

        # 提取指定索引下的词汇状态信息
        # lexicon_state_inputs should be in the shape of [max_num_of_lexicon words, lexicon_state_dimension]
        lexicon_state_inputs = tf.gather_nd(state_inputs, indices=indices_tensor)

        i_with_lexicon_state_input = tf.gather_nd(i, indices=[indices_tensor])
        j_with_lexicon_state_input = tf.gather_nd(j, indices=[indices_tensor])

        # 常规输入门操作
        state_input_gate = tf.matmul(tf.concat([char_inputs_with_lexicon_state,
                                                lexicon_state_inputs], axis=-1),
                                     self._linking_kernel)
        state_input_gate = tf.nn.sigmoid(tf.nn.bias_add(state_input_gate, self._linking_bias))

        # 为了后面评估子词库对字符的贡献概率，引入了额外的门控单元 state_char_input_gate
        state_char_input_gate = tf.concat([state_input_gate,
                                           tf.nn.sigmoid(i_with_lexicon_state_input)], axis=1)

        # softmax 评估子词库对字符的贡献概率
        state_gate_weights, char_gate_weight = tf.split(
            tf.nn.softmax(state_char_input_gate, axis=0),
            num_or_size_splits=[tf.shape(lexicon_state_inputs)[0], 1],
            axis=1)

        # 常规LSTM操作
        new_c_with_lexicon_state = tf.add(
            tf.reduce_sum(tf.multiply(state_gate_weights, lexicon_state_inputs), axis=0),
            tf.multiply(char_gate_weight, j_with_lexicon_state_input))

        return new_c_with_lexicon_state

    def _if_not_empty_lexicon_state(self, i, j,
                                    char_inputs, state_inputs,
                                    char_inputs_indices_for_lexicon,
                                    state_inputs_indices_for_lexicon, new_c_in):
        new_c_with_lexicon = self._new_c_with_lexicon(i=i, j=j,
                                                      char_inputs=char_inputs,
                                                      state_inputs=state_inputs,
                                                      indices_tensor=state_inputs_indices_for_lexicon)
        # 根据新生成的含词库信息的字符状态更新字符的细胞状态
        new_c_out = tf.scatter_nd_update(new_c_in,
                                         indices=char_inputs_indices_for_lexicon,
                                         updates=new_c_with_lexicon)

        return new_c_out


class LexiconLSTM(BasicLSTMCell):
    def __init__(self, dtype, reuse=None, name=None, **kwargs):
        super(LexiconLSTM, self).__init__(reuse=reuse, name=name, **kwargs)
        self._dtype = dtype

    def build(self, inputs_shape):
        if inputs_shape[-1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[-1].value
        h_depth = self._num_units

        # 需要指出此处的超参数3与LatticeLSTM构成某种对应关系
        self._kernel = self.add_variable(name='lexicon_kernel',
                                         shape=[input_depth + h_depth, 3 * self._num_units])
        self._bias = self.add_variable(name='lexicon_bias',
                                       shape=[3 * self._num_units],
                                       initializer=tf.zeros_initializer(dtype=self._dtype))
        self.built = True

    def call(self, inputs, state):
        sigmoid = tf.nn.sigmoid
        add = tf.add
        multiply = tf.multiply

        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        gate_inputs = tf.matmul(tf.concat([inputs, h], 1), self._kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)

        i, j, f = tf.split(value=gate_inputs, num_or_size_splits=3, axis=1)

        forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)

        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        return new_c


class LatticeLSTMCell(RNNCell):
    ''' inherit from: tf.nn.rnn_cell.RNNCell
        Lattice Long short-term memory unit recurrent network cell. the implementation is based on
        https://arxiv.org/pdf/1805.02023.pdf
        Please be noted that LatticeLSTMCell should be called within tf.nn.dynamic_rnn
    '''
    def __init__(self, char_num_units, lexicon_num_units, batch_size, max_lexicon_words_num,
                 word_length_tensor, seq_len, dtype, **kwargs):
        super(LatticeLSTMCell, self).__init__(**kwargs)
        '''        
        Parameters
        ----------
        char_num_units: int
            the num_units of char_lstm cell units. 
            this is expected to be the same as lexicon_num_units by the paper.

        lexicon_num_units: int
            the num_units of lexicon_lstm cell units. 
            this is expected to be the same as char_num_units by the paper.

        max_lexicon_words_num: int
            the upper bound of the lexicon words per characters. 

        batch_size: int
            batch_size of the input data

        seq_len: int
            sequence_length of the input data

        dtype:
            data type defined for LatticeLSTMCell variable

        word_length_tensor: tensor
            tensor, it contains a batch of lexicon word length. 
            this should be padded with 0 if there is matched
            lexicon word for the respective character.
            example:  char： 南 京 市 長 江 大 橋
                      word： 南京 市長
                      max_lexicon_words_num = 5
            then the word length tensor should be like tf.constant([[[2,0,0,0,0],  
                                                                     [0,0,0,0,0],
                                                                     [2,0,0,0,0],
                                                                     [0,0,0,0,0],
                                                                     [0,0,0,0,0],
                                                                     [0,0,0,0,0],
                                                                     [0,0,0,0,0]]], 
                                                                   dtype=tf.float32)
        '''
        # 简要描述逻辑：先针对字符，以该字符结束匹配词库中词汇信息，最大词汇个数为 max_lexicon_words_num
        # 然后计算匹配的词汇对该字符的贡献，并融入在字符的细胞状态中；
        # 其中为了加快计算逻辑，对含有词汇的字符与不含有词汇的字符进行了分离，
        # 此外为了对LSTM进行变种计算，进行了数据的堆叠计算工作.
        self._char_lstm = CharLSTM(dtype=dtype,
                                   num_units=char_num_units,
                                   batch_size=batch_size,
                                   lexicon_num_units=lexicon_num_units,
                                   name='character_lstm')

        self._lexicon_lstm = LexiconLSTM(dtype=dtype,
                                         num_units=lexicon_num_units,
                                         name='lexicon_word_lstm')

        # word_length_tensor should be in the shape of [batched_size, seq_len, max_lexicon_words_num]
        self.word_length_tensor = word_length_tensor
        self.max_lexicon_words_num = max_lexicon_words_num
        self.seq_len = seq_len
        self.time_step = 0
        self._dtype = dtype

        lexicon_state_init_value = tf.zeros(shape=[batch_size, self.seq_len,
                                                   self.max_lexicon_words_num,
                                                   lexicon_num_units])

        # lexicon_state_tensor should be in the shape of
        # [batch_size, seq_len, max_lexicon_words_num, state_dimension]
        self.lexicon_state_tensor = tf.Variable(initial_value=lexicon_state_init_value,
                                                trainable=False,
                                                dtype=self._dtype)

    def build(self, inputs_shape):
        # inputs shape should be in the shape
        # [[batch_size, char_embedding_size],
        #  [batch_size, max_lexicon_words_num, lexicon_word_embedding_size]]

        self._char_lstm.build(inputs_shape[0])
        self._lexicon_lstm.build(inputs_shape[1])
        self.lexicon_shape = inputs_shape[1]

        if self.lexicon_shape[1] != self.max_lexicon_words_num:
            raise ValueError('max_lexicon_words_num should be equal to lexicon input')

        self.built = True

    @property
    def state_size(self):
        return self._char_lstm.state_size

    @property
    def output_size(self):
        return self._char_lstm.output_size

    def zero_state(self, batch_size, dtype):
        return self._char_lstm.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        '''
        Parameters
        ----------
        inputs: list of tensors
            inputs here should be a tensors of character_inputs(character embedding inputs) and
            lexicon_inputs(lexicon word embedding inputs). char_embedding_inputs has shape [batch_size, char_embedding_size]
            lexicon_embedding_inputs has shape [batch_size, max_lexicon_words_num, lexicon_word_embedding_size]
            please be noted that number of lexicon words is expect to be upper bounded. so if the number of words
            for a character is less than max_lexicon_words_num, the word embedding inputs should be padded with zero.
            example: inputs = [char_embedding_inputs, word_embedding_inputs]

        state: tensors
            Either a single 2-D tensor, or a tuple of tensors matching the arity and shapes of state.

        Returns
        -------
        output: tensor
            tensor of hidden output of character units with shape [batch_size, self.output_size]

        new_state: tensor or a tuple of tensors
            Either a single 2-D tensor, or a tuple of tensors matching the arity and shapes of state.
        '''
        char_input = inputs[0]      # shape = [batch_size, char_embedding_size]
        lexicon_inputs = inputs[1]  # shape = [batch_size, max_lexicon_words_num, lexicon_word_embedding_size]

        # 根据时间步获取对应词库状态, 其中 self.lexicon_state_tensor 会迭代更新
        # lexicon_state_tensor should be in the shape of [batch_size, max_lexicon_words_num, state_dimension]
        lexicon_state_tensor = tf.gather(self.lexicon_state_tensor, axis=1, indices=self.time_step)
        char_hidden_output, char_state = self._char_lstm.call([char_input, lexicon_state_tensor], state)

        # max_lexicon_words_num可视作局部字符的time_step
        # 经过融合处理后，作为字符融合了词库的过程
        for word_index in range(self.max_lexicon_words_num):
            self.lexicon_state_tensor = self._update_lexicon_state_per_word(lexicon_inputs=lexicon_inputs,
                                                                            word_index=word_index,
                                                                            char_state=char_state)

        self.time_step = self.time_step + 1  # time_step 向后推进

        # reset the lexicon_state_tensor after finish a loop, 一个循环后进行重置
        self.lexicon_state_tensor = tf.cond(tf.equal(tf.mod(self.time_step, self.seq_len - 1), 0),
                                            true_fn=lambda: tf.assign(ref=self.lexicon_state_tensor,
                                                                      value=tf.zeros_like(
                                                                          self.lexicon_state_tensor)),
                                            false_fn=lambda: self.lexicon_state_tensor)

        self.time_step = np.remainder(self.time_step, self.seq_len - 1)  # 求模以便重置

        return char_hidden_output, char_state

    def _update_lexicon_state_per_word(self, lexicon_inputs, word_index, char_state):
        # 根据词索引获取对应对应词库值
        # lexicon_inputs should be in the shape of
        # [batch_size, max_lexicon_words_num, lexicon_word_embedding_size]

        # lexicon_input_per_word should be in the shape of [batch_size, state_dimension]
        lexicon_input_per_word = tf.gather(lexicon_inputs, axis=1, indices=word_index)

        # 根据时间步获取对应词长度
        # self.word_length_tensor should be in the shape of [batch_size, seq_len, max_lexicon_words_num]
        # word_length_per_time_step should be in the shape of [batch_size, max_lexicon_words_num]
        word_length_per_time_step = tf.gather(self.word_length_tensor, axis=1, indices=self.time_step)

        # 根据词索引获取对应时间步 词长度中的对应值, 即一个整数描述词汇的长度
        # word_length should be in the shape of [batch_size]
        word_length = tf.gather(word_length_per_time_step, axis=1, indices=word_index)

        # 引入char_state 为了与该字符进行关联，并根据依据索引变化的 lexicon_input_per_word 词库调整词库状态
        # lexicon_state should be in the shape of [batch_size, state_dimension]
        lexicon_state = self._lexicon_lstm.call(lexicon_input_per_word, char_state)

        # temp_lexicon_state_to_char_index should be an integer, 词库中词汇长度字符进行映射，经分词后词汇长度会不一致
        temp_lexicon_state_to_char_index = self.time_step + word_length - 1

        # 通过判断词汇长度不为0，获取对应的状态索引值，并组织为 batch_size 个数据
        # not equal result should be an bool, assert the match state
        # lexicon_state_index should be in the shape of [batch_size, 1]
        lexicon_state_index = tf.where(tf.not_equal(temp_lexicon_state_to_char_index,
                                                    self.time_step - 1))

        # 根据词库匹配命中的索引值，提取相应的索引值，每个对应一个索引，并组织为 batch_size 个数据
        # lexicon_state_to_char_index should be in the shape of [batch_size]
        lexicon_state_to_char_index = tf.gather_nd(temp_lexicon_state_to_char_index,
                                                   indices=lexicon_state_index)

        # 根据词汇长度的索引，提取对应的词库状态值，只针对词库长度不为0进行处理
        # lexicon_state_update should be in the shape of [batch_size, state_dimension]
        lexicon_state_update = tf.gather_nd(lexicon_state, indices=lexicon_state_index)

        # 根据词汇匹配索引值，在词汇大小方向上扩展并设置为1
        # word_index_for_stack should be in the shape of [batch_size]
        word_index_for_stack = tf.ones_like(lexicon_state_to_char_index) * word_index

        # word_index_for_stack should be in the shape of [batch_size]
        # tf.squeeze(lexicon_state_index)将剔除列方向上的维度
        lexicon_state_index_for_stack = tf.cast(tf.squeeze(lexicon_state_index), dtype=self._dtype)

        # 堆叠的逻辑，实际描述同一个逻辑，可理解为递进描述
        # indices should be in the shape of [batch_size, 3]
        # 此处采用堆叠3个向量与LexiconLSTM 中kernel与bias中超参数3保持一致，
        # 并与tf.split中的超参数3保持一致，即后续向量能被整除
        indices = tf.stack([lexicon_state_index_for_stack,   # 长度不为0的词汇对应的索引
                            lexicon_state_to_char_index,     # 提取长度不为0的词汇对应的索引
                            word_index_for_stack], axis=-1)  # 词汇长度对应的索引扩展后设置为1

        # updated_lexicon_state_tensor should be in the shape of
        # [batch_size, seq_len, max_lexicon_words_num, state_dimension]
        updated_lexicon_state_tensor = tf.scatter_nd_update(ref=self.lexicon_state_tensor,
                                                            indices=tf.cast(indices, dtype=tf.int32),
                                                            updates=lexicon_state_update)
        return updated_lexicon_state_tensor
