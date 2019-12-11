# 中文命名实体识别
主要涉及方法：CRF、BILSTM+CRF、IDCNN+CRF、BILSTM+CNN+CRF、Lattice-LSTM、Transformer+CRF、ALBERT+BILSTM+CRF
## CRF
- 参考：https://github.com/phychaos/transformer_crf
- data: https://github.com/phychaos/transformer_crf/tree/master/data
- 前期迭代的时候用CRF比较快，效果也还可以。可以快速的进行数据的迭代
## BILSTM+CRF
- 数据格式：{'text':'肯德基在哪里', 'label':'B-PRO I-PRO O O'}
- 目前主流的套路，深度学习+CRF，通过BILSTM抽取特征再结合CRF，不过BILSTM是序列模型，数据量上去之后训练速度会比较慢。
## IDCNN+CRF
- 参考：https://github.com/crownpku/Information-Extraction-Chinese
- data：https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF/data
- 相对于BILSTM来说引入cnn，更充分的利用GPU资源，可以加快训练速度，
## BILSTM+CNN+CRF
- 数据格式：{'text':'肯德基在哪里', 'label':'B-PRO I-PRO O O'}
- 在BILSTM的基础上加入CNN抽取特征，CNN对文本来说相当于抽取n-gram特征
## Lattice-LSTM
- 参考：https://github.com/lyssym/NER-toolkits/tree/master/tf_kit/lattice
- data：https://github.com/jiesutd/LatticeLSTM
- 对于文本来说，分词在不同场景下很难统一，所以通常对于抽取使用字级别，但是这样会损失词的信息。Lattice-LSTM引入外部词库来弥补
通过改造现有的LSTM网络来更好的融合词级别的信息。
## Transformer+CRF
- 参考：https://github.com/phychaos/transformer_crf
- data：https://github.com/phychaos/transformer_crf/tree/master/data
- 理论上相对于传统的LSTM，CNN的特征抽取能力都要强，[参考资料](http://note.youdao.com/noteshare?id=888534704767cf6c6130a7c589e2cbcf&sub=0AC60C08EC074FB58378F2DC2FF84C65)
## ALBERT+BILSTM+CRF
- 数据格式：{'text':'肯德基在哪里', 'label':'B-PRO I-PRO O O'}
- 通过ALBERT引入预训练信息来丰富语义信息

# 近期论文
- Hierarchically-Refined Label Attention Network for Sequence Labeling (EMNLP 2019) [paper](https://www.aclweb.org/anthology/D19-1422/)，[code](https://github.com/Nealcly/BiLSTM-LAN) - LAN
- TENER: Adapting Transformer Encoder for Named Entity Recognition (CoRR 2019) [paper](https://arxiv.org/abs/1911.04474)，[code](https://github.com/fastnlp/TENER)
