# -*- encoding: utf-8 -*-
"""
@File    : data_process.py
@Time    : 2020/7/8 10:44
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import math
import random
from al_bert import tokenization


class BatchManager(object):
    def __init__(self, data, batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.arrange_batch(sorted_data[int(i * batch_size): int((i + 1) * batch_size)]))
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


class ConvertSamples():

    def __init__(self, **kwargs):
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=kwargs['vocab_file'],
            do_lower_case=True)
        self.tag_to_id = {'O': 0, 'I-BRD': 1, 'I-PRO': 2, 'B-PRO': 3, 'I-KWD': 4, 'B-BRD': 5, 'I-POP': 6, 'B-KWD': 7,
                          'B-POP': 8, 'I-PRC': 9, 'I-FLR': 10, 'B-FLR': 11, 'B-PRC': 12, '[CLS]': 13, '[SEP]': 14}
        self.id_to_tag = {0: 'O', 1: 'I-BRD', 2: 'I-PRO', 3: 'B-PRO', 4: 'I-KWD', 5: 'B-BRD', 6: 'I-POP', 7: 'B-KWD',
                          8: 'B-POP', 9: 'I-PRC', 10: 'I-FLR', 11: 'B-FLR', 12: 'B-PRC', 13: '[CLS]', 14: '[SEP]'}

    def prepare_dataset(self, sentences, max_seq_length, lower=False, train=False):
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

    def convert_single_example(self, char_line, max_seq_length, tokenizer,label_line):
        """
        将一个样本进行分析，然后将字转化为id, 标签转化为lb
        """
        text_list = char_line.split(' ')

        tokens = []
        labels = []
        for i, word in enumerate(text_list):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                labels.append("O")
        # 序列截断
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(self.tag_to_id["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(self.tag_to_id[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(self.tag_to_id["[SEP]"])
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


convert_samples = ConvertSamples(vocab_file='D:\models\\albert_base_zh\\vocab.txt')


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def iob2(tags):
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


def bio_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    iCount = 0
    entity_tag = ""

    for c_idx in range(len(tags)):
        c, tag = string[c_idx], tags[c_idx]
        if c_idx < len(tags)-1:
            tag_next = tags[c_idx+1]
        else:
            tag_next = ''

        if tag[0] == 'B':
            entity_tag = tag[2:]
            entity_name = c
            entity_start = iCount
            if tag_next[2:] != entity_tag:
                item["entities"].append({"word": c, "start": iCount, "end": iCount + 1, "type": tag[2:]})
        elif tag[0] == "I":
            if tag[2:] != tags[c_idx-1][2:] or tags[c_idx-1][2:] == 'O':
                tags[c_idx] = 'O'
                pass
            else:
                entity_name = entity_name + c
                if tag_next[2:] != entity_tag:
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": iCount + 1, "type": entity_tag})
                    entity_name = ''
        iCount += 1
    return item