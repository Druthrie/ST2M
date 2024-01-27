import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy

from torch.utils.data._utils.collate import default_collate

# import spacy

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion-withpast generative model'''
# 数据调整为4的倍数，按句子AB中最长长度升序排序
class st2m_Text2Motion_withpast_DatasetV4(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = opt.max_motion_length
        self.max_length = 16
        self.pointer = 0
        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []

        for name in tqdm(id_list):
            try:
                motion = []
                motion_length = []

                for i in range(self.opt.mul_data_size):
                    motion_i = np.load(pjoin(opt.motion_dir, name + '_C%03d' % i + '.npy'))  # (frame,263)
                    length_i = len(motion_i)
                    gap = length_i % self.opt.unit_length

                    if i == 0:
                        motion_i = motion_i[gap:]
                    else:
                        motion_i = motion_i[:length_i-gap]

                    motion.append(motion_i)
                    motion_length.append(len(motion_i))

                text_data = []
                # print('pjoin(opt.text_dir, name + .txt)',pjoin(opt.text_dir, name + '.txt'))

                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        # print('line',line)
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        text_data.append(text_dict)
                # print('motion',motion)
                # print('motion_length',motion_length)
                # print('text_data',text_data)
                # print('debug')
                data_dict[name] = {'motion': motion,
                                   'length': motion_length,
                                   'text': text_data}

                new_name_list.append(name)
                length_list.append(max(motion_length[0], motion_length[1]))

            except:
                # Some motion may not exist in KIT dataset
                print(name)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
        np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)


    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length, side='right')
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length


    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        # return len(self.data_dict) - self.pointer
        return self.pointer


    def __getitem__(self, item):
        # idx = self.pointer + item
        idx = item
        id_name = self.name_list[idx]
        data = self.data_dict[self.name_list[idx]]
        motion_list, m_length_list, text_list = data['motion'], data['length'], data['text']
        word_embeddings = []
        pos_one_hots = []
        caption = []
        sent_len = []
        motion = []
        m_length = []
        for i in range(self.opt.mul_data_size):
            motion_i = motion_list[i]
            m_length_i = m_length_list[i]
            text_data_i = text_list[i]
            caption_i, tokens_i = text_data_i['caption'], text_data_i['tokens']
            caption_i_split = caption_i.split(' ')
            if len(caption_i_split) > self.opt.max_text_len + 2:
                caption_i = caption_i_split[0]
                for j in range(self.opt.max_text_len + 1):
                    caption_i = caption_i + ' ' + caption_i_split[j + 1]

            if len(tokens_i) < self.opt.max_text_len:
                # pad with "unk"
                tokens_i = ['sos/OTHER'] + tokens_i + ['eos/OTHER']
                sent_len_i = len(tokens_i)
                tokens_i = tokens_i + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len_i)
            else:
                # crop
                tokens_i = tokens_i[:self.opt.max_text_len]
                tokens_i = ['sos/OTHER'] + tokens_i + ['eos/OTHER']
                sent_len_i = len(tokens_i)
            pos_one_hots_i = []
            word_embeddings_i = []
            for token in tokens_i:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots_i.append(pos_oh[None, :])
                word_embeddings_i.append(word_emb[None, :])
            pos_one_hots_i = np.concatenate(pos_one_hots_i, axis=0)
            word_embeddings_i = np.concatenate(word_embeddings_i, axis=0)


            motion_i = (motion_i - self.mean) / self.std
            motion_i = np.concatenate((motion_i,
                                       np.zeros((self.max_length - m_length_i, motion_i.shape[1]))
                                       ), axis=0)

            word_embeddings.append(word_embeddings_i[None, :])
            pos_one_hots.append(pos_one_hots_i[None, :])
            caption.append(caption_i)
            sent_len.append(sent_len_i)
            motion.append(motion_i[None, :])
            m_length.append(m_length_i)

        word_embeddings = np.concatenate(word_embeddings, axis=0)
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        sent_len = np.array(sent_len)
        motion = np.concatenate(motion, axis=0)
        m_length = np.array(m_length)

        # print('word_embeddings.shape', word_embeddings.shape)
        # print('pos_one_hots.shape', pos_one_hots.shape)
        # print('len(caption)', len(caption))
        # print('sent_len.shape', sent_len.shape)
        # print('motion.shape', motion.shape)
        # print('m_length.shape', m_length.shape)



        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, id_name



'''For use of evaluations'''
class st2m_Text2Motion_withpast_Dataset_evalV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = opt.max_motion_length
        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = []
                motion_length = []

                for i in range(self.opt.mul_data_size):
                    motion_i = np.load(pjoin(opt.motion_dir, name + '_C%03d' % i + '.npy'))  # (frame,263)
                    length_i = len(motion_i)
                    gap = length_i % self.opt.unit_length

                    if i == 0:
                        motion_i = motion_i[gap:]
                    else:
                        motion_i = motion_i[:length_i - gap]

                    motion.append(motion_i)
                    motion_length.append(len(motion_i))

                text_data = []
                # print('pjoin(opt.text_dir, name + .txt)',pjoin(opt.text_dir, name + '.txt'))

                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        # print('line',line)
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        text_data.append(text_dict)
                # print('motion',motion)
                # print('motion_length',motion_length)
                # print('text_data',text_data)
                data_dict[name] = {'motion': motion,
                                   'length': motion_length,
                                   'text': text_data}

            except:
                # Some motion may not exist in KIT dataset
                print(name)
                pass

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = id_list

        # print('len(self.data_dict)', len(self.data_dict))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = item
        id_name = self.name_list[idx]
        data = self.data_dict[self.name_list[idx]]
        motion_list, m_length_list, text_list = data['motion'], data['length'], data['text']

        motion = np.concatenate(motion_list, axis=0)
        m_length = m_length_list[0] + m_length_list[1]
        caption = text_list[0]['caption'] + ' --> ' + text_list[1]['caption']
        tokens = text_list[0]['tokens'] + text_list[1]['tokens']

        if len(tokens) < self.opt.max_text_len * 2:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len * 2 + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len * 2]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        motion = (motion - self.mean) / self.std
        motion = np.concatenate((motion,
                                   np.zeros((self.max_motion_length * 2 - m_length, motion.shape[1]))
                                   ), axis=0)

        for i in range(self.opt.mul_data_size):
            motion_i = motion_list[i]
            m_length_i = m_length_list[i]
            text_data_i = text_list[i]
            caption_i, tokens_i = text_data_i['caption'], text_data_i['tokens']
            caption_i_split = caption_i.split(' ')
            if len(caption_i_split) > self.opt.max_text_len + 2:
                caption_i = caption_i_split[0]
                for j in range(self.opt.max_text_len + 1):
                    caption_i = caption_i + ' ' + caption_i_split[j + 1]

            if len(tokens_i) < self.opt.max_text_len:
                # pad with "unk"
                tokens_i = ['sos/OTHER'] + tokens_i + ['eos/OTHER']
                sent_len_i = len(tokens_i)
                tokens_i = tokens_i + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len_i)
            else:
                # crop
                tokens_i = tokens_i[:self.opt.max_text_len]
                tokens_i = ['sos/OTHER'] + tokens_i + ['eos/OTHER']
                sent_len_i = len(tokens_i)
            pos_one_hots_i = []
            word_embeddings_i = []
            for token in tokens_i:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots_i.append(pos_oh[None, :])
                word_embeddings_i.append(word_emb[None, :])
            pos_one_hots_i = np.concatenate(pos_one_hots_i, axis=0)
            word_embeddings_i = np.concatenate(word_embeddings_i, axis=0)


            motion_i = (motion_i - self.mean) / self.std
            motion_i = np.concatenate((motion_i,
                                       np.zeros((self.max_motion_length - m_length_i, motion_i.shape[1]))
                                       ), axis=0)

            if i == 0:
                word_embeddings_0 = word_embeddings_i
                pos_one_hots_0 = pos_one_hots_i
                caption_0 = caption_i
                sent_len_0 = sent_len_i
                motion_0 = motion_i
                m_length_0 = m_length_i
                tokens_0 = tokens_i
            elif i == 1:
                word_embeddings_1 = word_embeddings_i
                pos_one_hots_1 = pos_one_hots_i
                caption_1 = caption_i
                sent_len_1 = sent_len_i
                motion_1 = motion_i
                m_length_1 = m_length_i
                tokens_1 = tokens_i
        # print('word_embeddings_0: ',word_embeddings_0,word_embeddings_0.shape)
        # print('word_embeddings_1: ',word_embeddings_1,word_embeddings_1.shape)
        # print('word_embeddings: ',word_embeddings,word_embeddings.shape)
        # print('pos_one_hots_0: ',pos_one_hots_0,pos_one_hots_0.shape)
        # print('pos_one_hots_1: ',pos_one_hots_1,pos_one_hots_1.shape)
        # print('pos_one_hots: ',pos_one_hots,pos_one_hots.shape)
        # print('caption_0: ',caption_0)
        # print('caption_1: ',caption_1)
        # print('caption: ',caption)
        # print('sent_len_0:',sent_len_0)
        # print('sent_len_1:',sent_len_1)
        # print('sent_len:',sent_len)
        # print('motion_0:',motion_0,motion_0.shape)
        # print('motion_1:',motion_1,motion_1.shape)
        # print('motion:',motion,motion.shape)
        # print('m_length_0:',m_length_0)
        # print('m_length_1:',m_length_1)
        # print('m_length:',m_length)
        # print('tokens_0: ','_'.join(tokens_0))
        # print('tokens_1:','_'.join(tokens_1))
        # print('tokens:','_'.join(tokens))
        # print('debug')


        return word_embeddings_0, word_embeddings_1, pos_one_hots_0, pos_one_hots_1, caption_0, caption_1, \
               sent_len_0, sent_len_1, motion_0, motion_1, m_length_0, m_length_1, '_'.join(tokens_0), '_'.join(tokens_1), \
               word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), id_name

'''For use of training text motion matching model'''
class st2m_Text2Motion_withpast_Dataset_match(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = opt.max_motion_length
        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = []
                motion_length = 0

                for i in range(self.opt.mul_data_size):
                    # motion_i = np.load(pjoin(opt.motion_dir, name + '_C%03d' % i + '.npy'))  # (frame,263)
                    # motion.append(motion_i)
                    # motion_length += len(motion_i)

                    motion_i = np.load(pjoin(opt.motion_dir, name + '_C%03d' % i + '.npy'))  # (frame,263)
                    length_i = len(motion_i)
                    gap = length_i % self.opt.unit_length

                    if i == 0:
                        motion_i = motion_i[gap:]
                    else:
                        motion_i = motion_i[:length_i - gap]

                    motion.append(motion_i)
                    motion_length += len(motion_i)

                motion = np.concatenate(motion, axis=0)


                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    text_dict = {}
                    is_first = True
                    num = 0
                    for line in f.readlines():
                        num += 1
                        if num > self.opt.mul_data_size:
                            break
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')

                        caption_split = caption.split(' ')
                        if len(caption_split) > self.opt.max_text_len + 2:
                            caption = caption_split[0]
                            for idx in range(self.opt.max_text_len + 1):
                                caption = caption + ' ' + caption_split[idx + 1]

                        if len(tokens) > self.opt.max_text_len:
                            tokens = tokens[:self.opt.max_text_len]


                        if is_first:
                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            is_first = False
                        else:
                            text_dict['caption'] = text_dict['caption'] + ' --> ' + caption
                            # text_dict['tokens'] = text_dict['tokens'] + ['and/CCONJ', 'then/ADV'] + tokens
                            text_dict['tokens'] = text_dict['tokens'] + tokens

                data_dict[name] = {'motion': motion,
                                   'length': motion_length,
                                   'text': text_dict}

            except:
                # Some motion may not exist in KIT dataset
                print(name, '  data failed!!!')
                pass

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = id_list


    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = item
        id_name = self.name_list[idx]
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_dict = data['motion'], data['length'], data['text']

        caption, tokens = text_dict['caption'], text_dict['tokens']

        if len(tokens) < self.opt.max_text_len * self.opt.mul_data_size:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len * self.opt.mul_data_size + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len * self.opt.mul_data_size]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        motion = (motion - self.mean) / self.std
        motion = np.concatenate((motion, np.zeros((self.max_motion_length * self.opt.mul_data_size - m_length, motion.shape[1]))), axis=0)

        # print('word_embeddings: ',word_embeddings,word_embeddings.shape)
        # print('pos_one_hots: ',pos_one_hots,pos_one_hots.shape)
        # print('caption: ',caption)
        # print('sent_len: ',sent_len)
        # print('motion: ',motion,motion.shape)
        # print('m_length: ',m_length)
        # a = '_'.join(tokens)
        # print(a)
        # print('debug')

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)




class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                caption = line.split('#')[0].strip()
                dur = line.split('#')[1].strip()
                word_list, pos_list = self.process_text(caption)
                tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
                self.data_dict.append({'caption': caption, "tokens": tokens, "dur": int(dur)})

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))


    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens, dur = data['caption'], data['tokens'], data['dur']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, dur



