import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader

def pad_mask(tensors, two_dim=True):
    lengths = [tensor.shape[0] for tensor in tensors]
    max_len = max(lengths)
    feature_size = tensors[0].shape[-1]


    if two_dim:
        padded = torch.zeros(len(tensors), max_len)
    else:
        padded = torch.zeros(len(tensors), max_len, feature_size)

    masked = torch.zeros(len(tensors), max_len)

    for i, length in enumerate(lengths):
        padded[i, :length] = tensors[i]
        masked[i, :length] = 1

    return padded, masked

def pad_collate(batch):
    srcs, trgs, srcs_oh = zip(*batch)
    sents, sent_masks = pad_mask(srcs)
    strokes, stroke_masks = pad_mask(trgs, two_dim=False)
    srcs_oh, _ = pad_mask(srcs_oh, two_dim=False)
    return sents.int(), strokes.float(), srcs_oh.int(), sent_masks.int(), stroke_masks.int()


class HandwritingDataSet(data.Dataset):
    def __init__(self, train=False, filter=800):
        path = '/Users/bharathc/Desktop/Recruiting_Fall_2019/descript-research-test/my_sol/data/'
        strokes = np.load(path + 'strokes-py3.npy', encoding='latin1', allow_pickle=True)
        sentences = open(path + 'sentences.txt').read().splitlines()
        self.strokes = []
        self.sentences = []
        for stroke, sentence in zip(strokes, sentences):
            if stroke.shape[0] <= filter:
                self.strokes.append(stroke)
                self.sentences.append(sentence)

        data_index = int(0.9 * len(self.strokes))
        if train:
            self.strokes = self.strokes[:data_index]
            self.sentences = self.sentences[:data_index]
        else:
            self.strokes = self.strokes[data_index:]
            self.sentences = self.sentences[data_index:]

        self.vocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,."\'?-'
        self.vocab_size = len(self.vocab)
        self.char2int = {c:i+1 for i, c in enumerate(self.vocab)}
        self.int2char = {v: k for k,v in self.char2int.items()}
        self.ohot_lookup = np.eye(self.vocab_size)
        self.set = []
        self.gen_data()

    def __getitem__(self, idx):
        return self.set[idx]

    def __len__(self):
        return len(self.set)

    def gen_data(self):
        for sentence, stroke in zip(self.sentences, self.strokes):
            src = [self.vocab.index(c) for c in sentence if c in self.vocab]
            src_oh = [self.ohot_lookup[self.vocab.index(c)] for c in sentence if c in self.vocab]
            self.set.append(
                (torch.IntTensor(src), torch.FloatTensor(stroke), torch.IntTensor(src_oh))
            )

if __name__ == '__main__':
    ut_data = HandwritingDataSet()
    ut_loader = DataLoader(ut_data, batch_size=16, collate_fn=pad_collate, drop_last=True)
    unit_test = True   # uncomment to check if data is good
    if unit_test:
        for i, batch in enumerate(ut_loader):
            sents, strokes, oh_sents, sent_masks, stroke_masks = batch
            assert sents.shape[0] == 16
            assert strokes.shape[0] == 16 and strokes.shape[-1] == 3
            assert oh_sents.shape[0] == 16 and oh_sents.shape[-1] == ut_loader.dataset.vocab_size
            assert sent_masks.shape[0] == 16
            assert stroke_masks.shape[0] == 16
        print('Sentences shape: {}'.format(sents.shape))
        print('Sentence masks shape: {}'.format(sent_masks.shape))
        print('Strokes shape: {}'.format(strokes.shape))
        print('One hot encoding of sentences shape: {}'.format(oh_sents.shape))
        print('Stroke masks shape: {}'.format(stroke_masks.shape))
