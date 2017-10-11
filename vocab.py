from __future__ import print_function
import argparse
from collections import Counter
from itertools import chain

import torch

from dataset import Dataset
from util import read_corpus


class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.iteritems()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        retained_words = [w for w in word_freq if word_freq[w] >= freq_cutoff]
        print('number of word types: %d, number of word types w/ frequency >= %d: %d' % (len(word_freq), freq_cutoff,
                                                                                         len(retained_words)))

        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)

        for word in top_k_words:
            if len(vocab_entry) < size:
                if word_freq[word] >= freq_cutoff:
                    vocab_entry.add(word)

        return vocab_entry


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_vocab_size', default=50000, type=int, help='target vocabulary size')
    parser.add_argument('--freq_cutoff', default=1, type=int)

    parser.add_argument('--data_folder', type=str, required=True, help='file of target sentences')
    parser.add_argument('--output', default='vocab.bin', type=str, help='output vocabulary file')

    args = parser.parse_args()

    dataset = Dataset(args.data_folder)
    all_captions = list(chain(*[caps for id, img, caps in dataset.train_examples()]))

    vocab = VocabEntry.from_corpus(all_captions, args.tgt_vocab_size, args.freq_cutoff)

    torch.save(vocab, args.output)
    print('vocabulary saved to %s' % args.output)