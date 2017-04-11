from __future__ import print_function

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F

import numpy as np
from collections import defaultdict, Counter, namedtuple
from itertools import chain
import argparse, os, sys


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=914808182, type=int)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--sample_size', default=10, type=int)
    parser.add_argument('--embed_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--attention_size', default=256, type=int)
    parser.add_argument('--dropout', default=0., type=float)

    parser.add_argument('--src_vocab_size', default=20000, type=int)
    parser.add_argument('--tgt_vocab_size', default=20000, type=int)

    parser.add_argument('--train_src')
    parser.add_argument('--train_tgt')
    parser.add_argument('--dev_src')
    parser.add_argument('--dev_tgt')
    parser.add_argument('--test_src')
    parser.add_argument('--test_tgt')

    parser.add_argument('--decode_max_time_step', default=200, type=int)

    parser.add_argument('--valid_niter', default=500, type=int)
    parser.add_argument('--load_model', default=None, type=str)
    parser.add_argument('--save_to', default='model', type=str)
    parser.add_argument('--save_model_after', default=2)
    parser.add_argument('--save_to_file', default=None, type=str)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_niter', default=-1, type=int)

    args = parser.parse_args()
    # seed numpy
    np.random.seed(args.seed * 13 / 7)

    return args


def read_corpus(file_path):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def build_vocab(data, cutoff):
    vocab = defaultdict(lambda: 0)
    vocab['<unk>'] = 0
    vocab['<s>'] = 1
    vocab['</s>'] = 2
    vocab['<pad>'] = 3

    word_freq = Counter(chain(*data))
    non_singletons = [w for w in word_freq if word_freq[w] > 1 and w not in vocab]  # do not count <unk> in corpus
    print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq), len(non_singletons)))

    top_k_words = sorted(non_singletons, reverse=True, key=word_freq.get)[:cutoff - len(vocab)]
    for word in top_k_words:
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def build_id2word_vocab(vocab):
    return {v: k for k, v in vocab.iteritems()}


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in xrange(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        yield [data[i * batch_size + b][0] for b in range(cur_batch_size)], \
              [data[i * batch_size + b][1] for b in range(cur_batch_size)]


def data_iter(data, batch_size):
    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        np.random.shuffle(tuples)
        batched_data.extend(list(batch_slice(tuples, batch_size)))

    np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in xrange(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in xrange(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in xrange(batch_size)])

    return sents_t, masks


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def tensor_transform(linear, X):
    # X is a 3D tensor
    return linear(X.view(-1, X.size(2))).view(X.size(0), X.size(1), -1)


class NMT(nn.Module):
    def __init__(self, args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word, load_from=None, load_mode=None):
        super(NMT, self).__init__()

        self.args = args

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_vocab_id2word = src_vocab_id2word
        self.tgt_vocab_id2word = tgt_vocab_id2word

        self.src_embed = nn.Embedding(args.src_vocab_size, args.embed_size)
        self.tgt_embed = nn.Embedding(args.tgt_vocab_size, args.embed_size)

        self.encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size, bidirectional=True)
        self.decoder_lstm = nn.LSTMCell(args.embed_size + args.hidden_size * 2, args.hidden_size)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(args.embed_size, args.tgt_vocab_size)

        # transformation of decoder hidden states and context vectors before reading out target words
        self.dec_state_linear = nn.Linear(args.hidden_size + args.hidden_size * 2, args.embed_size)

        # transform encoding states to the first state in decoder
        self.dec_init_linear = nn.Linear(args.hidden_size * 2, args.hidden_size)

        self.att_src_linear = nn.Linear(args.hidden_size * 2, args.attention_size, bias=False)
        self.att_h_linear = nn.Linear(args.hidden_size, args.attention_size, bias=False)
        self.att_linear = nn.Linear(args.attention_size, 1)

    def forward(self, src_words, tgt_words):
        src_encodings, init_ctx_vec = self.encode(src_words)
        scores = self.decode(src_encodings, init_ctx_vec, tgt_words)

        return scores

    def encode(self, src_words, src_masks=None):
        """
        :param src_words: (src_sent_len, batch_size)
        :param src_masks: (src_sent_len)
        :return:
        """
        # (src_sent_len, batch_size, embed_size)
        src_word_embed = self.src_embed(src_words)

        # output: (src_sent_len, batch_size, hidden_size * 2)
        output, (last_state, last_cell) = self.encoder_lstm(src_word_embed)

        # (batch_size, hidden_size * 2)
        init_ctx_vec = torch.cat([last_cell[0], last_cell[1]], 1)
        init_ctx_vec = self.dec_init_linear(init_ctx_vec)

        return output, init_ctx_vec

    def decode(self, src_encoding, init_ctx_vec, tgt_words):
        """
        :param src_encoding: (src_sent_len, batch_size, hidden_size * 2)
        :param init_ctx_vec: (batch_size, hidden_size * 2)
        :param tgt_words: (tgt_sent_len, batch_size)
        :return:
        """
        init_cell = init_ctx_vec
        init_state = F.tanh(init_cell)
        batch_size = src_encoding.size(1)

        # pre-compute transformations for source sentences in calculating attention score
        # (src_sent_len, batch_size, attention_size)
        src_linear_for_att = tensor_transform(self.att_src_linear, src_encoding)

        hidden = (init_state, init_cell)
        ctx_tm1 = Variable(torch.zeros(batch_size, self.args.hidden_size * 2), requires_grad=False)
        if self.args.cuda:
            ctx_tm1 = ctx_tm1.cuda()

        tgt_word_embed = self.tgt_embed(tgt_words)
        scores = []

        # start from <S>, until y_{T-1}
        for y_tm1_embed in tgt_word_embed.split(split_size=1):
            stuff_to_concate = [y_tm1_embed.squeeze(0), ctx_tm1]
            print([v.size() for v in stuff_to_concate])
            x = torch.cat(stuff_to_concate, 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)

            ctx_t, alpha_t = self.attention(h_t, src_encoding, src_linear_for_att)

            read_out = F.tanh(self.dec_state_linear(torch.cat([h_t, ctx_t], 1)))
            score_t = self.readout(read_out)
            scores.append(score_t)

            ctx_tm1 = ctx_t
            hidden = h_t, cell_t

        scores = torch.stack(scores)
        return scores

    def attention(self, h_t, src_encoding, src_linear_for_att):
        # (1, batch_size, attention_size) + (src_sent_len, batch_size, attention_size) =>
        # (src_sent_len, batch_size, attention_size)
        att_hidden = F.tanh(self.att_h_linear(h_t).unsqueeze(0).expand_as(src_linear_for_att) + src_linear_for_att)

        # (batch_size, src_sent_len)
        att_weights = F.softmax(tensor_transform(self.att_linear, att_hidden).squeeze(2).permute(1, 0))

        # (batch_size, hidden_size * 2)
        ctx_vec = torch.bmm(src_encoding.permute(1, 2, 0), att_weights.unsqueeze(2)).squeeze(2)

        return ctx_vec, att_weights


def train(args):
    train_data_src = read_corpus(args.train_src)
    train_data_tgt = read_corpus(args.train_tgt)

    dev_data_src = read_corpus(args.dev_src)
    dev_data_tgt = read_corpus(args.dev_tgt)

    src_vocab = build_vocab(train_data_src, args.src_vocab_size)
    tgt_vocab = build_vocab(train_data_tgt, args.tgt_vocab_size)

    src_vocab_id2word = build_id2word_vocab(src_vocab)
    tgt_vocab_id2word = build_id2word_vocab(tgt_vocab)

    model = NMT(args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word)
    optimizer = torch.optim.Adam(model.parameters())

    vocab_mask = torch.ones(args.tgt_vocab_size)
    vocab_mask[tgt_vocab['<pad>']] = 0
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)

    if args.cuda:
        model = model.cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()

    train_data = zip(train_data_src, train_data_tgt)
    dev_data = zip(dev_data_src, dev_data_tgt)
    train_iter = patience = cum_loss = cum_examples = epoch = valid_num = best_model_iter = 0
    hist_valid_scores = []
    train_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1
            src_word_ids = word2id(src_sents, src_vocab)
            tgt_word_ids = word2id(tgt_sents, tgt_vocab)
            batch_size = len(src_sents)
            total_word_num = sum(len(s) for s in tgt_sents)

            if train_iter % args.valid_niter == 0:
                valid_num += 1
                print('epoch %d, iter %d, cum. loss %f, ' \
                      'cum. examples %d, time elapsed %f(s)' % (epoch, train_iter,
                                                                cum_loss / cum_examples,
                                                                cum_examples,
                                                                time.time() - train_time), file=sys.stderr)

                train_time = time.time()
                cum_loss = cum_examples = 0.

            model.zero_grad()

            src_words, src_masks = input_transpose(src_word_ids, src_vocab['<pad>'])
            tgt_words, tgt_masks = input_transpose(tgt_word_ids, tgt_vocab['<pad>'])

            src_words_var = Variable(torch.LongTensor(src_words), requires_grad=False)
            tgt_words_var = Variable(torch.LongTensor(tgt_words), requires_grad=False)

            if args.cuda:
                src_words_var = src_words_var.cuda()
                tgt_words_var = src_words_var.cuda()

            # list of (batch_size, tgt_vocab_size)
            scores = model(src_words_var, tgt_words_var[:-1])

            losses = []

            loss = cross_entropy_loss(scores.view(-1, scores.size(2)), tgt_words_var[1:].view(-1))

            # from <s> to y_{T-1}
            # for t, (score_t, tgt_y_t) in enumerate(zip(scores.split(32), tgt_words_var[1:].split(32))):
            #     score_t = score_t.view(-1, score_t.size(2))
            #     tgt_y_t = tgt_y_t.view(-1)
            #     loss_t = cross_entropy_loss(score_t, tgt_y_t)
            #
            #     losses.append(loss_t)

            # loss = torch.stack(losses).sum()
            ppl = np.exp(loss.data[0] / total_word_num)
            loss /= batch_size
            loss_val = loss.data[0]

            print('epoch %d, iter %d, loss=%f, ppl=%f' % (epoch, train_iter, loss_val, ppl))
            loss.backward()
            optimizer.step()

            cum_loss += loss_val * batch_size
            cum_examples += batch_size


if __name__ == '__main__':
    args = init_config()
    print(args, file=sys.stderr)

    if args.mode == 'train':
        train(args)