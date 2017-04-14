from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.nn.utils
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F

import numpy as np
from collections import defaultdict, Counter, namedtuple
from itertools import chain, islice
import argparse, os, sys


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5783287, type=int)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--mode', choices=['train', 'test', 'sample'], default='train')
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
    parser.add_argument('--clip_grad', default=5., type=float)
    parser.add_argument('--max_niter', default=-1, type=int)

    parser.add_argument('--sample_method', default='expand')

    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
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

        # attention
        self.att_src_linear = nn.Linear(args.hidden_size * 2, args.attention_size, bias=False)
        self.att_h_linear = nn.Linear(args.hidden_size, args.attention_size, bias=False)
        self.att_linear = nn.Linear(args.attention_size, 1)

        # readout dropout
        self.dropout = nn.Dropout(args.dropout)

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

        ctx_tm1 = Variable(init_cell.data.new(batch_size, self.args.hidden_size * 2).zero_(), requires_grad=False)

        tgt_word_embed = self.tgt_embed(tgt_words)
        scores = []

        # start from <S>, until y_{T-1}
        for y_tm1_embed in tgt_word_embed.split(split_size=1):
            x = torch.cat([y_tm1_embed.squeeze(0), ctx_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)

            ctx_t, alpha_t = self.attention(h_t, src_encoding, src_linear_for_att)

            read_out = F.tanh(self.dec_state_linear(torch.cat([h_t, ctx_t], 1)))
            read_out = self.dropout(read_out)

            score_t = self.readout(read_out)
            scores.append(score_t)

            ctx_tm1 = ctx_t
            hidden = h_t, cell_t

        scores = torch.stack(scores)
        return scores

    def translate(self, src_sent, beam_size=None, to_word=True):
        if not type(src_sent[0]) == list:
            src_sent = [src_sent]
        if not beam_size:
            beam_size = args.beam_size

        src_words, src_word_masks = input_transpose(src_sent, self.src_vocab['<pad>'])
        src_words = Variable(torch.LongTensor(src_words), volatile=True)
        if args.cuda:
            src_words = src_words.cuda()

        src_encoding, init_ctx_vec = self.encode(src_words)

        init_cell = init_ctx_vec
        init_state = F.tanh(init_cell)

        # pre-compute transformations for source sentences in calculating attention score
        # (src_sent_len, 1, attention_size)
        src_linear_for_att = tensor_transform(self.att_src_linear, src_encoding)

        ctx_tm1 = Variable(torch.zeros(beam_size, self.args.hidden_size * 2), volatile=True)
        hyp_scores = Variable(torch.zeros(beam_size), volatile=True)
        if args.cuda:
            ctx_tm1 = ctx_tm1.cuda()
            hyp_scores = hyp_scores.cuda()
        hidden = (init_state.expand(beam_size, init_state.size(1)), init_cell.expand(beam_size, init_cell.size(1)))

        eos_id = self.tgt_vocab['</s>']
        hypotheses = [[self.tgt_vocab['<s>']] for _ in xrange(beam_size)]
        completed_hypotheses = []
        completed_hypothesis_scores = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            t += 1
            hyp_num = len(hypotheses)

            expanded_src_encoding = src_encoding.expand(src_encoding.size(0), hyp_num, src_encoding.size(2))
            expanded_src_linear_for_att = src_linear_for_att.expand(src_linear_for_att.size(0), hyp_num, src_linear_for_att.size(2))

            y_tm1 = Variable(torch.LongTensor([hyp[-1] for hyp in hypotheses]), volatile=True)
            if args.cuda:
                y_tm1 = y_tm1.cuda()

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, ctx_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)

            ctx_t, alpha_t = self.attention(h_t, expanded_src_encoding, expanded_src_linear_for_att)

            read_out = F.tanh(self.dec_state_linear(torch.cat([h_t, ctx_t], 1)))
            read_out = self.dropout(read_out)

            score_t = self.readout(read_out)
            p_t = F.log_softmax(score_t)

            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=hyp_num)
            prev_hyp_ids = top_new_hyp_pos / args.tgt_vocab_size
            word_ids = top_new_hyp_pos % args.tgt_vocab_size
            new_hyp_scores = new_hyp_scores[top_new_hyp_pos.data]

            new_hypotheses = []

            live_hyp_ids = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data, word_ids.cpu().data, new_hyp_scores.cpu().data):
                hyp_tgt_words = hypotheses[prev_hyp_id] + [word_id]
                if word_id == eos_id:
                    completed_hypotheses.append(hyp_tgt_words)
                    completed_hypothesis_scores.append(new_hyp_score)
                else:
                    new_hypotheses.append(hyp_tgt_words)
                    live_hyp_ids.append(prev_hyp_id)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.LongTensor(live_hyp_ids)
            if args.cuda:
                live_hyp_ids = live_hyp_ids.cuda()

            hidden = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            ctx_tm1 = ctx_t[live_hyp_ids]
            hyp_scores = new_hyp_scores[live_hyp_ids]
            hypotheses = new_hypotheses

        if len(completed_hypotheses) == 0:
            completed_hypotheses = [hypotheses[0]]
            completed_hypothesis_scores = [0.0]

        if to_word:
            for i, hyp in enumerate(completed_hypotheses):
                completed_hypotheses[i] = [self.tgt_vocab_id2word[w] for w in hyp]

        ranked_hypotheses = sorted(zip(completed_hypotheses, completed_hypothesis_scores), key=lambda x: x[1], reverse=True)

        return [hyp for hyp, score in ranked_hypotheses]

    def sample(self, src_sent, sample_size=None, to_word=False):
        if not type(src_sent[0]) == list:
            src_sent = [src_sent]
        if not sample_size:
            sample_size = args.sample_size

        batch_size = len(src_sent) * sample_size

        src_words, src_word_masks = input_transpose(src_sent, self.src_vocab['<pad>'])
        src_words = Variable(torch.LongTensor(src_words), volatile=True)
        if args.cuda:
            src_words = src_words.cuda()

        src_encoding, init_ctx_vec = self.encode(src_words)

        # tile everything
        if args.sample_method == 'expand':
            # src_enc: (src_sent_len, sample_size, enc_size)
            # cat result: (src_sent_len, batch_size * sample_size, enc_size)
            src_encoding = torch.cat([src_enc.expand(src_enc.size(0), sample_size, src_enc.size(2)) for src_enc in src_encoding.split(1, dim=1)], 1)
            init_ctx_vec = torch.cat([init_ctx.expand(sample_size, init_ctx.size(1)) for init_ctx in init_ctx_vec.split(1, dim=0)], 0)
        elif args.sample_method == 'repeat':
            src_encoding = src_encoding.repeat(1, sample_size, 1)
            init_ctx_vec = init_ctx_vec.repeat(sample_size, 1)

        # src_encoding = src_encoding.expand(src_encoding.size(0), sample_size, src_encoding.size(2)).contiguous()
        # init_ctx_vec = init_ctx_vec.expand(sample_size, init_ctx_vec.size(1))

        init_cell = init_ctx_vec
        init_state = F.tanh(init_cell)

        # pre-compute transformations for source sentences in calculating attention score
        # (src_sent_len, batch_size, attention_size)
        src_linear_for_att = tensor_transform(self.att_src_linear, src_encoding)

        hidden = (init_state, init_cell)

        if not self.args.cuda:
            ctx_tm1 = Variable(torch.zeros(batch_size, self.args.hidden_size * 2), volatile=True)
        else:
            ctx_tm1 = Variable(torch.zeros(batch_size, self.args.hidden_size * 2), volatile=True).cuda()

        y_0 = Variable(torch.LongTensor([self.tgt_vocab['<s>'] for _ in xrange(batch_size)]), volatile=True)

        eos = self.tgt_vocab['</s>']
        eos_batch = torch.LongTensor([eos] * batch_size)
        if args.cuda:
            y_0 = y_0.cuda()
            eos_batch = eos_batch.cuda()

        samples = [y_0]

        t = 0
        while t < args.decode_max_time_step:
            t += 1

            # (sample_size)
            y_tm1 = samples[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, ctx_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)

            ctx_t, alpha_t = self.attention(h_t, src_encoding, src_linear_for_att)

            read_out = F.tanh(self.dec_state_linear(torch.cat([h_t, ctx_t], 1)))
            read_out = self.dropout(read_out)

            score_t = self.readout(read_out)
            p_t = F.softmax(score_t)

            y_t = torch.multinomial(p_t, num_samples=1).squeeze(1)
            samples.append(y_t)

            if torch.equal(y_t.data, eos_batch):
                break

            ctx_tm1 = ctx_t
            hidden = h_t, cell_t

        # post-processing
        completed_samples = [list() for _ in xrange(batch_size)]
        for y_t in samples:
            for i, sampled_word in enumerate(y_t.cpu().data):
                if len(completed_samples[i]) == 0 or completed_samples[i][-1] != eos:
                    completed_samples[i].append(sampled_word)

        if to_word:
            completed_samples = word2id(completed_samples, self.tgt_vocab_id2word)

        return completed_samples

    def attention(self, h_t, src_encoding, src_linear_for_att):
        # (1, batch_size, attention_size) + (src_sent_len, batch_size, attention_size) =>
        # (src_sent_len, batch_size, attention_size)
        att_hidden = F.tanh(self.att_h_linear(h_t).unsqueeze(0).expand_as(src_linear_for_att) + src_linear_for_att)

        # (batch_size, src_sent_len)
        att_weights = F.softmax(tensor_transform(self.att_linear, att_hidden).squeeze(2).permute(1, 0))

        # (batch_size, hidden_size * 2)
        ctx_vec = torch.bmm(src_encoding.permute(1, 2, 0), att_weights.unsqueeze(2)).squeeze(2)

        return ctx_vec, att_weights

    def save(self, path):
        print('save parameters to [%s]' % path, file=sys.stderr)
        torch.save(self.state_dict(), path)


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
    model.train()

    vocab_mask = torch.ones(args.tgt_vocab_size)
    vocab_mask[tgt_vocab['<pad>']] = 0
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)

    if args.cuda:
        model = model.cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()

    optimizer = torch.optim.Adam(model.parameters())

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

                print('begin validation ...', file=sys.stderr)
                model.eval()
                dev_hyps, dev_bleu = decode(model, dev_data)
                model.train()
                print('validation: iter %d, dev. bleu %f' % (train_iter, dev_bleu), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or dev_bleu > max(hist_valid_scores)
                hist_valid_scores.append(dev_bleu)

                if valid_num > args.save_model_after:
                    model_file = args.save_to + '.iter%d.bin' % train_iter
                    print('save model to [%s]' % model_file, file=sys.stderr)
                    model.save(model_file)

                if is_better:
                    patience = 0
                    best_model_iter = train_iter

                    if valid_num > args.save_model_after:
                        print('save currently the best model ..', file=sys.stderr)
                        os.system('ln -sf %s %s' % (model_file, args.save_to + '.bin'))
                else:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    if patience == args.patience:
                        print('early stop!', file=sys.stderr)
                        print('the best model is from iteration [%d]' % best_model_iter, file=sys.stderr)
                        exit(0)

            optimizer.zero_grad()

            src_words, src_masks = input_transpose(src_word_ids, src_vocab['<pad>'])
            tgt_words, tgt_masks = input_transpose(tgt_word_ids, tgt_vocab['<pad>'])

            src_words_var = Variable(torch.LongTensor(src_words), requires_grad=False)
            tgt_words_var = Variable(torch.LongTensor(tgt_words), requires_grad=False)

            if args.cuda:
                src_words_var = src_words_var.cuda()
                tgt_words_var = tgt_words_var.cuda()

            # (tgt_sent_len, batch_size, tgt_vocab_size)
            scores = model(src_words_var, tgt_words_var[:-1])
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
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            optimizer.step()

            cum_loss += loss_val * batch_size
            cum_examples += batch_size


def get_bleu(references, hypotheses):
    # compute BLEU
    bleu_score = corpus_bleu([[ref[1:-1]] for ref in references],
                             [hyp[1:-1] for hyp in hypotheses])

    return bleu_score


def decode(model, data):
    hypotheses = []
    begin_time = time.time()
    for src_sent, tgt_sent in data:
        src_sent_wids = word2id(src_sent, model.src_vocab)
        hyp = model.translate(src_sent_wids)[0]
        hypotheses.append(hyp)
        print('*' * 50)
        print('Source: ', ' '.join(src_sent))
        print('Target: ', ' '.join(tgt_sent))
        print('Hypothesis: ', ' '.join(hyp))

    elapsed = time.time() - begin_time
    bleu_score = get_bleu([tgt for src, tgt in data], hypotheses)

    print('decoded %d examples, took %d s' % (len(data), elapsed), file=sys.stderr)
    if args.save_to_file:
        print('save decoding results to %s' % args.save_to_file, file=sys.stderr)
        with open(args.save_to_file, 'w') as f:
            for hyp in hypotheses:
                f.write(' '.join(hyp[1:-1]) + '\n')

    return hypotheses, bleu_score


def test(args):
    train_data_src = read_corpus(args.train_src)
    train_data_tgt = read_corpus(args.train_tgt)

    src_vocab = build_vocab(train_data_src, args.src_vocab_size)
    tgt_vocab = build_vocab(train_data_tgt, args.tgt_vocab_size)

    src_vocab_id2word = build_id2word_vocab(src_vocab)
    tgt_vocab_id2word = build_id2word_vocab(tgt_vocab)

    test_data_src = read_corpus(args.test_src)
    test_data_tgt = read_corpus(args.test_tgt)

    model = NMT(args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word)
    model.eval()

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        model.load_state_dict(torch.load(args.load_model, map_location=lambda storage, loc: storage))
    if args.cuda:
        model = model.cuda()

    test_data = zip(test_data_src, test_data_tgt)

    hypotheses, bleu_score = decode(model, test_data)

    bleu_score = get_bleu([tgt for src, tgt in test_data], hypotheses)
    print('Corpus Level BLEU: %f' % bleu_score, file=sys.stderr)


def sample(args):
    train_data_src = read_corpus(args.train_src)
    train_data_tgt = read_corpus(args.train_tgt)

    src_vocab = build_vocab(train_data_src, args.src_vocab_size)
    tgt_vocab = build_vocab(train_data_tgt, args.tgt_vocab_size)

    src_vocab_id2word = build_id2word_vocab(src_vocab)
    tgt_vocab_id2word = build_id2word_vocab(tgt_vocab)

    model = NMT(args, src_vocab, tgt_vocab, src_vocab_id2word, tgt_vocab_id2word)
    model.eval()

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        model.load_state_dict(torch.load(args.load_model, map_location=lambda storage, loc: storage))
    if args.cuda:
        model = model.cuda()

    train_data = zip(train_data_src, train_data_tgt)
    print('begin sampling')

    check_every = 10
    train_iter = cum_samples = 0
    train_time = time.time()
    for src_sents, tgt_sents in islice(data_iter(train_data, batch_size=args.batch_size), 500):
        train_iter += 1
        src_word_ids = word2id(src_sents, src_vocab)
        samples = model.sample(src_word_ids, sample_size=args.sample_size, to_word=True)
        cum_samples += len(samples)
        print('%d samples' % len(samples))

        if train_iter % check_every == 0:
            elapsed = time.time() - train_time
            print('sampling speed: %d/s' % (cum_samples / elapsed))
            cum_samples = 0
            train_time = time.time()

        # print('*' * 80)
        # print('target:' + ' '.join(tgt_sents[0]))
        # print('samples:')
        # for i, sample in enumerate(samples, 1):
        #     print('[%d] %s' % (i, ' '.join(sample[1:-1])))
        # print('*' * 80)

if __name__ == '__main__':
    args = init_config()
    print(args, file=sys.stderr)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)
    elif args.mode == 'test':
        test(args)