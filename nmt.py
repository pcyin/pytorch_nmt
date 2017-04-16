from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nltk.translate.bleu_score import corpus_bleu
import time
import numpy as np
from collections import defaultdict, Counter, namedtuple
from itertools import chain, islice
import argparse, os, sys

from util import read_corpus, data_iter, batch_slice
from vocab import Vocab, VocabEntry


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5783287, type=int)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--mode', choices=['train', 'test', 'sample'], default='train')
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--sample_size', default=10, type=int)
    parser.add_argument('--embed_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--dropout', default=0., type=float)

    parser.add_argument('--train_src', type=str)
    parser.add_argument('--train_tgt', type=str)
    parser.add_argument('--dev_src', type=str)
    parser.add_argument('--dev_tgt', type=str)
    parser.add_argument('--test_src', type=str)
    parser.add_argument('--test_tgt', type=str)

    parser.add_argument('--decode_max_time_step', default=200, type=int)

    parser.add_argument('--valid_niter', default=500, type=int)
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--load_model', default=None, type=str)
    parser.add_argument('--save_to', default='model', type=str)
    parser.add_argument('--save_model_after', default=2)
    parser.add_argument('--save_to_file', default=None, type=str)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--uniform_init', default=None, type=float)
    parser.add_argument('--clip_grad', default=5., type=float)
    parser.add_argument('--max_niter', default=-1, type=int)
    parser.add_argument('--lr_decay', default=None, type=float)

    parser.add_argument('--sample_method', default='expand')

    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed * 13 / 7)

    return args


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
    def __init__(self, args, vocab):
        super(NMT, self).__init__()

        self.args = args

        self.vocab = vocab

        self.src_embed = nn.Embedding(len(vocab.src), args.embed_size, padding_idx=vocab.src['<pad>'])
        self.tgt_embed = nn.Embedding(len(vocab.tgt), args.embed_size, padding_idx=vocab.tgt['<pad>'])

        self.encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size / 2, bidirectional=True, dropout=args.dropout)
        self.decoder_lstm = nn.LSTMCell(args.embed_size + args.hidden_size, args.hidden_size)

        # transform encoding states to the first state in decoder
        self.dec_init_linear = nn.Linear(args.hidden_size, args.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        self.att_src_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(args.hidden_size, len(vocab.tgt))

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        # zero all biases
        self.dec_init_linear.bias.data.zero_()
        self.readout.bias.data.zero_()

    def forward(self, src_sents, src_sents_len, tgt_words):
        src_encodings, init_ctx_vec = self.encode(src_sents, src_sents_len)
        scores = self.decode(src_encodings, init_ctx_vec, tgt_words)

        return scores

    def encode(self, src_sents, src_sents_len):
        """
        :param src_sents: (src_sent_len, batch_size), sorted by the length of the source
        :param src_sents_len: (src_sent_len)
        """
        # (src_sent_len, batch_size, embed_size)
        src_word_embed = self.src_embed(src_sents)
        packed_src_embed = pack_padded_sequence(src_word_embed, src_sents_len)

        # output: (src_sent_len, batch_size, hidden_size)
        output, (last_state, last_cell) = self.encoder_lstm(packed_src_embed)
        output, _ = pad_packed_sequence(output)

        # (batch_size, hidden_size * 2)
        # decoder initialization vector
        dec_init_vec = torch.cat([last_cell[0], last_cell[1]], 1)
        dec_init_vec = self.dec_init_linear(dec_init_vec)

        return output, dec_init_vec

    def decode(self, src_encoding, dec_init_vec, tgt_sents):
        """
        :param src_encoding: (src_sent_len, batch_size, hidden_size)
        :param dec_init_vec: (batch_size, hidden_size)
        :param tgt_sents: (tgt_sent_len, batch_size)
        :return:
        """
        init_cell = dec_init_vec
        init_state = F.tanh(init_cell)
        hidden = (init_state, init_cell)

        new_tensor = init_cell.data.new
        batch_size = src_encoding.size(1)

        # (batch_size, src_sent_len, hidden_size)
        src_encoding = src_encoding.t()
        # initialize attentional vector
        att_tm1 = Variable(new_tensor(batch_size, self.args.hidden_size).zero_(), requires_grad=False)

        tgt_word_embed = self.tgt_embed(tgt_sents)
        scores = []

        # start from `<s>`, until y_{T-1}
        for y_tm1_embed in tgt_word_embed.split(split_size=1):
            # input feeding: concate y_tm1 and previous attentional vector
            x = torch.cat([y_tm1_embed.squeeze(0), att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding)

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))   # E.q. (5)
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)   # E.q. (6)
            scores.append(score_t)

            att_tm1 = att_t
            hidden = h_t, cell_t

        scores = torch.stack(scores)
        return scores

    def translate(self, src_sents, beam_size=None, to_word=True):
        """
        perform beam search
        TODO: batched beam search
        """
        if not type(src_sents[0]) == list:
            src_sents = [src_sents]
        if not beam_size:
            beam_size = args.beam_size

        src_sents_var = to_input_variable(src_sents, self.vocab.src, cuda=args.cuda, is_test=True)

        src_encoding, init_ctx_vec = self.encode(src_sents_var, [len(src_sents[0])])

        init_cell = init_ctx_vec
        init_state = F.tanh(init_cell)
        hidden = (init_state.expand(beam_size, init_state.size(1)), init_cell.expand(beam_size, init_cell.size(1)))

        att_tm1 = Variable(torch.zeros(beam_size, self.args.hidden_size), volatile=True)
        hyp_scores = Variable(torch.zeros(beam_size), volatile=True)
        if args.cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()

        eos_id = self.vocab.tgt['</s>']
        bos_id = self.vocab.tgt['<s>']
        tgt_vocab_size = len(self.vocab.tgt)

        hypotheses = [[bos_id] for _ in xrange(beam_size)]
        completed_hypotheses = []
        completed_hypothesis_scores = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            t += 1
            hyp_num = len(hypotheses)

            expanded_src_encoding = src_encoding.expand(src_encoding.size(0), hyp_num, src_encoding.size(2))

            y_tm1 = Variable(torch.LongTensor([hyp[-1] for hyp in hypotheses]), volatile=True)
            if args.cuda:
                y_tm1 = y_tm1.cuda()

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, expanded_src_encoding.t())

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)
            p_t = F.log_softmax(score_t)

            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=hyp_num)
            prev_hyp_ids = top_new_hyp_pos / tgt_vocab_size
            word_ids = top_new_hyp_pos % tgt_vocab_size
            # new_hyp_scores = new_hyp_scores[top_new_hyp_pos.data]

            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data, word_ids.cpu().data, top_new_hyp_scores.cpu().data):
                hyp_tgt_words = hypotheses[prev_hyp_id] + [word_id]
                if word_id == eos_id:
                    completed_hypotheses.append(hyp_tgt_words)
                    completed_hypothesis_scores.append(new_hyp_score)
                else:
                    new_hypotheses.append(hyp_tgt_words)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.LongTensor(live_hyp_ids)
            if args.cuda:
                live_hyp_ids = live_hyp_ids.cuda()

            hidden = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hyp_scores = Variable(torch.FloatTensor(new_hyp_scores), volatile=True) # new_hyp_scores[live_hyp_ids]
            if args.cuda:
                hyp_scores = hyp_scores.cuda()
            hypotheses = new_hypotheses

        if len(completed_hypotheses) == 0:
            completed_hypotheses = [hypotheses[0]]
            completed_hypothesis_scores = [0.0]

        if to_word:
            for i, hyp in enumerate(completed_hypotheses):
                completed_hypotheses[i] = [self.vocab.tgt.id2word[w] for w in hyp]

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
            ctx_tm1 = Variable(torch.zeros(batch_size, self.args.hidden_size), volatile=True)
        else:
            ctx_tm1 = Variable(torch.zeros(batch_size, self.args.hidden_size), volatile=True).cuda()

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
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding, src_linear_for_att)

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
        att_weights = F.softmax(tensor_transform(self.att_vec_linear, att_hidden).squeeze(2).permute(1, 0))

        # (batch_size, hidden_size * 2)
        ctx_vec = torch.bmm(src_encoding.permute(1, 2, 0), att_weights.unsqueeze(2)).squeeze(2)

        return ctx_vec, att_weights

    def dot_prod_attention(self, h_t, src_encoding, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # (batch_size, hidden_size, 1)
        h_t_linear = self.att_src_linear(h_t).unsqueeze(2)

        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding, h_t_linear).squeeze(2)
        if mask:
            att_weight.data.masked_fill_(mask, -float('inf'))
        att_weight = F.softmax(att_weight)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, att_weight

    def save(self, path):
        print('save parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


def to_input_variable(sents, vocab, cuda=False, is_test=False):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """

    word_ids = word2id(sents, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var


def evaluate_loss(model, data, crit):
    model.eval()
    cum_loss = 0.
    cum_tgt_words = 0.
    for src_sents, tgt_sents in data_iter(data, batch_size=args.batch_size, shuffle=False):
        pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`
        src_sents_len = [len(s) for s in src_sents]

        src_sents_var = to_input_variable(src_sents, model.vocab.src, cuda=args.cuda, is_test=True)
        tgt_sents_var = to_input_variable(tgt_sents, model.vocab.tgt, cuda=args.cuda, is_test=True)

        # (tgt_sent_len, batch_size, tgt_vocab_size)
        scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
        loss = crit(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))

        cum_loss += loss.data[0]
        cum_tgt_words += pred_tgt_word_num

    loss = cum_loss / cum_tgt_words
    return loss


def train(args):
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')

    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')

    train_data = zip(train_data_src, train_data_tgt)
    dev_data = zip(dev_data_src, dev_data_tgt)

    vocab = torch.load(args.vocab)

    model = NMT(args, vocab)
    model.train()

    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-args.uniform_init, args.uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)

    if args.cuda:
        model = model.cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_iter = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1

            src_sents_var = to_input_variable(src_sents, vocab.src, cuda=args.cuda)
            tgt_sents_var = to_input_variable(tgt_sents, vocab.tgt, cuda=args.cuda)

            batch_size = len(src_sents)
            src_sents_len = [len(s) for s in src_sents]
            pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`

            optimizer.zero_grad()

            # (tgt_sent_len, batch_size, tgt_vocab_size)
            scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])

            word_loss = cross_entropy_loss(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))
            loss = word_loss / batch_size
            word_loss_val = word_loss.data[0]
            loss_val = loss.data[0]

            loss.backward()
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            optimizer.step()

            report_loss += word_loss_val
            cum_loss += word_loss_val
            report_tgt_words += pred_tgt_word_num
            cum_tgt_words += pred_tgt_word_num
            report_examples += batch_size
            cum_examples += batch_size
            cum_batches += batch_size

            if train_iter % args.log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         np.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % args.valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_batches,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_batches = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)
                model.eval()

                # compute dev. ppl and bleu
                dev_loss = evaluate_loss(model, dev_data, cross_entropy_loss)
                dev_ppl = np.exp(dev_loss)
                dev_hyps, dev_bleu = decode(model, dev_data)

                model.train()
                print('validation: iter %d, dev. ppl %f, dev. bleu %f' % (train_iter, dev_ppl, dev_bleu), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or dev_bleu > max(hist_valid_scores)
                is_better_than_last = len(hist_valid_scores) == 0 or dev_bleu > hist_valid_scores[-1]
                hist_valid_scores.append(dev_bleu)

                if valid_num > args.save_model_after:
                    model_file = args.save_to + '.iter%d.bin' % train_iter
                    print('save model to [%s]' % model_file, file=sys.stderr)
                    model.save(model_file)

                if (not is_better_than_last) and args.lr_decay:
                    lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                    print('decay learning rate to %f' % lr, file=sys.stderr)
                    optimizer.param_groups[0]['lr'] = lr

                if is_better:
                    patience = 0
                    best_model_iter = train_iter

                    if valid_num > args.save_model_after:
                        print('save currently the best model ..', file=sys.stderr)
                        model_file_abs_path = os.path.abspath(model_file)
                        symlin_file_abs_path = os.path.abspath(args.save_to + '.bin')
                        os.system('ln -sf %s %s' % (model_file_abs_path, symlin_file_abs_path))
                else:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    if patience == args.patience:
                        print('early stop!', file=sys.stderr)
                        print('the best model is from iteration [%d]' % best_model_iter, file=sys.stderr)
                        exit(0)


def get_bleu(references, hypotheses):
    # compute BLEU
    bleu_score = corpus_bleu([[ref[1:-1]] for ref in references],
                             [hyp[1:-1] for hyp in hypotheses])

    return bleu_score


def decode(model, data):
    hypotheses = []
    begin_time = time.time()
    for src_sent, tgt_sent in data:
        hyp = model.translate(src_sent)[0]
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
    test_data_src = read_corpus(args.test_src, source='src')
    test_data_tgt = read_corpus(args.test_tgt, source='tgt')
    test_data = zip(test_data_src, test_data_tgt)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        args = params['args']
        state_dict = params['state_dict']

        model = NMT(args, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = model.cuda()

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