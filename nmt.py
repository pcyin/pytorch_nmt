from __future__ import print_function

import re

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import time
import numpy as np
from collections import defaultdict, Counter, namedtuple
from itertools import chain, islice
import argparse, os, sys

from dataset import Dataset
from util import read_corpus, data_iter, batch_slice, hamming_distance
from vocab import VocabEntry
from process_samples import generate_hamming_distance_payoff_distribution
import math


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')
    parser.add_argument('--mode', choices=['train', 'raml_train', 'test', 'sample', 'prob', 'interactive'],
                        default='train', help='run mode')
    parser.add_argument('--vocab', type=str, help='path of the serialized vocabulary')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--sample_size', default=10, type=int, help='sample size')
    parser.add_argument('--raw_image_size', default=4096)
    parser.add_argument('--embed_size', default=256, type=int, help='size of word embeddings')
    parser.add_argument('--hidden_size', default=256, type=int, help='size of LSTM hidden states')
    parser.add_argument('--dropout', default=0., type=float, help='dropout rate')

    parser.add_argument('--data_folder', type=str, help='path to the data folder')

    parser.add_argument('--decode_max_time_step', default=50, type=int, help='maximum number of time steps used '
                                                                              'in decoding and sampling')

    parser.add_argument('--valid_niter', default=500, type=int, help='every n iterations to perform validation')
    parser.add_argument('--valid_metric', default='bleu', choices=['bleu', 'ppl', 'word_acc', 'sent_acc'], help='metric used for validation')
    parser.add_argument('--log_every', default=50, type=int, help='every n iterations to log training statistics')
    parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    parser.add_argument('--save_model_after', default=2, help='save the model only after n validation iterations')
    parser.add_argument('--save_to_file', default=None, type=str, help='if provided, save decoding results to file')
    parser.add_argument('--save_nbest', default=False, action='store_true', help='save nbest decoding results')
    parser.add_argument('--patience', default=5, type=int, help='training patience')
    parser.add_argument('--uniform_init', default=None, type=float, help='if specified, use uniform initialization for all parameters')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    parser.add_argument('--max_niter', default=-1, type=int, help='maximum number of training iterations')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='decay learning rate if the validation performance drops')

    # raml training
    parser.add_argument('--temp', default=0.85, type=float, help='temperature in reward distribution')
    parser.add_argument('--raml_sample_mode', default='pre_sample',
                        choices=['pre_sample', 'hamming_distance', 'hamming_distance_impt_sample'],
                        help='sample mode when using RAML')
    parser.add_argument('--raml_sample_file', type=str, help='path to the sampled targets')
    parser.add_argument('--raml_bias_groundtruth', action='store_true', default=False, help='make sure ground truth y* is in samples')

    parser.add_argument('--smooth_bleu', action='store_true', default=False,
                        help='smooth sentence level BLEU score.')

    #TODO: greedy sampling is still buggy!
    parser.add_argument('--sample_method', default='random', choices=['random', 'greedy'])

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


class NMT(nn.Module):
    def __init__(self, args, vocab):
        super(NMT, self).__init__()

        self.args = args

        self.vocab = vocab

        self.tgt_embed = nn.Embedding(len(vocab), args.embed_size, padding_idx=vocab['<pad>'])

        self.image_encoding_linear = nn.Linear(args.raw_image_size, args.hidden_size, bias=True)
        self.decoder_lstm = nn.LSTMCell(args.embed_size + args.hidden_size, args.hidden_size)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(args.hidden_size, len(vocab), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size * 2, args.hidden_size)

    def forward(self, src_images, tgt_sents):
        src_encodings = self.encode(src_images)
        scores = self.decode(src_encodings, tgt_sents)

        return scores

    def encode(self, src_images):
        """
        :param src_images: (batch_size, image_encoding_size), sorted by the length of the source
        """
        src_images_encoding = self.image_encoding_linear(src_images)

        return src_images_encoding

    def decode(self, src_images_encoding, tgt_sents):
        """
        :param src_images_encoding: (batch_size, image_encoding_size)
        :param tgt_sents: (tgt_sent_len, batch_size)
        :return:
        """
        init_state = F.tanh(src_images_encoding)
        init_cell = src_images_encoding
        hidden = (init_state, init_cell)

        new_tensor = init_cell.data.new
        batch_size = src_images_encoding.size(0)

        # initialize attentional vector
        att_tm1 = Variable(new_tensor(batch_size, self.args.hidden_size).zero_(), requires_grad=False)

        # (tgt_sent_len, batch_size, embed_size)
        tgt_word_embed = self.tgt_embed(tgt_sents)
        scores = []

        # start from `<s>`, until y_{T-1}
        for y_tm1_embed in tgt_word_embed.split(split_size=1):
            # input feeding: concate y_tm1 and previous attentional vector
            x = torch.cat([y_tm1_embed.squeeze(0), att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            # no attention!
            ctx_t = src_images_encoding

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))   # E.q. (5)
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)   # E.q. (6)
            scores.append(score_t)

            att_tm1 = att_t
            hidden = h_t, cell_t

        scores = torch.stack(scores)
        return scores

    def translate(self, src_images, beam_size=None, to_word=True):
        """
        perform beam search
        TODO: batched beam search
        """
        if len(src_images.size()) == 1:
            src_images = src_images.unsqueeze(0)
        if not beam_size:
            beam_size = args.beam_size

        src_images_var = image_to_input_variable(src_images, cuda=args.cuda, is_test=True)

        src_images_encoding = self.encode(src_images_var)

        init_state = F.tanh(src_images_encoding)
        init_cell = src_images_encoding
        hidden = (init_state, init_cell)

        att_tm1 = Variable(torch.zeros(1, self.args.hidden_size), volatile=True)
        hyp_scores = Variable(torch.zeros(1), volatile=True)
        if args.cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()

        eos_id = self.vocab['</s>']
        bos_id = self.vocab['<s>']
        tgt_vocab_size = len(self.vocab)

        hypotheses = [[bos_id]]
        completed_hypotheses = []
        completed_hypothesis_scores = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            t += 1
            hyp_num = len(hypotheses)

            # (hyp_num, encoding_size)
            expanded_src_encoding = src_images_encoding.expand(hyp_num, src_images_encoding.size(1))

            y_tm1 = Variable(torch.LongTensor([hyp[-1] for hyp in hypotheses]), volatile=True)
            if args.cuda:
                y_tm1 = y_tm1.cuda()

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t = expanded_src_encoding

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)
            p_t = F.log_softmax(score_t)

            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
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
                completed_hypotheses[i] = [self.vocab.id2word[w] for w in hyp]

        ranked_hypotheses = sorted(zip(completed_hypotheses, completed_hypothesis_scores), key=lambda x: x[1], reverse=True)

        return [hyp for hyp, score in ranked_hypotheses]

    def save(self, path):
        print('save parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


def sent_to_input_variable(sents, vocab, cuda=False, is_test=False):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """

    word_ids = word2id(sents, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var


def image_to_input_variable(image, cuda=False, is_test=False):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    elif isinstance(image, list):
        image = torch.stack(image)
    if cuda:
        return Variable(image.cuda(), volatile=is_test, requires_grad=False)
    else:
        return Variable(image, volatile=is_test, requires_grad=False)


def init_training(args):
    vocab = torch.load(args.vocab)

    model = NMT(args, vocab)
    model.train()

    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-args.uniform_init, args.uniform_init)

    vocab_mask = torch.ones(len(vocab))
    vocab_mask[vocab['<pad>']] = 0
    nll_loss = nn.NLLLoss(weight=vocab_mask, size_average=False)
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)

    if args.cuda:
        model = model.cuda()
        nll_loss = nll_loss.cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return vocab, model, optimizer, nll_loss, cross_entropy_loss


def train(args):
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')

    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')

    train_data = zip(train_data_src, train_data_tgt)
    dev_data = zip(dev_data_src, dev_data_tgt)

    vocab, model, optimizer, nll_loss, cross_entropy_loss = init_training(args)

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_iter = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1

            src_sents_var = sent_to_input_variable(src_sents, vocab.src, cuda=args.cuda)
            tgt_sents_var = sent_to_input_variable(tgt_sents, vocab.tgt, cuda=args.cuda)

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

                if args.valid_metric in ['bleu', 'word_acc', 'sent_acc']:
                    dev_hyps = decode(model, dev_data)
                    dev_hyps = [hyps[0] for hyps in dev_hyps]
                    if args.valid_metric == 'bleu':
                        valid_metric = get_bleu([tgt for src, tgt in dev_data], dev_hyps)
                    else:
                        valid_metric = get_acc([tgt for src, tgt in dev_data], dev_hyps, acc_type=args.valid_metric)
                    print('validation: iter %d, dev. ppl %f, dev. %s %f' % (train_iter, dev_ppl, args.valid_metric, valid_metric),
                          file=sys.stderr)
                else:
                    valid_metric = -dev_ppl
                    print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl),
                          file=sys.stderr)

                model.train()

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                is_better_than_last = len(hist_valid_scores) == 0 or valid_metric > hist_valid_scores[-1]
                hist_valid_scores.append(valid_metric)

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


def read_raml_train_data(data_file, temp):
    train_data = dict()
    num_pattern = re.compile('^(\d+) samples$')
    with open(data_file) as f:
        while True:
            line = f.readline()
            if line is None or line == '':
                break

            assert line.startswith('***')

            src_sent = f.readline()[len('source: '):].strip()
            tgt_num = int(num_pattern.match(f.readline().strip()).group(1))
            tgt_samples = []
            tgt_scores = []
            for i in xrange(tgt_num):
                d = f.readline().strip().split(' ||| ')
                if len(d) < 2:
                    continue

                tgt_sent = d[0].strip()
                bleu_score = float(d[1])
                tgt_samples.append(tgt_sent)
                tgt_scores.append(bleu_score / temp)

            tgt_scores = np.exp(tgt_scores)
            tgt_scores = tgt_scores / np.sum(tgt_scores)

            tgt_entry = zip(tgt_samples, tgt_scores)
            train_data[src_sent] = tgt_entry

            line = f.readline()

    return train_data


def train_raml(args):
    tau = args.temp

    dataset = Dataset(args.data_folder)

    vocab, model, optimizer, nll_loss, cross_entropy_loss = init_training(args)

    if args.raml_sample_mode == 'pre_sample':
        # dict of (src, [tgt: (sent, prob)])
        print('read in raml training data...', file=sys.stderr, end='')
        begin_time = time.time()
        raml_samples = read_raml_train_data(args.raml_sample_file, temp=tau)
        print('done[%d s].' % (time.time() - begin_time))
    elif args.raml_sample_mode.startswith('hamming_distance'):
        print('sample from hamming distance payoff distribution')
        payoff_prob, Z_qs = generate_hamming_distance_payoff_distribution(max_sent_len=50,
                                                                          vocab_size=len(vocab) - 3,
                                                                          tau=tau)

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    report_weighted_loss = cum_weighted_loss = 0
    cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_iter = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin RAML training')

    # smoothing function for BLEU
    sm_func = None
    if args.smooth_bleu:
        sm_func = SmoothingFunction().method3

    while True:
        epoch += 1
        for batch_examples in dataset.batch_iter('train', batch_size=args.batch_size, shuffle=True):
            train_iter += 1

            raml_src_images = []
            raml_tgt_sents = []
            raml_tgt_weights = []

            if args.raml_sample_mode == 'pre_sample':
                raise NotImplementedError()
            elif args.raml_sample_mode in ['hamming_distance', 'hamming_distance_impt_sample']:
                for src_image, tgt_sents in batch_examples:
                    tgt_samples = []  # make sure the ground truth y* is in the samples
                    references = [_tgt_sent[1:-1] for _tgt_sent in tgt_sents]  # remove <s> and </s>
                    tgt_sample_weights = []

                    # now we have multiple samples, we use ancestral sampling
                    tgt_sent_lens = [len(_tgt_sent) - 2 for _tgt_sent in tgt_sents]  # remove <s> and </s> # TODO: check if period in target!
                    tgt_sents_num = len(tgt_sents)
                    min_tgt_sent_len = min(tgt_sent_lens)

                    tgt_dist_samples = []
                    # make sure the ground truth {y*}'s are in the samples
                    if args.raml_bias_groundtruth:
                        for tgt_id in xrange(tgt_sents_num):
                            tgt_dist_samples.append((tgt_id, 0))

                    sample_size = args.sample_size - len(tgt_dist_samples)
                    if sample_size > 0:
                        tgt_sent_id_samples = np.random.choice(range(tgt_sents_num), size=sample_size, replace=True)
                        e_samples = np.random.choice(range(min_tgt_sent_len + 1), p=payoff_prob[min_tgt_sent_len],
                                                     size=sample_size, replace=True)
                        tgt_dist_samples += zip(tgt_sent_id_samples, e_samples)

                    for sample_id, (tgt_sent_id, e) in enumerate(tgt_dist_samples):
                        tgt_sent = tgt_sents[tgt_sent_id]
                        if e > 0:
                            # sample a new tgt_sent $y$
                            tgt_sent_len = tgt_sent_lens[tgt_sent_id]
                            old_word_pos = np.random.choice(range(1, tgt_sent_len + 1), size=e, replace=False)
                            new_words = [vocab.id2word[wid] for wid in np.random.randint(3, len(vocab), size=e)]
                            sampled_tgt_sent = list(tgt_sent)
                            for pos, word in zip(old_word_pos, new_words):
                                sampled_tgt_sent[pos] = word
                        else:
                            sampled_tgt_sent = list(tgt_sent)

                        # print('y: %s' % ' '.join(sampled_tgt_sent))
                        tgt_samples.append((sampled_tgt_sent, tgt_sent_id))

                        # if enable importance sampling, compute unnormalized q-distribution value: \tilde{q}(y|{y*})
                        if args.raml_sample_mode == 'hamming_distance_impt_sample':
                            bleu_scores = [1. if (e == 0 and _i == tgt_sent_id)
                                           else sentence_bleu([references[_i]],
                                                              sampled_tgt_sent[1: -1],
                                                              smoothing_function=sm_func)
                                           for _i in xrange(tgt_sents_num)]
                            q_tilde = math.exp((1. / tgt_sents_num) * sum(bleu_scores) / tau)

                            mixture_probs = []
                            for _i in xrange(tgt_sents_num):
                                if e == 0 and _i == tgt_sent_id: p = 1. / tgt_sents_num
                                else:
                                    ref_sent = references[_i]
                                    hm = hamming_distance(ref_sent, sampled_tgt_sent[1:-1])
                                    Z = Z_qs[len(ref_sent)]
                                    p = 1. / tgt_sents_num * math.exp(-hm / tau) / Z
                                mixture_probs.append(p)
                            propose_dist_tilde = sum(mixture_probs)

                            importance_weight = q_tilde / propose_dist_tilde
                            tgt_sample_weights.append(importance_weight)
                        else:
                            tgt_sample_weights.append(1.)

                    # normalize importance sampling weights
                    if args.raml_sample_mode == 'hamming_distance_impt_sample':
                        normalizer = sum(tgt_sample_weights)
                        tgt_sample_weights = [w / normalizer for w in tgt_sample_weights]

                    raml_src_images.extend([src_image] * len(tgt_samples))
                    raml_tgt_sents.extend([sample for sample, ref_sent_id in tgt_samples])
                    raml_tgt_weights.extend(tgt_sample_weights)

            src_images_var = image_to_input_variable(raml_src_images, cuda=args.cuda)
            tgt_sents_var = sent_to_input_variable(raml_tgt_sents, vocab, cuda=args.cuda)
            weights_var = Variable(torch.FloatTensor(raml_tgt_weights), requires_grad=False)
            if args.cuda:
                weights_var = weights_var.cuda()

            batch_size = len(raml_src_images)  # batch_size = args.batch_size * args.sample_size
            pred_tgt_word_num = sum(len(s[1:]) for s in raml_tgt_sents)  # omitting leading `<s>`
            optimizer.zero_grad()

            # (tgt_sent_len, batch_size, tgt_vocab_size)
            scores = model(src_images_var, tgt_sents_var[:-1])
            # (tgt_sent_len * batch_size, tgt_vocab_size)
            log_scores = F.log_softmax(scores.view(-1, scores.size(2)))
            # remove leading <s> in tgt sent, which is not used as the target
            flattened_tgt_sents = tgt_sents_var[1:].view(-1)

            # batch_size * tgt_sent_len
            tgt_log_scores = torch.gather(log_scores, dim=1, index=flattened_tgt_sents.unsqueeze(1)).squeeze(1)
            unweighted_loss = -tgt_log_scores * (1. - torch.eq(flattened_tgt_sents, 0).float())  # apply mask
            weighted_loss = unweighted_loss * weights_var.repeat(scores.size(0))
            weighted_loss = weighted_loss.sum()
            weighted_loss_val = weighted_loss.data[0]
            nll_loss_val = unweighted_loss.sum().data[0]
            # weighted_log_scores = log_scores * weights.view(-1, scores.size(2))
            # weighted_loss = nll_loss(weighted_log_scores, flattened_tgt_sents)

            loss = weighted_loss / batch_size
            # nll_loss_val = nll_loss(log_scores, flattened_tgt_sents).data[0]

            loss.backward()
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            optimizer.step()

            report_weighted_loss += weighted_loss_val
            cum_weighted_loss += weighted_loss_val
            report_loss += nll_loss_val
            cum_loss += nll_loss_val
            report_tgt_words += pred_tgt_word_num
            cum_tgt_words += pred_tgt_word_num
            report_examples += batch_size
            cum_examples += batch_size
            cum_batches += batch_size

            if train_iter % args.log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, '
                      'avg. ppl %.2f cum. examples %d, '
                      'speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                       report_weighted_loss / report_examples,
                                                                       np.exp(report_loss / report_tgt_words),
                                                                       cum_examples,
                                                                       report_tgt_words / (time.time() - train_time),
                                                                       time.time() - begin_time),
                      file=sys.stderr)

                train_time = time.time()
                report_loss = report_weighted_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % args.valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, '
                      'cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                          cum_weighted_loss / cum_batches,
                                                          np.exp(cum_loss / cum_tgt_words),
                                                          cum_examples),
                      file=sys.stderr)

                cum_loss = cum_weighted_loss = cum_batches = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)
                model.eval()

                # compute dev. ppl and bleu
                dev_hyps = decode(model, dataset.valid_examples())
                dev_hyps = [hyps[0] for hyps in dev_hyps]
                if args.valid_metric == 'bleu':
                    valid_metric = get_bleu([tgt for src, tgt in dataset.valid_examples()], dev_hyps)
                else:
                    valid_metric = get_acc([tgt for src, tgt in dataset.valid_examples()], dev_hyps, acc_type=args.valid_metric)
                print('validation: iter %d, dev. %s %f' % (train_iter, args.valid_metric, valid_metric),
                      file=sys.stderr)

                model.train()

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                is_better_than_last = len(hist_valid_scores) == 0 or valid_metric > hist_valid_scores[-1]
                hist_valid_scores.append(valid_metric)

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
    references = [[ref[1:-1] for ref in refs] for refs in references]
    hypotheses = [hyp[1:-1] for hyp in hypotheses]
    bleu_score = corpus_bleu(references, hypotheses)

    return bleu_score


def get_acc(references, hypotheses, acc_type='word'):
    assert acc_type == 'word_acc' or acc_type == 'sent_acc'
    cum_acc = 0.

    for refs, hyp in zip(references, hypotheses):
        refs = [ref[1:-1] for ref in refs]
        hyp = hyp[1:-1]
        if acc_type == 'word_acc':
            acc = max(len([1 for ref_w, hyp_w in zip(ref, hyp) if ref_w == hyp_w]) / float(len(hyp) + 1e-6) for ref in refs)
        else:
            acc = 1. if max(all(ref_w == hyp_w for ref_w, hyp_w in zip(ref, hyp)) for ref in refs) else 0.
        cum_acc += acc

    acc = cum_acc / len(hypotheses)
    return acc


def decode(model, data, verbose=True):
    """
    decode the dataset and compute sentence level acc. and BLEU.
    """
    hypotheses = []
    begin_time = time.time()

    example_num = 0
    for src_image, tgt_sents in data:
        hyps = model.translate(src_image)
        hypotheses.append(hyps)
        example_num += 1

        if verbose:
            print('*' * 50)
            for i, tgt_sent in enumerate(tgt_sents):
                print('Target %d: %s' % (i, ' '.join(tgt_sent)))
            print('Top Hypothesis: ', ' '.join(hyps[0]))

    elapsed = time.time() - begin_time

    print('decoded %d examples, took %d s' % (example_num, elapsed), file=sys.stderr)

    return hypotheses


def compute_lm_prob(args):
    """
    given source-target sentence pairs, compute ppl and log-likelihood
    """
    test_data_src = read_corpus(args.test_src, source='src')
    test_data_tgt = read_corpus(args.test_tgt, source='tgt')
    test_data = zip(test_data_src, test_data_tgt)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        state_dict = params['state_dict']

        model = NMT(saved_args, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = model.cuda()

    f = open(args.save_to_file, 'w')
    for src_sent, tgt_sent in test_data:
        src_sents = [src_sent]
        tgt_sents = [tgt_sent]

        batch_size = len(src_sents)
        src_sents_len = [len(s) for s in src_sents]
        pred_tgt_word_nums = [len(s[1:]) for s in tgt_sents]  # omitting leading `<s>`

        # (sent_len, batch_size)
        src_sents_var = sent_to_input_variable(src_sents, model.vocab.src, cuda=args.cuda, is_test=True)
        tgt_sents_var = sent_to_input_variable(tgt_sents, model.vocab.tgt, cuda=args.cuda, is_test=True)

        # (tgt_sent_len, batch_size, tgt_vocab_size)
        scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
        # (tgt_sent_len * batch_size, tgt_vocab_size)
        log_scores = F.log_softmax(scores.view(-1, scores.size(2)))
        # remove leading <s> in tgt sent, which is not used as the target
        # (batch_size * tgt_sent_len)
        flattened_tgt_sents = tgt_sents_var[1:].view(-1)
        # (batch_size * tgt_sent_len)
        tgt_log_scores = torch.gather(log_scores, 1, flattened_tgt_sents.unsqueeze(1)).squeeze(1)
        # 0-index is the <pad> symbol
        tgt_log_scores = tgt_log_scores * (1. - torch.eq(flattened_tgt_sents, 0).float())
        # (tgt_sent_len, batch_size)
        tgt_log_scores = tgt_log_scores.view(-1, batch_size) # .permute(1, 0)
        # (batch_size)
        tgt_sent_scores = tgt_log_scores.sum(dim=0).squeeze()
        tgt_sent_word_scores = [tgt_sent_scores[i].data[0] / pred_tgt_word_nums[i] for i in xrange(batch_size)]

        for src_sent, tgt_sent, score in zip(src_sents, tgt_sents, tgt_sent_word_scores):
            f.write('%s ||| %s ||| %f\n' % (' '.join(src_sent), ' '.join(tgt_sent), score))

    f.close()


def test(args):
    dataset = Dataset(args.data_folder)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        state_dict = params['state_dict']

        model = NMT(saved_args, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = model.cuda()

    hypotheses = decode(model, dataset.test_examples())
    top_hypotheses = [hyps[0] for hyps in hypotheses]

    bleu_score = get_bleu([tgt for src, tgt in dataset.test_examples()], top_hypotheses)
    word_acc = get_acc([tgt for src, tgt in dataset.test_examples()], top_hypotheses, 'word_acc')
    sent_acc = get_acc([tgt for src, tgt in dataset.test_examples()], top_hypotheses, 'sent_acc')
    print('Corpus Level BLEU: %f, word level acc: %f, sentence level acc: %f' % (bleu_score, word_acc, sent_acc),
          file=sys.stderr)

    if args.save_to_file:
        print('save decoding results to %s' % args.save_to_file, file=sys.stderr)
        with open(args.save_to_file, 'w') as f:
            for hyps in hypotheses:
                f.write(' '.join(hyps[0][1:-1]) + '\n')


def interactive(args):
    assert args.load_model, 'You have to specify a pre-trained model'
    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    vocab = params['vocab']
    saved_args = params['args']
    state_dict = params['state_dict']

    model = NMT(saved_args, vocab)
    model.load_state_dict(state_dict)

    model.eval()

    if args.cuda:
        model = model.cuda()

    while True:
        src_sent = raw_input('Source Sentence:')
        src_sent = src_sent.strip().split(' ')
        hyps = model.translate(src_sent)
        for i, hyp in enumerate(hyps, 1):
            print('Hypothesis #%d: %s' % (i, ' '.join(hyp)))


def sample(args):
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')
    train_data = zip(train_data_src, train_data_tgt)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        opt = params['args']
        state_dict = params['state_dict']

        model = NMT(opt, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = model.cuda()

    print('begin sampling')

    check_every = 10
    train_iter = cum_samples = 0
    train_time = time.time()
    for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
        train_iter += 1
        samples = model.sample(src_sents, sample_size=args.sample_size, to_word=True)
        cum_samples += sum(len(sample) for sample in samples)

        if train_iter % check_every == 0:
            elapsed = time.time() - train_time
            print('sampling speed: %d/s' % (cum_samples / elapsed), file=sys.stderr)
            cum_samples = 0
            train_time = time.time()

        for i, tgt_sent in enumerate(tgt_sents):
            print('*' * 80)
            print('target:' + ' '.join(tgt_sent))
            tgt_samples = samples[i]
            print('samples:')
            for sid, sample in enumerate(tgt_samples, 1):
                print('[%d] %s' % (sid, ' '.join(sample[1:-1])))
            print('*' * 80)


if __name__ == '__main__':
    args = init_config()
    print(args, file=sys.stderr)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'raml_train':
        train_raml(args)
    elif args.mode == 'sample':
        sample(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'prob':
        compute_lm_prob(args)
    elif args.mode == 'interactive':
        interactive(args)
    else:
        raise RuntimeError('unknown mode')