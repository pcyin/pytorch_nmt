from __future__ import print_function
from nltk.translate.bleu_score import sentence_bleu
import sys
import re
import argparse
import torch
from util import read_corpus
import numpy as np
from scipy.misc import comb
from vocab import Vocab, VocabEntry
import math


def is_valid_sample(sent):
    tokens = sent.split(' ')
    return len(tokens) >= 1 and len(tokens) < 50


def sample_from_model(args):
    para_data = args.parallel_data
    sample_file = args.sample_file
    output = args.output

    tgt_sent_pattern = re.compile('^\[(\d+)\] (.*?)$')
    para_data = [l.strip().split(' ||| ') for l in open(para_data)]

    f_out = open(output, 'w')
    f = open(sample_file)
    f.readline()
    for src_sent, tgt_sent in para_data:
        line = f.readline().strip()
        assert line.startswith('****')
        line = f.readline().strip()
        print(line)
        assert line.startswith('target:')

        tgt_sent2 = line[len('target:'):]
        assert tgt_sent == tgt_sent2

        line = f.readline().strip() # samples

        tgt_sent = ' '.join(tgt_sent.split(' ')[1:-1])
        tgt_samples = set()
        for i in xrange(1, 101):
            line = f.readline().rstrip('\n')
            m = tgt_sent_pattern.match(line)

            assert m, line
            assert int(m.group(1)) == i

            sampled_tgt_sent = m.group(2).strip()

            if is_valid_sample(sampled_tgt_sent):
                tgt_samples.add(sampled_tgt_sent)

        line = f.readline().strip()
        assert line.startswith('****')

        tgt_samples.add(tgt_sent)
        tgt_samples = list(tgt_samples)

        assert len(tgt_samples) > 0

        tgt_ref_tokens = tgt_sent.split(' ')
        bleu_scores = []
        for tgt_sample in tgt_samples:
            bleu_score = sentence_bleu([tgt_ref_tokens], tgt_sample.split(' '))
            bleu_scores.append(bleu_score)

        tgt_ranks = sorted(range(len(tgt_samples)), key=lambda i: bleu_scores[i], reverse=True)

        print('%d samples' % len(tgt_samples))

        print('*' * 50, file=f_out)
        print('source: ' + src_sent, file=f_out)
        print('%d samples' % len(tgt_samples), file=f_out)
        for i in tgt_ranks:
            print('%s ||| %f' % (tgt_samples[i], bleu_scores[i]), file=f_out)
        print('*' * 50, file=f_out)

    f_out.close()


def get_new_ngram(ngram, n, vocab):
    """
    replace ngram `ngram` with a newly sampled ngram of the same length
    """

    new_ngram_wids = [np.random.randint(3, len(vocab)) for i in xrange(n)]
    new_ngram = [vocab.id2word[wid] for wid in new_ngram_wids]

    return new_ngram


def sample_ngram(args):
    src_sents = read_corpus(args.src, 'src')
    tgt_sents = read_corpus(args.tgt, 'src')  # do not read in <s> and </s>
    f_out = open(args.output, 'w')

    vocab = torch.load(args.vocab)
    tgt_vocab = vocab.tgt

    for src_sent, tgt_sent in zip(src_sents, tgt_sents):
        src_sent = ' '.join(src_sent)

        tgt_len = len(tgt_sent)
        tgt_samples = []
        tgt_samples_distort_rates = []    # how many unigrams are replaced

        # generate 100 samples

        # append itself
        tgt_samples.append(tgt_sent)
        tgt_samples_distort_rates.append(0)

        for sid in xrange(args.sample_size - 1):
            n = np.random.randint(1, min(tgt_len, args.max_ngram_size + 1)) # we do not replace the last token: it must be a period!

            idx = np.random.randint(tgt_len - n)
            ngram = tgt_sent[idx: idx+n]
            new_ngram = get_new_ngram(ngram, n, tgt_vocab)

            sampled_tgt_sent = list(tgt_sent)
            sampled_tgt_sent[idx: idx+n] = new_ngram

            # compute the probability of this sample
            # prob = 1. / args.max_ngram_size * 1. / (tgt_len - 1 + n) * 1 / (len(tgt_vocab) ** n)

            tgt_samples.append(sampled_tgt_sent)
            tgt_samples_distort_rates.append(n)

        # compute bleu scores or edit distances and rank the samples by bleu scores
        rewards = []
        for tgt_sample, tgt_sample_distort_rate in zip(tgt_samples, tgt_samples_distort_rates):
            if args.reward == 'bleu':
                reward = sentence_bleu([tgt_sent], tgt_sample)
            else:
                reward = -tgt_sample_distort_rate

            rewards.append(reward)

        tgt_ranks = sorted(range(len(tgt_samples)), key=lambda i: rewards[i], reverse=True)
        # convert list of tokens into a string
        tgt_samples = [' '.join(tgt_sample) for tgt_sample in tgt_samples]

        print('*' * 50, file=f_out)
        print('source: ' + src_sent, file=f_out)
        print('%d samples' % len(tgt_samples), file=f_out)
        for i in tgt_ranks:
            print('%s ||| %f' % (tgt_samples[i], rewards[i]), file=f_out)
        print('*' * 50, file=f_out)

    f_out.close()


def generate_hamming_distance_payoff_distribution(max_sent_len, tau=1.):
    """compute the q distribution for Hamming Distance (substitution only) as in the RAML paper"""
    probs = dict()
    Z_qs = dict()
    for sent_len in xrange(1, max_sent_len + 1):
        counts = [1]  # e = 0, count = 1
        for e in xrange(1, sent_len + 1):
            counts.append(comb(sent_len, e))

        weighted_counts = [math.exp(-e / tau) * count for e, count in enumerate(counts)]
        Z_qs[sent_len] = Z_q = sum(weighted_counts)
        prob = [count / Z_q for count in weighted_counts]
        probs[sent_len] = prob

        # print('sent_len=%d, %s' % (sent_len, prob))

    return probs, Z_qs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sample_from_model', 'sample_ngram'], required=True)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--src', type=str)
    parser.add_argument('--tgt', type=str)
    parser.add_argument('--parallel_data', type=str)
    parser.add_argument('--sample_file', type=str)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--reward', choices=['bleu', 'edit_dist'], default='bleu')
    parser.add_argument('--max_ngram_size', type=int, default=4)

    args = parser.parse_args()

    if args.mode == 'sample_ngram':
        sample_ngram(args)
    elif args.mode == 'sample_from_model':
        sample_from_model(args)