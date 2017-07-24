from collections import defaultdict
import numpy as np
import torch
from torch.autograd import Variable


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in xrange(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        batch_data = [data[i * batch_size + b] for b in xrange(cur_batch_size)]

        yield batch_data


def length_array_to_mask_tensor(length_array, cuda=False):
    max_len = length_array[0]
    batch_size = len(length_array)

    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        mask[i][:seq_len] = 0

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def data_iter_ensure_length(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle: np.random.shuffle(tuples)

        batches = batch_size(tuples, batch_size)
        for batch in batches:
            src_sents, tgt_sents = zip(*batch)
            cur_batch_size = len(batch)

            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

            batched_data.append((src_sents, tgt_sents))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def data_iter(data, batch_size, shuffle=True):
    """
        randomly permute data and partition into batches
    """
    data_size = len(data)
    indices = np.array(range(data_size))

    if shuffle:
        np.random.shuffle(indices)

    batch_slices = batch_slice(indices, batch_size)
    for item_indices in batch_slices:
        batch = [data[i] for i in item_indices]
        src_sents, tgt_sents = zip(*batch)

        src_ids = sorted(range(len(batch)), key=lambda src_id: len(src_sents[src_id]), reverse=True)
        src_sents = [src_sents[src_id] for src_id in src_ids]
        tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        yield src_sents, tgt_sents


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
    return linear(X.contiguous().view(-1, X.size(2))).view(X.size(0), X.size(1), -1)