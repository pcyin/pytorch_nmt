from collections import defaultdict
import numpy as np

def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_slice(data, batch_size, sort=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in xrange(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        yield src_sents, tgt_sents


def data_iter(data, batch_size, shuffle=True):
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
        batched_data.extend(list(batch_slice(tuples, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def generate_force_attention_map(src_sents, tgt_sents, force_attention_rules):
    force_attention_map = defaultdict(list)
    for src_words, tgt_word in force_attention_rules:
        src_words = src_words.split(' ')

        for batch_id, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
            if tgt_word in tgt_sent:
                tgt_word_idx = tgt_sent.index(tgt_word)
                src_word_idx = seq_index(src_sent, src_words)
                if src_word_idx:
                    # currently we only use the first occurrence of source word
                    for _src_idx in xrange(src_word_idx[0][0], src_word_idx[0][1]):
                        force_attention_map[tgt_word_idx].append((batch_id, _src_idx))

    return force_attention_map


def seq_index(a, b):
    if b[0] in a:
        return [(i, i + len(b)) for i in xrange(len(a)) if a[i:i + len(b)] == b]
    return []