from __future__ import print_function
from nmt import *
from torch.nn import utils


class RLNMT(NMT):
    def __init__(self, args, vocab):
        super(RLNMT, self).__init__(args, vocab)

        # RL-specific parameters - baseline
        self.baseline = nn.Linear(args.decoder_hidden_size, 1, bias=True)
        self.baseline.bias.data.zero_()

    def get_rl_loss(self, src_sents, tgt_sents, reward_func, sample_size=None):
        if sample_size is None:
            sample_size = self.args.sample_size

        src_sents_num = len(src_sents)
        batch_size = src_sents_num * sample_size

        # (src_sent_len, src_sents_num)
        src_sents_var = to_input_variable(src_sents, self.vocab.src, cuda=self.args.cuda, is_test=False)
        # (src_sent_len, src_sents_num, encoder_hidden_size)
        src_encoding, (dec_init_state, dec_init_cell) = self.encode(src_sents_var, [len(s) for s in src_sents])

        # (batch_size, hidden_size)
        dec_init_state = dec_init_state.repeat(sample_size, 1)
        dec_init_cell = dec_init_cell.repeat(sample_size, 1)
        hidden = (dec_init_state, dec_init_cell)

        # (src_sent_len, batch_size, encoder_hidden_size)
        src_encoding = src_encoding.repeat(1, sample_size, 1)
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)
        # (batch_size, src_sent_len, encoder_hidden_size)
        src_encoding = src_encoding.t()
        # (batch_size, src_sent_len, hidden_size)
        src_encoding_att_linear = src_encoding_att_linear.t()

        new_tensor = dec_init_state.data.new
        att_tm1 = Variable(new_tensor(batch_size, self.args.decoder_hidden_size).zero_())
        y_0 = Variable(torch.LongTensor([self.vocab.tgt['<s>'] for _ in xrange(batch_size)]))

        eos = self.vocab.tgt['</s>']
        sample_ends = torch.ByteTensor([0] * batch_size)
        all_ones = torch.ByteTensor([1] * batch_size)
        if self.args.cuda:
            y_0 = y_0.cuda()
            sample_ends = sample_ends.cuda()
            all_ones = all_ones.cuda()
        new_byte_tensor = sample_ends.new

        samples = [y_0]
        sample_losses = []
        baselines = []
        maskes = []
        # self.decoder_lstm.set_dropout_masks(batch_size)
        t = 0
        while t < self.args.decode_max_time_step:
            t += 1

            # (batch_size)
            y_tm1 = samples[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding, src_encoding_att_linear)

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
            att_t = self.dropout(att_t)

            # (batch_size, vocab_size)
            score_t = self.readout(att_t)  # E.q. (6)
            p_t = F.softmax(score_t)

            # (batch_size, 1)
            if self.args.sample_method == 'random':
                y_t = torch.multinomial(p_t, num_samples=1).detach()
            elif self.args.sample_method == 'greedy':
                _, y_t = torch.topk(p_t, k=1, dim=1)

            # (batch_size)
            sample_p_t = torch.gather(p_t, 1, y_t).squeeze(1)
            y_t = y_t.squeeze(1)
            sample_loss_t = -torch.log(sample_p_t)
            b_t = F.tanh(self.baseline(h_t.detach()))

            samples.append(y_t)
            sample_losses.append(sample_loss_t)
            baselines.append(b_t)

            sample_ends |= torch.eq(y_t, eos).byte().data
            if torch.equal(sample_ends, all_ones):
                break

            att_tm1 = att_t
            hidden = h_t, cell_t

        # compute sequence-level reward
        completed_samples = [[] for _ in xrange(batch_size)]
        for t, y_t in enumerate(samples[1:]):  # omit leading <s>
            mask_t = []
            for j, wid in enumerate(y_t.cpu().data):
                is_valid = t == 0 or completed_samples[j][-1] != eos
                if is_valid:
                    completed_samples[j].append(wid)
                    mask_t.append(0)
                else:
                    mask_t.append(1)
            mask_t = new_byte_tensor(mask_t)
            maskes.append(mask_t)

        for i, src_sent_samples in enumerate(completed_samples):
            completed_samples[i] = word2id(src_sent_samples, self.vocab.tgt.id2word)

        rewards = []
        for i, hyp_sent in enumerate(completed_samples):
            tgt_sent_id = i % src_sents_num
            tgt_sent = tgt_sents[tgt_sent_id][1:-1]  # remove <s> </s>
            hyp_sent = hyp_sent[:-1]  # remove ending </s>
            r = reward_func(hyp_sent, tgt_sent)
            rewards.append(r)
        rewards = Variable(new_tensor(rewards))

        # get the loss!
        losses = []
        losses_b = []
        for t, (loss_t, b_t, mask_t) in enumerate(zip(sample_losses, baselines, maskes)):
            loss_t = (rewards - b_t) * loss_t
            loss_bt = (rewards - b_t) ** 2

            loss_t.data.masked_fill_(mask_t, 0.)
            loss_bt.data.masked_fill_(mask_t, 0.)

            losses.append(loss_t)
            losses_b.append(loss_bt)

        loss_rl = torch.cat(losses).sum() / batch_size
        loss_b = torch.cat(losses_b).sum() / batch_size

        return loss_rl, loss_b


def train_rl(args):
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')

    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')

    train_data = zip(train_data_src, train_data_tgt)
    dev_data = zip(dev_data_src, dev_data_tgt)

    vocab = torch.load(args.vocab)
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        state_dict = params['state_dict']
        # TODO: how to revise arg copying?
        saved_args.sample_method = args.sample_method
        saved_args.sample_size = args.sample_size

        model = RLNMT(saved_args, vocab)
        try:
            model.load_state_dict(state_dict)
        except KeyError:
            print('*** Warning Loading State Dict: missing parameters ***', file=sys.stderr)
    else:
        vocab = torch.load(args.vocab)
        model = RLNMT(args, vocab)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model = model.cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()

    # define reward function
    sm_func = SmoothingFunction().method3
    def reward_func(hyp, tgt):
        return sentence_bleu([tgt], hyp, smoothing_function=sm_func)

    train_iter = patience = report_loss = report_rl_loss = report_b_loss = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = best_model_iter = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin REINFORCE Learning', file=sys.stderr)

    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1
            batch_size = args.sample_size * len(src_sents)

            optimizer.zero_grad()

            loss_rl, loss_b = model.get_rl_loss(src_sents, tgt_sents,
                                                reward_func=reward_func, sample_size=args.sample_size)

            loss = loss_rl + loss_b
            loss_val = loss.data[0]
            loss_rl_val = loss_rl.data[0]
            loss_b_val = loss_b.data[0]

            loss.backward()
            # clip gradient
            grad_norm = nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            optimizer.step()

            report_loss += loss_val * batch_size
            report_rl_loss += loss_rl_val * batch_size
            report_b_loss += loss_b_val * batch_size
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % args.log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, '
                      'avg. rl loss %.2f, avg. baseline loss %.2f, ' \
                      'cum. examples %d, speed %.2f examples/sec, '
                      'time elapsed %.2f sec' % (epoch, train_iter, report_loss / report_examples,
                                                 report_rl_loss / report_examples, report_b_loss / report_examples,
                                                 cum_examples, report_examples / (time.time() - train_time),
                                                 time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_rl_loss = report_b_loss = report_examples = 0.

            # perform validation
            if train_iter % args.valid_niter == 0:
                valid_num += 1

                print('begin validation ...', file=sys.stderr)
                model.eval()

                # compute dev. ppl and bleu
                dev_loss = evaluate_loss(model, dev_data, cross_entropy_loss, args)
                dev_ppl = np.exp(dev_loss)

                if args.valid_metric in ['bleu', 'word_acc', 'sent_acc', 'logical_form_acc']:
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
