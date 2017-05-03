import os

train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

for temp in [0.2, 0.3, 0.6, 0.7]:
    job_name = 'iwslt14.raml.512enc.corrupt_ngram.10samples.t%.3f' % temp
    train_log = 'train.' + job_name + '.log'
    model_name = 'model.' + job_name
    job_file = 'scripts/train.%s.sh' % job_name
    decode_file = job_name + '.test.en'
    with open(job_file, 'w') as f:
        f.write("""#!/bin/sh

echo Current commit id: `git rev-parse HEAD` > logs/{train_log}

python nmt.py \
    --cuda \
    --mode raml_train \
    --vocab data/iwslt.vocab.bin \
    --save_to models/{model_name} \
    --valid_niter 15400 \
    --valid_metric ppl \
    --beam_size 5 \
    --batch_size 10 \
    --sample_size 10 \
    --hidden_size 256 \
    --embed_size 256 \
    --uniform_init 0.1 \
    --dropout 0.2 \
    --clip_grad 5.0 \
    --lr_decay 0.5 \
    --temp {temp} \
    --raml_sample_file data/samples.corrupt_ngram.10.bleu_score.txt \
    --raml_sample_mode uniform_n_weight \
    --train_src {train_src} \
    --train_tgt {train_tgt} \
    --dev_src {dev_src} \
    --dev_tgt {dev_tgt} 2>>logs/{train_log}

python nmt.py \
    --cuda \
    --mode test \
    --load_model models/{model_name}.bin \
    --beam_size 5 \
    --decode_max_time_step 100 \
    --save_to_file decode/{decode_file} \
    --test_src {test_src} \
    --test_tgt {test_tgt}

echo "test result" >> logs/{train_log}
perl multi-bleu.perl {test_tgt} < decode/{decode_file} >> logs/{train_log}
      
""".format(model_name=model_name, temp=temp, 
    train_src=train_src, train_tgt=train_tgt, 
    dev_src=dev_src, dev_tgt=dev_tgt,
    test_src=test_src, test_tgt=test_tgt,
    train_log=train_log, decode_file=decode_file))

    os.system('bash submit_job.sh %s' % job_file)
