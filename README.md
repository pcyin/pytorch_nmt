**A neural machine translation model written in pytorch.**

With 256-dimensional LSTM hidden size, it achieves a training speed of 14000 words/sec and 26.9 BLEU score on the IWSLT 2014 Germen-English dataset (Ranzato et al., 2015).

## File Structure

* `nmt.py`: main file
* `vocab.py`: script used to generate `.bin` vocabulary file from parallel corpus
* `util.py`: script containing helper functions
* `run_raml_exp.py|test_raml_models.py`: helper scripts to train|test RAML models with different temperature settings (refer to [Norouzi et al., 2016] for details)

## Usage

* Generate Vocabulary Files

```
python vocab.py
```

* Vanilla Maximum Likelihood Training

```
. scripts/run_mle.sh
```

* Reward Augmented Maximum Likelihood Training (Norouzi et al., 2016)

```
. scripts/run_raml.sh
```

* Reinforcement Learning (Coming soon)

## TODO:

* batched decoding as in openNMT
