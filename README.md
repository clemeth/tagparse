# Tagparse

A tagger and a parser that can be trained separately or jointly.

The parser is a shallow version of Dozat & Manning's 2017 deep biaffine attention parser, as devised by Glavaš and Vulić (2021b).

The parser implementation builds on [TowerParse](https://github.com/codogogo/towerparse) (Glavaš and Vulić, 2021a). Credit is given in the source code where relevant.

## Instructions

To train a model, perform the following steps:

0. Dependencies can be installed with `pipenv` using the `Pipfile` directly. Otherwise, inspect the file and install them how you please.
1. Extract the label vocabulary from your training files using the `extract_vocabs` method in `utils.py`.
2. Edit `config.py` to your needs. The following parameters are available:
  - `batch-size` (int): The batch size.
  - `bert_lr` (float): The learning rate for the BERT parameters.
  - `cased` (bool): `True` to keep letter case in training data. `False` to lowercase.
  - `parser_lr` (float): The learning rate for the parser parameters.
  - `tagger_lr` (float): The learning rate for the tagger parameters.
  - `device` (str): The processing device to run on. `'cpu'` for a regular CPU, and typically something like `'cuda:0'` for GPUs.
  - `early_stop` (int): The number of epochs after which to quit training if validation loss does not decrease.
  - `epochs` (int): The maximum number of epochs to run training for.
  - `expand_iota` (bool): `True` to adscript iota supscripts, `False` to do nothing.
  - `expand_rough` (bool): `True` to add a *heta* to words with rough breathing. `False` to do nothing.
  - `ignore_punct` (bool): `True` to ignore punctuation and gap tokens during evaluation with `test.py`.
  - `last_layer_dropout` (float): The dropout probability of the last layer.
  - `max_subword_len` (int): The maximum number of subword tokens per sentence. Needs to be higher than the maximum number of subword tokens that any sentence in the data is tokenized into.
  - `max_word_len` (int): The maximum number of word tokens per sentence. Needs to be higher than the longest sentence in the data.
  - `mode` (str): `'tag'`, `'parse'` or `'joint'`.
  - `model_name` (str): The Hugging Face path or local path to the transformer model to use. The bundled tokenizer will also be loaded.
  - `models_path` (str): Path to where models are saved.
  - `name` (str): A name for the model to be trained/loaded.
  - `num_warmup_steps` (int): The number of warmup steps for the optimizer.
  - `pad_value` (int): A pad value. Needs to be negative.
  - `print_gold` (bool): `True` to print the gold annotation in predictions during evaluation with `test.py`.
  - `scheduler` (str): The type of scheduler to load from the `get_scheduler` method from `transformers`.
  - `seed` (int): The RNG seed.
  - `subword_prefix` (str): The subword prefix used by the transformer model.
  - `test_path`: Path to the CoNLL-U file with testing data.
  - `train_path`: Path to the CoNLL-U file with training data.
  - `val_path`: Path to the CoNLL-U file with validation data.
  - `vocabs_path`: Path to the JSON file with the extracted label vocabulary (see step 1)
3. Run `train.py`.

To evaluate a model, keep the same config file and run `test.py`. 

## Bibliography

- Dozat, T. and C. D. Manning. 2017. Deep Biaffine Attention for Neural Dependency Parsing. Proceedings of ICLR 2017. <https://openreview.net/forum?id=Hk95PK9le>.
- Glavaš, G. and I. Vulić. 2021a. Climbing the Tower of Treebanks: Improving Low-Resource Dependency Parsing via Hierarchical Source Selection. Findings of ACL-IJCNLP 2021, pp. 4878–4888. <https://dx.doi.org/10.18653/v1/2021.findings-acl.431>
- Glavaš, G. and I. Vulić. 2021b. Is Supervised Syntactic Parsing Beneficial for Language Understanding Tasks? An Empirical Investigation. Proceedings of ACL, pp. 3090–3104. <https://dx.doi.org/10.18653/v1/2021.eacl-main.270>.