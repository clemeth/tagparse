from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

import config

import conllu
import greek_accentuation.characters
import json
import numpy as np
import random
import string
import torch
import torch.nn.functional as F

CONLLU_COLUMNS = [
    'id',
    'form',
    'lemma',
    'upos',
    'xpos',
    'feats',
    'head',
    'deprel',
]

CONLLU_COLUMNS_WITH_VOCAB = [
    'upos', 'xpos', 'feats', 'deprel'
]

IGNORE_FEATS = ['VERBSEM', 'NOUNSEM']

LANG_SPECIFIC_TOKENS = [
    'ϝ',
    '𐅵',
    '𐅃',
    '𐅄',
    'ϛ',
    'ͱ',
    '†',
]

REPLACE_DICT = {
    'γ̓': 'γ',
    'γ̣': 'γ',
    'δ̓': 'δ',
    'ε̣': 'ε',
    'ῃ̈́': 'η',
    'θ̓': 'θ',
    'λ̓': 'λ',
    'ο͂': 'ο',
    'π̓': 'π',
    'τ̓': 'τ',
    'τ̈': 'τ',
    'φ̓': 'φ',
    'ω̄': 'ω',
    'ʹ': '’', # U+2019 : RIGHT SINGLE QUOTATION MARK is the correct apostrophe mark
    ' ́': '’',
    'ʼ': '’',
    ' ̓': '’',
    ' ̔': '’',
    '̓': '’',
    '̔': '’',
    '̔ ̔': '’',
    '́"': '\"',
    '”': '\"',
    '“': '\"',
    '—': '-'
}

IOTA_CHARS = 'ῳῴῲῷᾠᾤᾢᾦᾡᾥᾣᾧ' + 'ῃῄῂῇᾐᾔᾒᾖᾑᾕᾓᾗ' + 'ᾳᾴᾲᾷᾀᾄᾂᾆᾁᾅᾃᾇ' + \
             'ῼᾨᾬᾪᾮᾩᾯᾫᾯ'    + 'ῌᾘᾜᾚᾞᾙᾝᾛᾟ' + 'ᾼᾈᾌᾊᾎᾉᾍᾋᾏ'

ROUGH_CHARS = 'ἁἅἃἇᾁᾅᾃᾇ' + 'ἑἕἓ' + 'ἡἥἣἧᾑᾕᾓᾗ' + 'ἱἵἳἷ' + 'ὁὅὃ' + 'ὑὕὓὗ' + 'ὡὥὣὧᾡᾥᾣᾧ' + \
              'ἉἍἋἏᾉᾍᾋᾏ' + 'ἙἝἛ' + 'ἩἭἫἯᾙᾝᾛᾟ' + 'ἹἽἻἿ' + 'ὉὍὋ' + 'ὙὝὛὟ' + 'ὩὭὫὯᾩᾭᾫᾯ'

CONSONANTS = 'βγδζͱθκλμνξπρστφχψ' + 'ΒΓΔΖͰΘΚΛΜΝΞΠΡΣΤΦΧΨ'


class SetEncoder(json.JSONEncoder):
   def default(self, obj):
      if isinstance(obj, set):
         return list(obj)
      return json.JSONEncoder.default(self, obj)


class CoNLLDataset(Dataset):
    def __init__(self, data_path, cap=None):
        with open(data_path) as f:
            print(f"Parsing {data_path}")
            sents = conllu.parse(f.read(), fields=CONLLU_COLUMNS)
            print(f"Finished parsing {data_path}")
        try:
            self.data = sents[:cap]
        except:
            self.data = sents
        self.vocabs = load_vocabs(config.vocabs_path)
        self.label2idx = {}
        self.idx2label = {}
        for column in CONLLU_COLUMNS_WITH_VOCAB:
            self.label2idx[column] = {}
            self.idx2label[column] = {}
            for idx, label in enumerate(self.vocabs[column]):
                self.label2idx[column][label] = idx
                self.idx2label[column][idx] = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        tokens = []
        labels = defaultdict(list)
        for token in sent:
            tokens.append(token['form'])
            labels['upos'].append(self.label2idx['upos'][token['upos']])
            try:
                labels['xpos'].append(self.label2idx['xpos'][token['xpos']])
            except KeyError:
                labels['xpos'].append(self.label2idx['xpos']['[UNK_XPOS]'])
            try:
                labels['feats'].append(
                    self.label2idx['feats'][feats_to_string(token['feats'])])
            except KeyError:
                labels['feats'].append(self.label2idx['feats']['[UNK_FEATS]'])
            labels['head'].append(token['head'])
            labels['deprel'].append(self.label2idx['deprel'][token['deprel']])

        return tokens, labels


def extract_vocabs(input_path, output_path):
    vocabs = {}
    for column in CONLLU_COLUMNS_WITH_VOCAB:
        vocabs[column] = set()

    with open(input_path) as in_:
        sents = conllu.parse(in_.read())
        for sent in tqdm(sents):
            for token in sent:
                for column, label in token.items():
                    if column not in CONLLU_COLUMNS_WITH_VOCAB:
                        continue
                    if column == 'feats':
                        label = feats_to_string(label)
                    vocabs[column].add(label)

    with open(output_path, 'w') as out:
        json.dump(vocabs, out, cls=SetEncoder)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_language_specific_tokens(model, tokenizer):
    # A Greek specialty
    tokenizer.add_tokens(LANG_SPECIFIC_TOKENS)
    model.resize_token_embeddings(len(tokenizer))


def load_vocabs(vocabs_path):
    with open(vocabs_path) as f:
        vocabs = json.load(f)
        vocabs['feats'].insert(0, '[UNK_FEATS]')
        vocabs['xpos'].insert(1, '[UNK_XPOS]')
    return vocabs


def collate_fn(batch, tokenizer):
    device = config.device
    mode = config.mode

    # Featurize tokens
    input_ids, attention_masks, word_start_positions, lengths = zip(
        *[featurize_tokens(x, tokenizer) for (x, Y) in batch])

    X = {
        'input_ids': torch.stack(input_ids).to(device),
        'attention_mask': torch.stack(attention_masks).to(device),
        'word_start_positions': torch.stack(word_start_positions).to(device),
        'lengths': torch.stack(lengths).to(device)
    }

    # Featurize labels
    featurized_labels = zip(
        *[featurize_labels(Y) for (x, Y) in batch])

    Y = defaultdict()

    if mode in ['tag', 'joint']:
        Y['upos'] = torch.stack(next(featurized_labels)).to(device)
        Y['xpos'] = torch.stack(next(featurized_labels)).to(device)
        Y['feats'] = torch.stack(next(featurized_labels)).to(device)
    if mode in ['parse', 'joint']:
        Y['head'] = torch.stack(next(featurized_labels)).to(device)
        Y['deprel'] = torch.stack(next(featurized_labels)).to(device)

    return (X, Y)


def normalize_tokens(tokens):
    tokens_norm = []
    for w in tokens:
        w_norm = ''
        preceding_consonant = False
        for c in w:
            if not config.cased:
                c = c.lower()
            if c in CONSONANTS:
                preceding_consonant = True
            c_norm = greek_accentuation.characters.base(c)
            if config.expand_iota:
                if c in IOTA_CHARS:
                    c_norm += 'ι'
            if config.expand_rough:
                if not preceding_consonant and c in ROUGH_CHARS:
                    w_norm = 'ͱ' + w_norm
                    preceding_consonant = True
            w_norm += c_norm
        for k, v in REPLACE_DICT.items():
            w_norm = w_norm.replace(k, v)
        tokens_norm.append(w_norm)

    return tokens_norm


def featurize_tokens(tokens, tokenizer):
    tokens = normalize_tokens(tokens)

    max_word_len = config.max_word_len
    max_subword_len = config.max_subword_len

    if len(tokens) > max_word_len:
        raise RuntimeError(
            f"Number of tokens ({len(tokens)}) exceeds `max_word_len` ({max_word_len}). Increase `max_word_len` or remove sentence from dataset: {' '.join(tokens)}")

    # Encode sentence
    encoding = tokenizer.encode_plus(
        tokens, is_split_into_words=True, add_special_tokens=True, return_offsets_mapping=True, max_length=max_subword_len, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    offset_mapping = encoding['offset_mapping']

    if len(input_ids) > config.max_subword_len:
        raise RuntimeError(
            f"Number of subword tokens ({len(input_ids)}) exceeds `max_subword_len` ({max_subword_len}). Increase max_subword_len` or remove sentence from dataset: {' '.join(tokens)}")

    # Get subword strings
    subwords = tokenizer.convert_ids_to_tokens(input_ids)

    # Get word start positions
    word_start_positions, lengths = get_word_start_positions(
        tokens, subwords, offset_mapping)
    if lengths == -1:
        raise RuntimeError(
            f"Could not get `word_start_positions` and `lengths`. Sentence in question: {' '.join(tokens)}")

    # Zero-pad up to the sequence length
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_length = max_subword_len - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)

    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    word_start_positions = torch.tensor(
        word_start_positions, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return (input_ids, attention_mask, word_start_positions, lengths)


def featurize_labels(labels):
    pad_value = config.pad_value
    mode = config.mode

    if mode in ['tag', 'joint']:
        seq_len = len(labels['upos'])
        pad_len = config.max_word_len - seq_len

        upos = torch.tensor(labels['upos'], dtype=torch.long)
        xpos = torch.tensor(labels['xpos'], dtype=torch.long)
        feats = torch.tensor(labels['feats'], dtype=torch.long)
        upos = F.pad(upos, (0, pad_len), value=pad_value)
        xpos = F.pad(xpos, (0, pad_len), value=pad_value)
        feats = F.pad(feats, (0, pad_len), value=pad_value)

    if mode in ['parse', 'joint']:
        seq_len = len(labels['head'])
        pad_len = config.max_word_len - seq_len

        heads = torch.tensor(labels['head'], dtype=torch.long)
        deprels = torch.tensor(labels['deprel'], dtype=torch.long)
        heads = F.pad(heads, (0, pad_len), value=pad_value)
        deprels = F.pad(deprels, (0, pad_len), value=pad_value)

    if mode == 'tag':
        return (upos, xpos, feats)
    elif mode == 'parse':
        return (heads, deprels)
    elif mode == 'joint':
        return (upos, xpos, feats, heads, deprels)


def get_word_start_positions(tokens, subwords, offset_mapping):
    positions = []

    subtoken_mask = np.array(offset_mapping)[:,0] != 0

    for i, is_subword in enumerate(subtoken_mask[1:-1]):
        if not is_subword:
            positions.append(i+1)

    real_len = len(positions)

    if real_len != len(tokens):
        print(
            f"`real_len` {real_len} does not match number of `tokens` {len(tokens)}. Number of subword_tokens: {len(subwords)}. `max_length`: {config.max_subword_len}")
        print(positions)
        print(subwords)
        print(tokens)
        for i in range(len(positions)):
            print(positions[i], tokens[i])
        return None, -1

    positions.append(len(subwords) - 1)
    extension = [-1] * (config.max_word_len + 1 - len(positions))
    positions.extend(extension)
    return positions, real_len


def merge_subword_tokens(subword_outputs, word_starts, hidden_size):
    instances = []

    # Handling instance by instance
    for i in range(len(subword_outputs)):
        subword_vecs = subword_outputs[i]
        word_vecs = []
        starts = word_starts[i]
        for j in range(len(starts) - 1):
            k = j + 1
            if starts[k] <= 0:
                break
            elif starts[k] == starts[j]:
                while starts[k] == starts[j]:
                    k += 1

            start = starts[j]
            end = starts[k]
            vecs_range = subword_vecs[start: end]
            word_vecs.append(torch.mean(vecs_range, 0).unsqueeze(0))

        instances.append(word_vecs)

    t_insts = []

    zero_tens = torch.zeros(hidden_size).unsqueeze(0)
    zero_tens = zero_tens.to(config.device)

    for inst in instances:
        if len(inst) < config.max_word_len:
            for i in range(config.max_word_len - len(inst)):
                inst.append(zero_tens)
        t_insts.append(torch.cat(inst, dim=0).unsqueeze(0))

    w_tens = torch.cat(t_insts, dim=0)
    return w_tens


def feats_to_string(feats):
    fields = []
    if isinstance(feats, dict):
        for key, value in feats.items():
            if key in IGNORE_FEATS:
                continue
            if key == 'PUNCT':  # Specific to our dataset. Can be removed.
                return key
            if value is None:
                value = "_"
            fields.append('='.join((key, value)))
        return '|'.join(fields)
    return ''


def write_shortened_dataset(input_path, output_path, max_subword_len, tokenizer):
    too_long = []
    tokenizer.add_tokens(LANG_SPECIFIC_TOKENS)
    with open(input_path) as f:
        print(f"Parsing {input_path}")
        sents = conllu.parse(f.read(), fields=CONLLU_COLUMNS)
        print(f"Finished parsing {input_path}")
    for i, sent in enumerate(sents):
        tokens = normalize_tokens([token['form'] for token in sent])
        encoding = tokenizer.encode_plus(tokens, is_split_into_words=True, add_special_tokens=True)
        if len(encoding['input_ids']) > max_subword_len:
            print(sent)
            too_long.append(i)

    new_sents = [sent for i, sent in enumerate(sents) if i not in too_long]

    with open(output_path, 'w') as f:
        f.writelines([sent.serialize() for sent in new_sents])
    print(f"Wrote new sentences to {output_path}")
    print(f'Removed {len(too_long)} sentences')