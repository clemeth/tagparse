from functools import partial
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertConfig
from tqdm import tqdm

from utils import CoNLLDataset, collate_fn, add_language_specific_tokens
from models import BertForTagging, BertForParsing, BertForJointTaggingAndParsing

import config
import os
import torch

# Print config parameters
for param in dir(config):
    if not param.startswith('__'):
        param_val = getattr(config, param)
        if isinstance(param_val, str):
            param_val = f"'{param_val}'" 
        print(f'{param} = {param_val}')
print()

# Load tokenizer, model and config
tokenizer = BertTokenizerFast.from_pretrained(config.model_name)

model_path = f'{config.models_path}{config.name}/'
model_config = BertConfig.from_pretrained(
    model_path + 'config.json', local_files_only=True)

mode = config.mode
if mode == 'tag':
    model = BertForTagging.from_pretrained(model_path, config=model_config)
elif mode == 'parse':
    model = BertForParsing.from_pretrained(model_path, config=model_config)
elif mode == 'joint':
    model = BertForJointTaggingAndParsing.from_pretrained(
        model_path, config=model_config)

# Add language specific tokens
add_language_specific_tokens(model, tokenizer)

# Send model to device; loader will send data automatically
model.to(config.device)

# Load data and loader
collate_fn = partial(collate_fn, tokenizer=tokenizer)
test = CoNLLDataset(config.test_path)
test_loader = DataLoader(test, batch_size=config.batch_size,
                         collate_fn=collate_fn, shuffle=False)

# Predictions
upos = []
xpos = []
feats = []
heads = []
deprels = []

# Test loop
for i, (X, Y) in enumerate(tqdm(test_loader, desc=f"Testing (batch_size={config.batch_size}, test_size={len(test_loader.dataset)})")):
    with torch.no_grad():
        outputs = iter(model(X))
        loss = next(outputs)

        if mode in ['tag', 'joint']:
            upos_scores = next(outputs)
            xpos_scores = next(outputs)
            feats_scores = next(outputs)

            upos_preds = upos_scores.argmax(-1)
            xpos_preds = xpos_scores.argmax(-1)
            feats_preds = feats_scores.argmax(-1)

        if mode in ['parse', 'joint']:
            head_scores = next(outputs)
            deprel_scores = next(outputs)

            head_preds = head_scores.argmax(-1)
            if len(head_preds.shape) == 1:
                head_preds = head_preds.unsqueeze(0)

            deprel_preds = deprel_scores.argmax(-1)
            deprel_preds = deprel_preds.gather(-1,
                                               head_preds.unsqueeze(-1)).squeeze(-1)

    lengths = X['lengths']

    if mode in ['tag', 'joint']:
        for j in range(len(upos_preds)):
            upos.append(upos_preds[j][:lengths[j]])
            xpos.append(xpos_preds[j][:lengths[j]])
            feats.append(feats_preds[j][:lengths[j]])

    if mode in ['parse', 'joint']:
        for j in range(len(deprel_preds)):
            heads.append(head_preds[j][:lengths[j]])
            deprels.append(deprel_preds[j][:lengths[j]])

# Evaluate and save predictions
total_tokens = 0
correct_upos = 0
correct_xpos = 0
correct_feats = 0
correct_tag = 0
correct_heads = 0
correct_deprels = 0
correct_heads_and_deprels = 0

preds = []

DEPRELS_IGNORE = [
    'PUNCT',
    'punct',
    'GAP'
]

UPOS_IGNORE = [
    'PUNCT',
    'GAP'
]

for i in range(len(test_loader.dataset)):
    pred = []
    for j in range(len(test_loader.dataset[i][0])):
        idx = j + 1
        token = test_loader.dataset[i][0][j]

        if mode in ['tag', 'joint']:
            upos_pred = test_loader.dataset.idx2label['upos'][upos[i][j].item(
            )]
            xpos_pred = test_loader.dataset.idx2label['xpos'][xpos[i][j].item(
            )]
            feats_pred = test_loader.dataset.idx2label['feats'][feats[i][j].item(
            )]

            upos_gold = test_loader.dataset[i][1]['upos'][j]
            upos_gold = test_loader.dataset.idx2label['upos'][upos_gold]

            xpos_gold = test_loader.dataset[i][1]['xpos'][j]
            xpos_gold = test_loader.dataset.idx2label['xpos'][xpos_gold]

            feats_gold = test_loader.dataset[i][1]['feats'][j]
            feats_gold = test_loader.dataset.idx2label['feats'][feats_gold]

            wrong_tag = False

            if upos_pred != upos_gold or xpos_pred != xpos_gold or feats_pred != feats_gold:
                wrong_tag = True

        if mode in ['parse', 'joint']:
            head_pred = heads[i][j].item()
            deprel_pred = test_loader.dataset.idx2label['deprel'][deprels[i][j].item(
            )]

            head_gold = test_loader.dataset[i][1]['head'][j]
            deprel_gold = test_loader.dataset[i][1]['deprel'][j]
            deprel_gold = test_loader.dataset.idx2label['deprel'][deprel_gold]

            wrong_parse = False

            if head_pred != head_gold or deprel_pred != deprel_gold:
                wrong_parse = True

        if mode in ['tag', 'joint']:
            if not config.ignore_punct or upos_gold not in UPOS_IGNORE:
                if upos_pred == upos_gold:
                    correct_upos += 1

                if xpos_pred == xpos_gold:
                    correct_xpos += 1

                if feats_pred == feats_gold:
                    correct_feats += 1

                if not wrong_tag:
                    correct_tag += 1

                if mode == 'tag':
                    total_tokens += 1

        if mode in ['parse', 'joint']:
            if not config.ignore_punct or deprel_gold not in DEPRELS_IGNORE:
                correct_head = False
                if head_pred == head_gold:
                    correct_head = True
                    correct_heads += 1
                if deprel_pred == deprel_gold:
                    if correct_head:
                        correct_heads_and_deprels += 1
                    correct_deprels += 1

                total_tokens += 1

        if mode == 'tag':
            pred.append((idx, token, upos_pred, xpos_pred, feats_pred))
            if wrong_tag and config.print_gold:
                pred.append(('Gold:', upos_gold, xpos_gold,
                            feats_gold))
        elif mode == 'parse':
            pred.append((idx, token, head_pred, deprel_pred))
            if wrong_parse and config.print_gold:
                pred.append(('Gold:', head_gold, deprel_gold))
        elif mode == 'joint':
            pred.append((idx, token, upos_pred, xpos_pred,
                        feats_pred, head_pred, deprel_pred))
            if (wrong_tag or wrong_parse) and config.print_gold:
                pred.append(('Gold:', upos_gold, xpos_gold,
                            feats_gold, head_gold, deprel_gold))

    preds.append(pred)

upos_acc = correct_upos/total_tokens
xpos_acc = correct_xpos/total_tokens
feats_acc = correct_feats/total_tokens
tag_acc = correct_tag/total_tokens
uas = correct_heads/total_tokens
las = correct_heads_and_deprels/total_tokens
la = correct_deprels/total_tokens

print(f"UPOS: {upos_acc}")
print(f"XPOS: {xpos_acc}")
print(f"UFeats: {feats_acc}")
print(f"UPOS & XPOS & Ufeats: {tag_acc}")
print(f"UAS: {uas}")
print(f"LAS: {las}")
print(f"LA: {la}")

if not os.path.exists('preds'):
    os.makedirs('preds')

with open(f'preds/{config.name}.txt', 'w+') as f:
    # Write config parameters
    for param in dir(config):
        if not param.startswith('__'):
            param_val = getattr(config, param)
            if isinstance(param_val, str):
                param_val = f"'{param_val}'" 
            f.write(f'{param} = {param_val}\n')

    f.write('\n')
    for pred in preds:
        for token in pred:
            f.write(str(token) + '\n')
        f.write('\n')
