from functools import partial
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizerFast, BertConfig, AdamW, get_scheduler
from tqdm import tqdm

from utils import CoNLLDataset, collate_fn, set_seed, add_language_specific_tokens
from models import BertForTagging, BertForParsing, BertForJointTaggingAndParsing

import torch
import config
import math
import os

# Set random seeds for reproducibility
set_seed(config.seed)

# Load tokenizer, model and config
tokenizer = BertTokenizerFast.from_pretrained(config.model_name)

model_config = BertConfig.from_pretrained(config.model_name)

mode = config.mode
if mode == 'tag':
    model = BertForTagging.from_pretrained(
        config.model_name, config=model_config)
elif mode == 'parse':
    model = BertForParsing.from_pretrained(
        config.model_name, config=model_config)
elif mode == 'joint':
    model = BertForJointTaggingAndParsing.from_pretrained(
        config.model_name, config=model_config)
    
# Add language specific tokens
add_language_specific_tokens(model, tokenizer)

# Send model to device; loader will send data automatically
model.to(config.device)

# Load data
train = CoNLLDataset(config.train_path)
val = CoNLLDataset(config.val_path)

# Load loaders
collate_fn = partial(collate_fn, tokenizer=tokenizer)
train_loader = DataLoader(
    train, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val, batch_size=config.batch_size,
                        collate_fn=collate_fn, shuffle=False)

# Init optimizer
bert_params = model.bert.parameters()
bert_param_names = [f'bert.{k}' for (k, v) in model.bert.named_parameters()]
head_params = [v for (k, v) in model.named_parameters()
               if k not in bert_param_names]

if mode in ['tag', 'parse']:
    optimizer = AdamW([{'params': bert_params, 'lr': config.bert_lr},
                       {'params': head_params, 'lr': config.classifier_lr}])
elif mode == 'joint':
    tagger_param_names = ['classifier_upos.weight', 'classifier_upos.bias', 'classifier_xpos.weight',
                          'classifier_xpos.bias', 'classifier_feats.weight', 'classifier_feats.bias']
    parser_param_names = ['classifier_head.weight', 'classifier_deprel.weight']
    tagger_params = [v for (k, v) in model.named_parameters()
                     if k in tagger_param_names]
    parser_params = [v for (k, v) in model.named_parameters()
                     if k in parser_param_names]
    optimizer = AdamW([{'params': bert_params, 'lr': config.bert_lr},
                       {'params': tagger_params, 'lr': config.tagger_lr},
                       {'params': parser_params, 'lr': config.parser_lr}])

# Init scheduler
lr_scheduler = get_scheduler(config.scheduler, optimizer=optimizer, num_warmup_steps=config.num_warmup_steps, num_training_steps=config.epochs * len(train_loader))

# Total stats
writer = SummaryWriter(f'runs/{config.name}/')
finished_epochs = 0
best_epoch = -1
epochs_without_improvement = 0
best_avg_val_loss = math.inf

# Print config parameters
for param in dir(config):
    if not param.startswith('__'):
        param_val = getattr(config, param)
        if isinstance(param_val, str):
            param_val = f"'{param_val}'" 
        print(f'{param} = {param_val}')
print()

# Epoch loop
for epoch in range(1, config.epochs+1):
    print(f"Running epoch {epoch}/{config.epochs}")

    # Epoch stats
    train_loss_sum = 0.0
    val_loss_sum = 0.0
    num_train_batches = 0
    num_val_batches = 0

    # Training loop
    model.train()
    for i, (X, Y) in enumerate(tqdm(train_loader, desc=f"Training (batch_size={config.batch_size}, train_size={len(train_loader.dataset)})")):
        train_loss = model(X, Y)[0]
        train_loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_loss_sum += train_loss.item()
        num_train_batches += 1

    # Training stats
    avg_train_loss = train_loss_sum/num_train_batches
    print(f"Average train loss: {avg_train_loss}")
    writer.add_scalar('Loss/train', avg_train_loss, epoch)

    # Validation loop
    model.eval()
    for i, (X, Y) in enumerate(tqdm(val_loader, desc=f"Evaluating (batch_size={config.batch_size}, val_size={len(val_loader.dataset)})")):
        with torch.no_grad():
            val_loss = model(X, Y)[0]

        val_loss_sum += val_loss.item()
        num_val_batches += 1

    # Validation stats
    avg_val_loss = val_loss_sum/num_val_batches
    print(f"Average val loss: {avg_val_loss}")
    writer.add_scalar('Loss/val', avg_val_loss, epoch)

    # Save model if val loss is best yet
    if avg_val_loss < best_avg_val_loss:
        epochs_without_improvement = 0
        best_avg_val_loss = avg_val_loss
        best_epoch = epoch
        model_path = f'{config.models_path}{config.name}/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_pretrained(model_path)
        print(f"Saved model to {model_path}")
    else:
        epochs_without_improvement += 1

    finished_epochs += 1
    print(f"Finished epoch {epoch}\n")

    if epochs_without_improvement >= config.early_stop:
        print(
            f"{epochs_without_improvement} epochs without improvement. Early stopping.")
        break

print(f"Finished training {finished_epochs} epochs. Best epoch: {best_epoch}. val loss: {best_avg_val_loss}")
