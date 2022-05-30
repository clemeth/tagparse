from torch import nn
from transformers import BertPreTrainedModel, BertModel

from utils import load_vocabs, merge_subword_tokens

import config
import torch


class BertForTagging(BertPreTrainedModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(config.last_layer_dropout)

        self.hidden_size = model_config.hidden_size
        self.vocabs = load_vocabs(config.vocabs_path)
        self.criterion = nn.CrossEntropyLoss()

        self.num_upos = len(self.vocabs['upos'])
        self.num_xpos = len(self.vocabs['xpos'])
        self.num_feats = len(self.vocabs['feats'])

        self.classifier_upos = nn.Linear(self.hidden_size, self.num_upos)
        self.classifier_xpos = nn.Linear(self.hidden_size, self.num_xpos)
        self.classifier_feats = nn.Linear(self.hidden_size, self.num_feats)

    def forward(self, batch=None, labels=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        word_start_positions = batch['word_start_positions']

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask)
        sequence_output = outputs[0]  # Last hidden state
        sequence_output = self.dropout(sequence_output)

        word_outputs = merge_subword_tokens(
            sequence_output, word_start_positions, self.hidden_size)

        upos_scores = self.classifier_upos(word_outputs)
        xpos_scores = self.classifier_xpos(word_outputs)
        feats_scores = self.classifier_feats(word_outputs)

        loss = None
        if labels is not None:
            loss = self.get_loss(upos_scores, xpos_scores,
                                 feats_scores, labels)

        return (loss, upos_scores, xpos_scores, feats_scores)

    def get_loss(self, upos_scores, xpos_scores, feats_scores, labels):
        mask = labels['upos'].ne(config.pad_value)

        upos_scores, upos_labels = upos_scores[mask], labels['upos'][mask]
        xpos_scores, xpos_labels = xpos_scores[mask], labels['xpos'][mask]
        feats_scores, feats_labels = feats_scores[mask], labels['feats'][mask]

        upos_loss = self.criterion(upos_scores, upos_labels)
        xpos_loss = self.criterion(xpos_scores, xpos_labels)
        feats_loss = self.criterion(feats_scores, feats_labels)

        return upos_loss + xpos_loss + feats_loss


class BertForParsing(BertPreTrainedModel):
    def __init__(self, model_config, joint=False):
        super().__init__(model_config)

        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(config.last_layer_dropout)

        self.hidden_size = model_config.hidden_size
        self.vocabs = load_vocabs(config.vocabs_path)
        self.criterion = nn.CrossEntropyLoss()

        self.num_deprel = len(self.vocabs['deprel'])

        self.classifier_head = Biaffine(
            n_in=self.hidden_size, n_out=1, bias_x=True, bias_y=False)
        self.classifier_deprel = Biaffine(
            n_in=self.hidden_size, n_out=self.num_deprel, bias_x=True, bias_y=True)

    def forward(self, batch, labels=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        word_start_positions = batch['word_start_positions']

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask)
        sequence_output = outputs[0]  # Last hidden state
        sequence_output = self.dropout(sequence_output)

        word_outputs = merge_subword_tokens(
            sequence_output, word_start_positions, self.hidden_size)
        word_outputs_heads = torch.cat(
            [outputs[1].unsqueeze(1), word_outputs], dim=1)

        head_scores = self.classifier_head(word_outputs, word_outputs_heads)
        head_scores = head_scores.squeeze()

        deprel_scores = self.classifier_deprel(
            word_outputs, word_outputs_heads)
        deprel_scores = deprel_scores.permute(0, 2, 3, 1)

        loss = None
        if labels is not None:
            loss = self.get_loss(head_scores, deprel_scores, labels)

        return (loss, head_scores, deprel_scores)

    def get_loss(self, head_scores, deprel_scores, labels):
        if len(head_scores.shape) == 2:
            head_scores = head_scores.unsqueeze(0)

        mask = labels['head'].ne(config.pad_value)

        head_scores, head_labels = head_scores[mask], labels['head'][mask]
        deprel_scores, deprel_labels = deprel_scores[mask], labels['deprel'][mask]
        deprel_scores = deprel_scores[torch.arange(len(head_labels)), head_labels]

        head_loss = self.criterion(head_scores, head_labels)
        deprel_loss = self.criterion(deprel_scores, deprel_labels)

        return head_loss + deprel_loss


class BertForJointTaggingAndParsing(BertPreTrainedModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(config.last_layer_dropout)

        self.hidden_size = model_config.hidden_size
        self.vocabs = load_vocabs(config.vocabs_path)
        self.criterion = nn.CrossEntropyLoss()

        self.num_upos = len(self.vocabs['upos'])
        self.num_xpos = len(self.vocabs['xpos'])
        self.num_feats = len(self.vocabs['feats'])
        self.num_deprel = len(self.vocabs['deprel'])

        self.classifier_upos = nn.Linear(self.hidden_size, self.num_upos)
        self.classifier_xpos = nn.Linear(self.hidden_size, self.num_xpos)
        self.classifier_feats = nn.Linear(self.hidden_size, self.num_feats)
        self.classifier_head = Biaffine(
            n_in=self.hidden_size, n_out=1, bias_x=True, bias_y=False)
        self.classifier_deprel = Biaffine(
            n_in=self.hidden_size, n_out=self.num_deprel, bias_x=True, bias_y=True)

    def forward(self, batch, labels=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        word_start_positions = batch['word_start_positions']

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # Last hidden state
        sequence_output = self.dropout(sequence_output)

        word_outputs = merge_subword_tokens(
            sequence_output, word_start_positions, self.hidden_size)
        word_outputs_heads = torch.cat(
            [outputs[1].unsqueeze(1), word_outputs], dim=1)

        upos_scores = self.classifier_upos(word_outputs)
        xpos_scores = self.classifier_xpos(word_outputs)
        feats_scores = self.classifier_feats(word_outputs)

        head_scores = self.classifier_head(word_outputs, word_outputs_heads)
        head_scores = head_scores.squeeze()
        deprel_scores = self.classifier_deprel(
            word_outputs, word_outputs_heads)
        deprel_scores = deprel_scores.permute(0, 2, 3, 1)

        loss = None
        if labels is not None:
            loss = self.get_loss(upos_scores, xpos_scores,
                                 feats_scores, head_scores, deprel_scores, labels)

        return (loss, upos_scores, xpos_scores, feats_scores, head_scores, deprel_scores)

    def get_loss(self, upos_scores, xpos_scores, feats_scores, head_scores, deprel_scores, labels):
        if len(head_scores.shape) == 2:
            head_scores = head_scores.unsqueeze(0)

        mask = labels['upos'].ne(config.pad_value)

        upos_scores, upos_labels = upos_scores[mask], labels['upos'][mask]
        xpos_scores, xpos_labels = xpos_scores[mask], labels['xpos'][mask]
        feats_scores, feats_labels = feats_scores[mask], labels['feats'][mask]

        upos_loss = self.criterion(upos_scores, upos_labels)
        xpos_loss = self.criterion(xpos_scores, xpos_labels)
        feats_loss = self.criterion(feats_scores, feats_labels)

        head_scores, head_labels = head_scores[mask], labels['head'][mask]
        deprel_scores, deprel_labels = deprel_scores[mask], labels['deprel'][mask]
        deprel_scores = deprel_scores[torch.arange(len(head_labels)), head_labels]

        head_loss = self.criterion(head_scores, head_labels)
        deprel_loss = self.criterion(deprel_scores, deprel_labels)

        return upos_loss + xpos_loss + feats_loss + head_loss + deprel_loss


class Biaffine(nn.Module):
    # Taken from TowerParse (Glavaš and Vulić 2021a).
    # https://github.com/codogogo/towerparse/blob/b55b57f2c9b8f71f7bf61a4d4b6110466b58ee68/biaffine.py#L72
    # Original credit: Class taken from https://github.com/yzhangcs/biaffine-parser

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.init_weights()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def init_weights(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        return s
