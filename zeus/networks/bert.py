# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Bert network."""
from zeus.modules.module import Module
from zeus.modules.operators import ops
from zeus.common.class_factory import ClassType, ClassFactory
from zeus.modules.tensformers import BertEncoder, BertEmbeddings, Pooler


@ClassFactory.register(ClassType.NETWORK)
class BertClassifier(Module):
    """Bert Classifier."""

    def __init__(self, config, num_labels=2):
        super(BertClassifier, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = ops.Dropout(config.hidden_dropout_prob)
        self.classifier = ops.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_bert_weights)

    @property
    def pretrained_hook(self):
        """Get Pertrained Hook."""
        self.strict = False
        return 'pretrained_bert_classifier_hook'

    def call(self, inputs, target=None):
        """Call BertClassifier."""
        input_ids, token_type_ids, attention_mask = inputs.get("input_ids")[0], inputs.get(
            "token_type_ids"), inputs.get("attention_mask")
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


@ClassFactory.register(ClassType.NETWORK)
class BertModel(Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer")."""

    def __init__(self, config):
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = Pooler(config)
        # self.apply(self.init_bert_weights)

    def call(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        """Call Bert Model."""
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
