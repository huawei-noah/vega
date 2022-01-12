# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bert network."""
from vega.modules.module import Module
from vega.common.class_factory import ClassType, ClassFactory
from vega.modules.operators import ops
from transformers import BertForSequenceClassification, PretrainedConfig, BertConfig, BertModel


@ClassFactory.register(ClassType.NETWORK)
class TransformersPretrainedModule(Module):
    """Base Class to handle classes from transformers."""

    def __init__(self, config):
        super(TransformersPretrainedModule, self).__init__()
        self.config = BertConfig(**config)

    def init_bert_weights(self, module):
        """Initialize the weights."""
        if module.__class__.__name__ in ('Linear', 'Embedding'):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif module.__class__.__name__ == 'BertLayerNorm':
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if module.__class__.__name__ == 'Linear' and module.bias is not None:
            module.bias.data.zero_()

    def load_state_dict(self, state_dict=None, strict=True):
        """Load and convert state dict."""
        state_dict = {k.replace('.gamma', '.weight').replace('.beta', '.bias'): v for k, v in state_dict.items()}
        super().load_state_dict(state_dict, strict)


@ClassFactory.register(ClassType.NETWORK)
class BertClassification(TransformersPretrainedModule):
    """Bert for Classification task."""

    def __init__(self, config):
        super(BertClassification, self).__init__(config)
        self.bert = BertModel(self.config)
        self.dropout = ops.Dropout(self.config.hidden_dropout_prob)
        self.classifier = ops.Linear(self.config.hidden_size, self.config.num_labels)

    def call(self, input_ids, **inputs):
        """Call model."""
        pooled_output = self.bert(input_ids, **inputs)
        if isinstance(pooled_output, dict):
            pooled_output = pooled_output['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


@ClassFactory.register(ClassType.NETWORK)
class TinyBertForPreTraining(TransformersPretrainedModule):
    """Tiny Bert Model."""

    def __init__(self, config, fit_size=768):
        super(TinyBertForPreTraining, self).__init__(config)
        self.bert = BertModel(self.config)
        self.apply(self.init_bert_weights)
        self.fit_dense = ops.Linear(self.config.hidden_size, fit_size)

    def call(self, input_ids, token_type_ids=None, attention_mask=None, is_student=False):
        """Call model."""
        outs = self.bert(input_ids, token_type_ids, attention_mask, output_hidden_states=True)
        sequence_output, pooled_output = outs.hidden_states, outs.pooler_output
        if not is_student:
            return pooled_output, sequence_output
        tmp = []
        for s_id, sequence_layer in enumerate(sequence_output):
            tmp.append(self.fit_dense(sequence_layer))
        sequence_output = tmp
        return pooled_output, sequence_output


@ClassFactory.register(ClassType.NETWORK)
class BertClassificationHeader(Module):
    """Header for classification task."""

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0.1):
        super(BertClassificationHeader, self).__init__()
        self.dropout = ops.Dropout(hidden_dropout_prob)
        self.classifier = ops.Linear(hidden_size, num_labels)

    def call(self, pooled_output):
        """Call model."""
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
