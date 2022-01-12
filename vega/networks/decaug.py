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

"""DecAug network"""
from vega.common import ClassFactory, ClassType
from vega.modules.operators import ops
from vega.modules.module import Module


@ClassFactory.register(ClassType.NETWORK)
class DecAug(Module):
    """Create DecAug Network."""

    def __init__(self, hdim, num_classes=7, num_concept=3, **kwargs):
        super(DecAug, self).__init__()
        self.category_branch = ops.Linear(hdim, hdim)
        self.concept_branch = ops.Linear(hdim, hdim)
        self.relu = ops.Relu(inplace=True)
        self.fc0 = ops.Linear(hdim, num_classes)
        self.fcc0 = ops.Linear(hdim, num_concept)
        self.classification = ops.Linear(hdim * 2, num_classes)

    def forward(self, x):
        B, _ = x.shape
        instance_embs = x #torch.reshape(x, (B, -1))

        category_embs = self.category_branch(instance_embs)
        logits_category = self.fc0(category_embs)
        logits_category = ops.softmax(logits_category, dim=1)

        concept_embs = self.concept_branch(instance_embs)
        logits_concept = self.fcc0(concept_embs)
        logits_concept = ops.softmax(logits_concept, dim=1)
        # concept branch
        output = None
        if not self.training:
            embs = ops.concat((category_embs, concept_embs), 1)
            output = self.classification(embs)
        return output, logits_category, logits_concept, category_embs, concept_embs, self
