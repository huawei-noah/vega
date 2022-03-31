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

"""DecAug loss"""

import random
import torch
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module
from vega.modules.operators import ops


@ClassFactory.register(ClassType.LOSS)
class DecAugLoss(Module):

    def __init__(self, balance1=0.01, balance2=0.01, balanceorth=0.01, epsilon=1e-8, perturbation=1.0):
        super(DecAugLoss, self).__init__()
        self.balance1 = balance1
        self.balance2 = balance2
        self.balanceorth = balanceorth
        self.epsilon = epsilon
        self.perturbation = perturbation
        self.cross_entropy = ClassFactory.get_cls(ClassType.LOSS, "CrossEntropyLoss")()

    def forward(self, x, targets=None):
        _, logits_category, logits_concept, feature_category, feature_concept, model = x
        gt_label, gt_concept = targets
        loss1 = self.cross_entropy(logits_category, gt_label)
        loss2 = self.cross_entropy(logits_concept, gt_concept)
        parm = {}
        for name, parameters in model.named_parameters():
            parm[name] = parameters
        # concept branch
        w_branch = parm['concept_branch.weight']
        w_tensor = parm['fcc0.weight']
        # classification branch
        w_branch_l = parm['category_branch.weight']
        w_tensor_l = parm['fc0.weight']

        w_out = parm['classification.weight']
        b_out = parm['classification.bias']

        w = ops.matmul(w_tensor, w_branch)
        grad = -1 * w[gt_concept] + ops.matmul(logits_concept.detach(), w)
        grad_norm = grad / (grad.norm(2, dim=1, keepdim=True) + self.epsilon)

        w_l = ops.matmul(w_tensor_l, w_branch_l)
        grad_l = -1 * w_l[gt_label] + ops.matmul(logits_category.detach(), w_l)
        grad_norm_l = grad_l / (grad_l.norm(2, dim=1, keepdim=True) + self.epsilon)
        b, L = grad_norm_l.shape

        grad_norm = grad_norm.reshape(b, 1, L)
        grad_norm_l = grad_norm_l.reshape(b, L, 1)
        loss_orth = ((torch.bmm(grad_norm, grad_norm_l).cuda()) ** 2).sum()

        grad_aug = -1 * w_tensor[gt_concept] + ops.matmul(logits_concept.detach(), w_tensor)
        FGSM_attack =  self.perturbation * (grad_aug.detach() / (grad_aug.detach().norm(2, dim=1, keepdim=True) + self.epsilon))

        ratio = random.random()
        feature_aug = ratio * FGSM_attack
        embs = ops.concat((feature_category, feature_concept + feature_aug), 1)
        output = ops.matmul(embs, w_out.transpose(0, 1)) + b_out

        loss_class = self.cross_entropy(output, gt_label)
        loss = loss_class + self.balance1 * loss1 + self.balance2 * loss2 + self.balanceorth * loss_orth
        return loss
