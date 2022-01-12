# -*- coding: utf-8 -*-

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

"""Cross Entropy Loss."""
import torch
import torch.nn.functional as F
from vega.modules.module import Module
from vega.common import ClassFactory, ClassType
from .reduce_loss import weight_reduce_loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    """Cross entropy losses.

    :param pred: predict result
    :param label: gt label
    :param weight: weight
    :param reduction: reduce function
    :param avg_factor: avg factor
    :return: loss
    """
    loss = F.cross_entropy(pred, label, reduction='none')
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    """Expand binary labels.

    :param labels: labels
    :param label_weights: label weights
    :param label_channels: label channels
    :return: binary label and label weights
    """
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    """Binary cross entropy loss.

    :param pred: predict result
    :param label: gt label
    :param weight:  weight
    :param reduction: reduce function
    :param avg_factor: avg factor
    :return: loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None):
    """Mask cross entropy loss.

    :param pred: predict result
    :param target: target
    :param label: gt label
    :param reduction: reduce function
    :param avg_factor: avg factor
    :return: loss
    """
    if reduction == 'mean' and avg_factor is None:
        num_rois = pred.size()[0]
        inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
        pred_slice = pred[inds, label].squeeze(1)
        return F.binary_cross_entropy_with_logits(pred_slice, target, reduction='mean')[None]
    else:
        raise ValueError('Falied to calculate mask cross_entropy.')


@ClassFactory.register(ClassType.NETWORK)
class CustomCrossEntropyLoss(Module):
    """Cross Entropy Loss."""

    def __init__(self, desc):
        """Init Cross Entropy loss.

        :param desc: config dict
        """
        super(CustomCrossEntropyLoss, self).__init__()
        self.use_sigmoid = desc['use_sigmoid'] if 'use_sigmoid' in desc else False
        self.use_mask = desc['use_mask'] if 'use_mask' in desc else False
        self.reduction = desc['reduction'] if 'reduction' in desc else 'mean'
        self.loss_weight = desc['loss_weight'] if 'loss_weight' in desc else 1.0
        if self.use_sigmoid:
            self.loss_function = binary_cross_entropy
        elif self.use_mask:
            self.loss_function = mask_cross_entropy
        else:
            self.loss_function = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward compute.

        :param cls_score: class score
        :param label: gt labels
        :param weight: weights
        :param avg_factor: avg factor
        :param reduction_override: reduce function
        :return: loss
        """
        if reduction_override in (None, 'none', 'mean', 'sum'):
            reduction = (
                reduction_override if reduction_override else self.reduction)
            loss_cls = self.loss_weight * self.loss_function(cls_score, label, weight, reduction=reduction,
                                                             avg_factor=avg_factor, **kwargs)
            return loss_cls
        else:
            raise ValueError('Falied to calculate CustomCrossEntropyLoss.')
