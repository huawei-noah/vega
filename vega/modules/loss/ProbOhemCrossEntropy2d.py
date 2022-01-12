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

"""ProbOhemCrossEntropy2d loss function."""
import torch
import torch.nn.functional as F
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module


@ClassFactory.register(ClassType.LOSS)
class ProbOhemCrossEntropy2d(Module):
    """ProbOhemCrossEntropy2d loss class."""

    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=None,
                 down_ratio=1, use_weight=False, aux_weight=None, **kwargs):
        """Construct the ProbOhemCrossEntropy2d class.

        :param ignore_label: label to be ignored when computing loss.
        :param reduction: the reduction of CrossEntropyLoss.
        :param thresh: threshold to judge to kept the mask.
        :param min_kept: minimum of pixels left.
        :param use_weight: weather to use weight corresponding to the class.
        :param aux_weight: weight of auxiliary output and main output.
        :param **kwargs: to get the image_size and batch_size parameters.
        """
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.aux_weight = aux_weight
        if min_kept is None:
            image_height, image_width = kwargs['image_size']
            min_kept = int(kwargs['batch_size'] * image_height * image_width // 16)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, results, target):
        """Compute loss.

        :param results: the results of the inference.
        :param target: ground truth.
        :return: loss
        """
        aux_weight = None
        aux_weight = self.aux_weight
        loss = []
        saved_target = target
        if aux_weight is None:
            aux_weight = [1.0] * len(results)
        for i, pred in enumerate(results):
            b, c, h, w = pred.size()
            target = saved_target
            target = target.view(-1)
            valid_mask = target.ne(self.ignore_label)
            target = target * valid_mask.long()
            num_valid = valid_mask.sum()
            prob = F.softmax(pred, dim=1)
            prob = (prob.transpose(0, 1)).reshape(c, -1)
            if self.min_kept > num_valid:
                print('Labels: {}'.format(num_valid))
            elif num_valid > 0:
                prob = prob.masked_fill_(~valid_mask, 1)
                mask_prob = prob[
                    target, torch.arange(len(target), dtype=torch.long)]
                threshold = self.thresh
                if self.min_kept > 0:
                    index = mask_prob.argsort()
                    threshold_index = index[min(len(index), self.min_kept) - 1]
                    if mask_prob[threshold_index] > self.thresh:
                        threshold = mask_prob[threshold_index]
                    kept_mask = mask_prob.le(threshold)
                    target = target * kept_mask.long()
                    valid_mask = valid_mask * kept_mask
            target = target.masked_fill_(~valid_mask, self.ignore_label)
            target = target.view(b, h, w)
            loss.append(aux_weight[i] * self.criterion(pred, target))
        return sum(loss)
