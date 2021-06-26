# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""FocalLoss for unbalanced data."""
from vega.modules.operators import ops
from vega.modules.module import Module
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.LOSS)
class FocalLoss(Module):
    """Focal Loss for unbalanced data.

    :param alpha(1D Tensor, Variable): the scalar factor for this criterion
    :param gamma(float, double): gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
    putting more focus on hard, misclassiﬁed examples
    :param size_average(bool): By default, the losses are averaged over observations for each minibatch.
    However, if the field size_average is set to False, the losses are instead summed for each minibatch.
    """

    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = ops.ones(class_num, 1)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def call(self, inputs, targets):
        """Compute loss.

        :param inputs: predict data.
        :param targets: true data.
        :return:
        """
        N = inputs.size(0)
        C = inputs.size(1)
        P = ops.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * (ops.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
