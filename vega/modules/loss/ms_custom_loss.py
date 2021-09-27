# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""CustomSoftmaxCrossEntropyWithLogits."""

from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LOSS)
class CustomSoftmaxCrossEntropyWithLogits(SoftmaxCrossEntropyWithLogits):
    """CustomSoftmaxCrossEntropyWithLogits loss class."""

    def __init__(self, sparse=True, reduction='none'):
        super(CustomSoftmaxCrossEntropyWithLogits, self).__init__(sparse=True)
        self.reshape = P.Reshape()
        self.squeeze = P.Squeeze(1)

    def construct(self, logits, labels):
        """Forward of CustomSoftmaxCrossEntropyWithLogits."""
        logits = self.reshape(logits, (-1, F.shape(logits)[-1]))
        labels = self.reshape(labels, (-1, 1))
        labels = self.squeeze(labels)
        if self.sparse:
            if self.reduction == 'mean':
                x = self.sparse_softmax_cross_entropy(logits, labels)
                return x
            labels = self.one_hot(labels, F.shape(logits)[-1], self.on_value, self.off_value)
        x = self.softmax_cross_entropy(logits, labels)[0]
        return self.get_loss(x)
