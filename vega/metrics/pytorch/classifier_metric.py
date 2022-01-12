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

"""Metric of classifier task."""
from functools import partial
from vega.metrics.pytorch.metrics import MetricBase
from vega.common import ClassFactory, ClassType
import sklearn.metrics as me


def accuracy(output, target, top_k=(1,)):
    """Calculate classification accuracy between output and target.

    :param output: output of classification network
    :type output: pytorch tensor
    :param target: ground truth from dataset
    :type target: pytorch tensor
    :param top_k: top k of metric, k is an interger
    :type top_k: tuple of interger
    :return: results of top k
    :rtype: list

    """
    labels_count = output.shape[1]
    max_k = labels_count if max(top_k) > labels_count else max(top_k)
    batch_size = target.size(0)
    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res


@ClassFactory.register(ClassType.METRIC, alias='accuracy')
class Accuracy(MetricBase):
    """Calculate classification accuracy between output and target."""

    __metric_name__ = 'accuracy'

    def __init__(self, topk=(1, 5)):
        """Init Accuracy metric."""
        self.topk = topk
        self.sum = [0.] * len(topk)
        self.data_num = 0
        self.pfm = [0.] * len(topk)

    def __call__(self, output, target, *args, **kwargs):
        """Perform top k accuracy.

        :param output: output of classification network
        :param target: ground truth from dataset
        :return: pfm
        """
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(target, tuple) or isinstance(target, list):
            target = target[0]
        res = accuracy(output, target, self.topk)
        n = output.size(0)
        self.data_num += n
        self.sum = [self.sum[index] + item.item() * n for index, item in enumerate(res)]
        self.pfm = [item / self.data_num for item in self.sum]
        return res

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.sum = [0.] * len(self.topk)
        self.data_num = 0
        self.pfm = [0.] * len(self.topk)

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        if len(self.pfm) == 1:
            return self.pfm[0]
        perf_dict = {}
        perf_dict[self.name] = self.pfm[0]
        perf_dict.update({'{}_top{}'.format(self.name, self.topk[idx]): value for idx, value in enumerate(self.pfm)})
        return perf_dict


@ClassFactory.register(ClassType.METRIC)
class SklearnMetrics(MetricBase):
    """Wrapper class for Sklearn Metrics."""

    def __init__(self, name, **kwargs):
        super().__init__()
        self.__metric_name__ = name
        self.metric_func = getattr(me, name)
        if kwargs:
            self.metric_func = partial(self.metric_func, kwargs)

    def __call__(self, output, target, *args, **kwargs):
        """Perform top k accuracy.

        :param output: output of classification network
        :param target: ground truth from dataset
        :return: pfm
        """
        _, y_pred = output.topk(1, 1, True, True)
        y_pred = y_pred.t().detach().cpu().numpy()[0]
        y_true = target.detach().cpu().numpy()
        self.pfm = self.metric_func(y_true, y_pred)
        return self.pfm

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        pass

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return self.pfm
