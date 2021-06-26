# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of nlp task."""
import sklearn.metrics as me

from vega.common import ClassFactory, ClassType
from vega.metrics.pytorch.metrics import MetricBase


@ClassFactory.register(ClassType.METRIC)
class NlpMetrics(MetricBase):
    """Wrapper class for nlp Metrics."""

    def __init__(self, names=None, **kwargs):
        super().__init__()
        self.y_true_list = []
        self.y_pred_list = []
        if names is None:
            names = ['accuracy_score', 'f1_score']
        else:
            names = names if isinstance(names, list) else [names]
        self.metric_funcs = {name: getattr(me, name) for name in names}

    def __call__(self, output, target, *args, **kwargs):
        """Perform top k accuracy.

        :param output: output of classification network
        :param target: ground truth from dataset
        :return: pfm
        """
        if isinstance(output, dict):
            output = output['logits']
        _, y_pred = output.topk(1, 1, True, True)
        y_pred = y_pred.T.tolist()[0]
        if isinstance(y_pred, list):
            self.y_pred_list.extend(y_pred)
        else:
            self.y_pred_list.append(y_pred)
        y_true = target.tolist()
        if isinstance(y_true, list):
            self.y_true_list.extend(y_true)
        else:
            self.y_true_list.append(y_true)
        return None

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.y_true_list = []
        self.y_pred_list = []

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return {name: fn(self.y_true_list, self.y_pred_list) for name, fn in self.metric_funcs.items()}


@ClassFactory.register(ClassType.METRIC, alias='accuracy_score')
class AccuracyScore(NlpMetrics):
    """AccuracyScore."""

    def __init__(self):
        super(AccuracyScore, self).__init__('accuracy_score')


@ClassFactory.register(ClassType.METRIC, alias='f1_score')
class F1Score(NlpMetrics):
    """AccuracyScore."""

    def __init__(self):
        super(F1Score, self).__init__('f1_score')
