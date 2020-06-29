# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Base class for metrics. All metric class should be implement this base class."""
from functools import partial
from inspect import isfunction
from copy import deepcopy
import importlib
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import Config


class MetricBase(object):
    """Provide base metrics class for all custom metric to implement."""

    def __call__(self, output, target, *args, **kwargs):
        """Perform top k accuracy. called in train and valid step.

        :param output: output of classification network
        :param target: ground truth from dataset
        """
        raise NotImplementedError

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        raise NotImplementedError

    def summary(self):
        """Summary all cached records, called after valid."""
        raise NotImplementedError


class Metrics(object):
    """Metrics class of all metrics defined in cfg.

    :param metric_cfg: metric part of config
    :type metric_cfg: dict or Config
    """

    __supported_call__ = ['accuracy', 'DetMetric', 'IoUMetric', 'SRMetric',
                          'JDDTrainerPSNRMetric']

    def __init__(self, metric_cfg):
        """Init Metrics."""
        metric_config = deepcopy(metric_cfg)
        self.mdict = {}
        if not isinstance(metric_config, list):
            metric_config = [metric_config]
        for metric_item in metric_config:
            metric_name = metric_item.pop('type')
            if ClassFactory.is_exists(ClassType.METRIC, metric_name):
                metric_class = ClassFactory.get_cls(
                    ClassType.METRIC, metric_name)
            else:
                metric_class = getattr(importlib.import_module(
                    'vega.core.metrics'), metric_name)
            if isfunction(metric_class):
                metric_class = partial(metric_class, **metric_item)
            else:
                metric_class = metric_class(**metric_item)
            self.mdict[metric_name] = metric_class
        self.mdict = Config(self.mdict)

    def __call__(self, output=None, target=None, *args, **kwargs):
        """Calculate all supported metrics by using output and target.

        :param output: predicted output by networks
        :type output: torch tensor
        :param target: target label data
        :type target: torch tensor
        :return: performance of metrics
        :rtype: list
        """
        pfms = []
        for key in self.mdict:
            metric = self.mdict[key]
            if key in self.__supported_call__:
                pfms.append(metric(output, target, *args, **kwargs))
        return pfms
        # if len(pfms) == 1:
        #     return pfms[0]
        # else:
        #     return pfms

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        for val in self.mdict.values():
            val.reset()

    @property
    def names(self):
        """Return metrics names."""
        names = [name for name in self.mdict if name in self.__supported_call__]
        return names
        # if len(names) == 1:
        #     return names[0]
        # else:
        #     return names

    @property
    def results(self):
        """Return metrics results."""
        results = [self.mdict[name].summary()
                   for name in self.mdict if name in self.__supported_call__]
        return deepcopy(results)

    @property
    def results_dict(self):
        """Return metrics results dict."""
        rdict = {}
        for key in self.mdict:
            rdict[key] = self.mdict[key].summary()
        return deepcopy(rdict)

    def __getattr__(self, key):
        """Get a metric by key name.

        :param key: metric name
        :type key: str
        """
        return self.mdict[key]
