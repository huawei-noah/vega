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
from zeus import metrics as metrics
from zeus.common import Config
from zeus.common import ClassFactory, ClassType
from zeus.trainer.conf import MetricsConfig


class MetricBase(object):
    """Provide base metrics class for all custom metric to implement."""

    __metric_name__ = None

    def __call__(self, output, target, *args, **kwargs):
        """Perform metric. called in train and valid step.

        :param output: output of network
        :param target: ground truth from dataset
        """
        raise NotImplementedError

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        raise NotImplementedError

    def summary(self):
        """Summary all cached records, called after valid."""
        raise NotImplementedError

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return 'MAX'

    @property
    def name(self):
        """Get metric name."""
        return self.__metric_name__ or self.__class__.__name__

    @property
    def result(self):
        """Call summary to get result and parse result to dict.

        :return dict like: {'acc':{'name': 'acc', 'value': 0.9, 'reward_mode': 'MAX'}}
        """
        value = self.summary()
        if isinstance(value, dict):
            return value
        return {self.name: value}


class Metrics(object):
    """Metrics class of all metrics defined in cfg.

    :param metric_cfg: metric part of config
    :type metric_cfg: dict or Config
    """

    config = MetricsConfig()

    def __init__(self, metric_cfg=None):
        """Init Metrics."""
        self.mdict = {}
        metric_config = self.config.to_json() if not metric_cfg else deepcopy(metric_cfg)
        if not isinstance(metric_config, list):
            metric_config = [metric_config]
        for metric_item in metric_config:
            ClassFactory.get_cls(ClassType.METRIC, self.config.type)
            metric_name = metric_item.pop('type')
            metric_class = ClassFactory.get_cls(ClassType.METRIC, metric_name)
            if isfunction(metric_class):
                metric_class = partial(metric_class, **metric_item.get("params", {}))
            else:
                metric_class = metric_class(**metric_item.get("params", {}))
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
            pfms.append(metric(output, target, *args, **kwargs))
        return pfms

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        for val in self.mdict.values():
            val.reset()

    @property
    def results(self):
        """Return metrics results."""
        res = {}
        for name, metric in self.mdict.items():
            res.update(metric.result)
        return res

    @property
    def objectives(self):
        """Return objectives results."""
        return {name: self.mdict.get(name).objective for name in self.mdict}

    def __getattr__(self, key):
        """Get a metric by key name.

        :param key: metric name
        :type key: str
        """
        return self.mdict[key]


ClassFactory.register_from_package(metrics, ClassType.METRIC)
