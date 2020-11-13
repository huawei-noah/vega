# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric Target Termination."""
from zeus.common import ClassFactory, ClassType
import logging
from zeus.common.general import General
import copy


class QuotaCompare(object):
    """Determine whether to satisfy target."""

    def __init__(self, type):
        if type not in ['restrict', 'target']:
            raise ValueError('Input type must be restriction or target.')
        self.filter_types, self.terminate_types = [], []
        if type == 'restrict':
            self.filter_types = ['FlopsParamsFilter', 'LatencyFilter']
            self.terminate_types = ['DurationTerminate', 'TrialTerminate']
        elif type == 'target':
            self.terminate_types = ['TargetTerminate']
        self.filter_compares = dict()
        self.terminate_compares = dict()
        for filter in self.filter_types:
            t_cls = ClassFactory.get_cls(ClassType.QUOTA, filter)
            self.filter_compares[filter] = t_cls()
        for terminate in self.terminate_types:
            t_cls = ClassFactory.get_cls(ClassType.QUOTA, terminate)
            self.terminate_compares[terminate] = t_cls()
        self.filter_rules = copy.deepcopy(General.quota.filter_rules)
        self.filters_to_params = {
            'FlopsParamsFilter': 'flops_params',
            'LatencyFilter': 'max_latency'
        }

    def is_filtered(self, res):
        """Quota Compare filter function."""
        exact_filters = []
        for filter in self.filter_types:
            if self.filters_to_params[filter] in self.filter_rules:
                exact_filters.append(filter)
        filter_to_bool = dict()
        for filter in exact_filters:
            filter_to_bool[filter] = str(self.filter_by_name(res, filter))
        filter_rules_str = copy.deepcopy(self.filter_rules)
        for filter in exact_filters:
            filter_rules_str = filter_rules_str.replace(self.filters_to_params[filter], filter_to_bool[filter])
        return bool(eval(filter_rules_str))

    def is_halted(self, *args, **kwargs):
        """Quota Compare halt function."""
        for compare in self.terminate_compares.values():
            if compare.is_halted(args, kwargs):
                return True
        return False

    def filter_by_name(self, res, name):
        """Filter sample by filter rule name."""
        filter = self.filter_compares[name]
        if filter.is_filtered(res):
            return True
        return False
