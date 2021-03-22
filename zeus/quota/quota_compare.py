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
from zeus.common.general import General
import copy


class QuotaCompare(object):
    """Determine whether to satisfy target."""

    def __init__(self, type):
        if type not in ['restrict', 'target']:
            raise ValueError('Input type must be restriction or target.')
        self.filter_types, self.terminate_types = [], []
        self.filter_compares = dict()
        self.terminate_compares = dict()
        self.filters_to_params = dict()
        self._init_compare_types(type)
        for filter in self.filter_types:
            t_cls = ClassFactory.get_cls(ClassType.QUOTA, filter)
            self.filter_compares[filter] = t_cls()
        for terminate in self.terminate_types:
            t_cls = ClassFactory.get_cls(ClassType.QUOTA, terminate)
            self.terminate_compares[terminate] = t_cls()
        self.filter_rules = copy.deepcopy(General.quota.filter_rules)

    def is_filtered(self, res):
        """Quota Compare filter function."""
        if len(self.filter_types) == 0:
            return False
        exact_filters = []
        for filter in self.filter_types:
            if self.filters_to_params[filter] in self.filter_rules:
                exact_filters.append(filter)
        filter_to_bool = dict()
        for filter in exact_filters:
            filter_to_bool[filter] = 'self.filter_compares[\'{}\'].is_filtered(res)'.format(filter)
        filter_rules_str = copy.deepcopy(self.filter_rules)
        for filter in exact_filters:
            filter_rules_str = filter_rules_str.replace(self.filters_to_params[filter], filter_to_bool[filter])
        return bool(eval(filter_rules_str))

    def is_halted(self, *args, **kwargs):
        """Quota Compare halt function."""
        if len(self.terminate_types) == 0:
            return False
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

    def _init_compare_types(self, type):
        """Initialize compare types."""
        if type == 'restrict':
            restrict_config = copy.deepcopy(General.quota.restrict)
            if restrict_config.flops or restrict_config.params:
                self.filter_types.append('FlopsParamsFilter')
            if restrict_config.latency:
                self.filter_types.append('LatencyFilter')
            if restrict_config.model_valid:
                self.filter_types.append('ValidFilter')
            if restrict_config.duration:
                self.terminate_types.append('DurationTerminate')
            if restrict_config.trials:
                self.terminate_types.append('TrialTerminate')
        elif type == 'target':
            target_config = copy.deepcopy(General.quota.target)
            if target_config.type:
                self.terminate_types.append('TargetTerminate')
        self.filters_to_params = {
            'ValidFilter': 'model_valid',
            'FlopsParamsFilter': 'flops_params',
            'LatencyFilter': 'max_latency'
        }
