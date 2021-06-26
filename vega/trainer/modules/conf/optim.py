# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default optimizer configs."""
from vega.common import ConfigSerializable


class OptimConfig(ConfigSerializable):
    """Default Optim Config."""

    _class_type = "trainer.optimizer"
    _exclude_keys = ['type']
    _update_all_attrs = True
    type = 'Adam'
    params = {"lr": 0.1}

    @classmethod
    def from_dict(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        cls = super(OptimConfig, cls).from_dict(data, skip_check)
        if "params" not in data:
            cls.params = {}
        return cls

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules = {"type": {"type": str},
                 "params": {"type": dict}}
        return rules


class OptimMappingDict(object):
    """Optimizer Mapping Dictionary."""

    type_mapping_dict = dict(
        SGD=dict(torch='SGD', tf='MomentumOptimizer', ms='Momentum'),
        Adam=dict(torch='Adam', tf='AdamOptimizer', ms='Adam'),
        RMSProp=dict(torch='RMSProp', tf='RMSPropOptimizer', ms='RMSProp')
    )

    params_mapping_dict = dict(
        SGD=dict(
            lr=dict(torch='lr', tf='learning_rate', ms='learning_rate'),
            momentum=dict(torch='momentum', tf='momentum', ms='momentum'),
            weight_decay=dict(torch='weight_decay', tf='weight_decay', ms='weight_decay'),
        ),
        Adam=dict(
            lr=dict(torch='lr', tf='learning_rate', ms='learning_rate'),
            weight_decay=dict(torch='weight_decay', tf='weight_decay', ms='weight_decay'),
        ),
        RMSProp=dict(
            lr=dict(torch='lr', tf='learning_rate', ms='learning_rate'),
            weight_decay=dict(torch='weight_decay', tf='weight_decay', ms='weight_decay'),
        )
    )
