# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default lr_scheduler configs."""
from zeus.common import ConfigSerializable


class LrSchedulerConfig(ConfigSerializable):
    """Default LrScheduler Config."""

    _class_type = "trainer.lr_scheduler"
    _update_all_attrs = True
    _exclude_keys = ['type']
    type = 'MultiStepLR'
    params = {"milestones": [75, 150], "gamma": 0.5}

    @classmethod
    def from_json(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        cls = super(LrSchedulerConfig, cls).from_json(data, skip_check)
        if "params" not in data:
            cls.params = {}
        return cls

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules = {"type": {"type": str},
                 "params": {"type": dict}}
        return rules


class LrSchedulerMappingDict(object):
    """Lr Scheduler Mapping Dictionary."""

    type_mapping_dict = dict(
        StepLR=dict(torch='StepLR', tf='StepLR'),
        MultiStepLR=dict(torch='MultiStepLR', tf='MultiStepLRWarmUp'),
        CosineAnnealingLR=dict(torch='CosineAnnealingLR', tf='CosineAnnealingLR'),
    )

    params_mapping_dict = dict(
        StepLR=dict(
            step_size=dict(torch='step_size', tf='step_size'),
            gamma=dict(torch='gamma', tf='gamma'),
        ),
        MultiStepLR=dict(
            milestones=dict(torch='milestones', tf='milestones'),
            gamma=dict(torch='gamma', tf='gamma'),
            warmup=dict(torch=None, tf='warmup'),
        ),
        CosineAnnealingLR=dict(
            T_max=dict(torch='T_max', tf='T_max'),
            warmup=dict(torch=None, tf='warmup'),
            eta_min=dict(torch='eta_min', tf='eta_min'),
        )
    )
