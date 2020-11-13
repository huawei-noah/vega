# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default loss configs."""
from zeus.common import ConfigSerializable


class LossConfig(ConfigSerializable):
    """Default Loss Config."""

    _class_type = "trainer.loss"
    _exclude_keys = ['type']
    _update_all_attrs = True
    type = 'CrossEntropyLoss'
    params = {}

    @classmethod
    def from_json(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        cls = super(LossConfig, cls).from_json(data, skip_check)
        if "params" not in data:
            cls.params = {}
        return cls

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules = {"type": {"type": str},
                 "params": {"type": dict}}
        return rules


class LossMappingDict(object):
    """Loss Mapping Dictionary."""

    type_mapping_dict = dict(
        CrossEntropyLoss=dict(torch='CrossEntropyLoss', tf='CrossEntropyLoss',
                              ms='SoftmaxCrossEntropyWithLogits'),
        MixAuxiliaryLoss=dict(torch='MixAuxiliaryLoss', tf='MixAuxiliaryLoss', ms=None),
        L1Loss=dict(torch='L1Loss', tf='absolute_difference', ms="L1Loss"),
    )

    params_mapping_dict = dict(
        CrossEntropyLoss=dict(
            ignore_index=dict(torch='ignore_index', tf='ignore_index', ms=None),
            is_grad=dict(torch=None, tf=None, ms='is_grad'),
            sparse=dict(torch=None, tf=None, ms='sparse'),
        ),
        MixAuxiliaryLoss=dict(
            loss_base=dict(torch='loss_base', tf='loss_base'),
            aux_weight=dict(torch='aux_weight', tf='aux_weight'),
        )
    )
