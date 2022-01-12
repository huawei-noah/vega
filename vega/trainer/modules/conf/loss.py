# -*- coding=utf-8 -*-

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
"""Default loss configs."""
from vega.common import ConfigSerializable
import vega


class LossConfig(ConfigSerializable):
    """Default Loss Config."""

    _class_type = "trainer.loss"
    _exclude_keys = ['type']
    _update_all_attrs = True
    type = 'CrossEntropyLoss'

    @classmethod
    def from_dict(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        cls = super(LossConfig, cls).from_dict(data, skip_check)
        if vega.is_ms_backend():
            if "params" not in data:
                cls.params = {'sparse': True}
            elif "sparse" not in data.params:
                cls.params.update({'sparse': True})
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
        MixAuxiliaryLoss=dict(torch='MixAuxiliaryLoss', tf='MixAuxiliaryLoss', ms='MixAuxiliaryLoss'),
        L1Loss=dict(torch='L1Loss', tf='absolute_difference', ms="L1Loss"),
        MSELoss=dict(torch='MSELoss', tf='mean_squared_error', ms=None),
    )

    params_mapping_dict = dict(
        CrossEntropyLoss=dict(
            ignore_index=dict(torch='ignore_index', tf='ignore_index', ms=None),
            sparse=dict(torch=None, tf=None, ms='sparse'),
        ),
        MixAuxiliaryLoss=dict(
            loss_base=dict(torch='loss_base', tf='loss_base', ms='loss_base'),
            aux_weight=dict(torch='aux_weight', tf='aux_weight', ms='aux_weight'),
        )
    )
