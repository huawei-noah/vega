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
        SGD=dict(torch='SGD', tf='MomentumOptimizer', ms='SGD'),
        Momentum=dict(torch='SGD', tf='MomentumOptimizer', ms='Momentum'),
        Adam=dict(torch='Adam', tf='AdamOptimizer', ms='Adam'),
        RMSProp=dict(torch='RMSProp', tf='RMSPropOptimizer', ms='RMSProp')
    )

    params_mapping_dict = dict(
        SGD=dict(
            lr=dict(torch='lr', tf='learning_rate', ms='learning_rate'),
            momentum=dict(torch='momentum', tf='momentum', ms='momentum'),
            weight_decay=dict(torch='weight_decay', tf='weight_decay', ms='weight_decay'),
            no_decay_params=dict(torch=None, tf=None, ms='no_decay_params'),
            loss_scale=dict(torch=None, tf=None, ms='loss_scale'),
        ),
        Momentum=dict(
            lr=dict(torch='lr', tf='learning_rate', ms='learning_rate'),
            momentum=dict(torch='momentum', tf='momentum', ms='momentum'),
            weight_decay=dict(torch='weight_decay', tf='weight_decay', ms='weight_decay'),
            no_decay_params=dict(torch=None, tf=None, ms='no_decay_params'),
            loss_scale=dict(torch=None, tf=None, ms='loss_scale'),
        ),
        Adam=dict(
            lr=dict(torch='lr', tf='learning_rate', ms='learning_rate'),
            weight_decay=dict(torch='weight_decay', tf='weight_decay', ms='weight_decay'),
            no_decay_params=dict(torch=None, tf=None, ms='no_decay_params'),
            loss_scale=dict(torch=None, tf=None, ms='loss_scale'),
        ),
        RMSProp=dict(
            lr=dict(torch='lr', tf='learning_rate', ms='learning_rate'),
            weight_decay=dict(torch='weight_decay', tf='weight_decay', ms='weight_decay'),
            no_decay_params=dict(torch=None, tf=None, ms='no_decay_params'),
            loss_scale=dict(torch=None, tf=None, ms='loss_scale'),
        )
    )
