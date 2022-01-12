# -*- coding: utf-8 -*-

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

"""Compressed model filter."""
from collections import OrderedDict
from vega.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.PRETRAINED_HOOK)
def pretrained_bert_classifier_hook(state_dict, prefix, local_metadata, *args, **kwargs):
    """Convert state_dict name according to prefix."""
    org = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('bert'):
            org[key] = value
    state_dict.clear()
    state_dict.update(org)
    return state_dict


@ClassFactory.register(ClassType.PRETRAINED_HOOK)
def pretrained_bert_hook(state_dict, prefix, local_metadata, *args, **kwargs):
    """Convert state_dict name according to prefix."""
    local_state_dict = OrderedDict()
    prefix = 'bert'
    for key, value in state_dict.items():
        if key.startswith(prefix):
            key = key.replace(prefix + '.', '')
            local_state_dict[key] = value
    state_dict.clear()
    state_dict.update(local_state_dict)
    return state_dict
