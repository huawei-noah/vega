# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
