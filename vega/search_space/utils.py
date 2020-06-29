# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Utils function of SearchSpace."""
from vega.core.common.class_factory import ClassFactory, ClassType


def get_search_space(cls_name):
    """Get Search Space by class name.

    :param cls_name: class name
    :return: Search Space cls
    """
    return ClassFactory.get_cls(ClassType.SEARCH_SPACE, cls_name, bing_cfg=False)
