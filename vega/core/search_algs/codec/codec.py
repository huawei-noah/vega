# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined the basic Codec."""
from vega.common import ClassFactory, ClassType


class Codec(object):
    """Base Codec class.

    :param codec_name: Codec name.
    :type codec_name: str
    :param search_space: input search_space.
    :type search_space: SearchSpace
    """

    def __new__(cls, *args, **kwargs):
        """Create search algorithm instance by ClassFactory."""
        if cls.__name__ != 'Codec':
            return super().__new__(cls)
        if kwargs.get('type'):
            t_cls = ClassFactory.get_cls(ClassType.CODEC, kwargs.pop('type'))
        else:
            t_cls = ClassFactory.get_cls(ClassType.CODEC)
        return super().__new__(t_cls)

    def __init__(self, search_space=None, **kwargs):
        """Init Codec."""
        self.search_space = search_space

    def encode(self, desc):
        """Encode function need to implement."""
        raise NotImplementedError

    def decode(self, code):
        """Decode function need to implement."""
        raise NotImplementedError
