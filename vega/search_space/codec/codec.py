# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined the basic Codec."""
import os
import copy
from vega.core.common import Config
from vega.core.common.utils import update_dict


class Codec(object):
    """Base Codec class.

    :param codec_name: Codec name.
    :type codec_name: str
    :param search_space: input search_space.
    :type search_space: SearchSpace
    """

    _subclasses = {}

    @classmethod
    def _get_subclasses(cls):
        """Return sub class for codec."""
        subclasses = {}
        for subclass in cls.__subclasses__():
            subclasses[subclass.__name__] = subclass
        return subclasses

    @classmethod
    def subclasses(cls):
        """Return sub class for codec."""
        if not cls._subclasses:
            cls._subclasses = cls._get_subclasses()
        return cls._subclasses

    def __new__(cls, codec_name, search_space=None):
        """Build new Codec."""
        for subclass_name, subclass in cls.subclasses().items():
            if subclass_name == str(codec_name):
                return super(Codec, cls).__new__(subclass)
        return super(Codec, cls).__new__(cls)

    def __init__(self, codec_name, search_space=None):
        """Init Codec."""
        return

    def encode(self, desc):
        """Encode function need to implement."""
        raise NotImplementedError

    def decode(self, code):
        """Decode function need to implement."""
        raise NotImplementedError
