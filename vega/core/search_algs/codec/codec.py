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
