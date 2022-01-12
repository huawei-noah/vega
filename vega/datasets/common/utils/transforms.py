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

"""This is a class for Transforms."""

from vega.datasets.transforms.Compose import Compose, ComposeAll
from vega.datasets.transforms.Compose_pair import Compose_pair
from vega.common import ClassFactory, ClassType


class Transforms(object):
    """This is the base class of the transform.

    The Transforms provide several basic method like append, insert, remove and replace.
    """

    def __init__(self, transform_list=None):
        """Construct Transforms class."""
        self.__transform__ = []
        self._new(transform_list)

    def __call__(self, *args):
        """Call fuction."""
        if len(args) == 1:
            return Compose(self.__transform__)(*args)
        elif len(args) == 2:
            return Compose_pair(self.__transform__)(*args)
        else:
            return ComposeAll(self.__transform__)(*args)

    def _new(self, transform_list):
        """Private method, which generate a list of transform.

        :param transform_list: a series of transforms
        :type transform_list: list
        """
        if isinstance(transform_list, list):
            for trans in transform_list:
                if isinstance(trans, tuple):
                    transform = ClassFactory.get_cls(ClassType.TRANSFORM, trans[0])
                    self.__transform__.append(transform(*trans[1:]))
                elif isinstance(trans, object):
                    self.__transform__.append(trans)
                else:
                    raise ValueError("Unsupported type ({}) to create transforms".format(trans))
        else:
            raise ValueError("Transforms ({}) is not a list".format(transform_list))

    def replace(self, transform_list):
        """Replace the transforms with the new transforms.

        :param transform_list: a series of transforms
        :type transform_list: list
        """
        if isinstance(transform_list, list):
            self.__transform__[:] = []
            self._new(transform_list)

    def append(self, *args, **kwargs):
        """Append a transform to the end of the list.

        :param *args: positional arguments
        :type *args: tuple
        :param ** kwargs: keyword argumnets
        :type ** kwargs: dict
        """
        if isinstance(args[0], str):
            transform = ClassFactory.get_cls(ClassType.TRANSFORM, args[0])
            self.__transform__.append(transform(**kwargs))
        else:
            self.__transform__.append(args[0])

    def insert(self, index, *args, **kwargs):
        """Insert a transform into the list.

        :param index: Insertion position
        :type index: int
        :param *args: positional arguments
        :type *args: tuple
        :param ** kwargs: keyword argumnets
        :type ** kwargs: dict
        """
        if isinstance(args[0], str):
            transform = ClassFactory.get_cls(ClassType.TRANSFORM, args[0])
            self.__transform__.insert(index, transform(**kwargs))
        else:
            self.__transform__.insert(index, args[0])

    def remove(self, transform_name):
        """Remove a transform from the transform_list.

        :param transform_name: name of transform
        :type transform_name: str
        """
        if isinstance(transform_name, str):
            for trans in self.__transform__:
                if transform_name == trans.__class__.__name__:
                    self.__transform__.remove(trans)
