# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Transforms."""
from vega.datasets.pytorch.transforms import Compose
from vega.datasets.pytorch.transforms import Compose_pair
from vega.core.common.class_factory import ClassFactory, ClassType


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
            raise ValueError("Length of args must be either 1 or 2")

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
                    raise ValueError("Unsupported type ({}) to create transforms" .format(trans))
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
