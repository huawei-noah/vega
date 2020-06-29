# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Compose."""


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        """Construct the Compose class."""
        self.transforms = transforms

    def __call__(self, img):
        """Call function of Compose."""
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        """Construct method."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
