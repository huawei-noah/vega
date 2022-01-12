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


class ComposeAll(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        """Construct the Compose class."""
        self.transforms = transforms

    def __call__(self, *inputs):
        """Call function of Compose."""
        for t in self.transforms:
            inputs = t(*inputs)
        return inputs

    def __repr__(self):
        """Construct method."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
