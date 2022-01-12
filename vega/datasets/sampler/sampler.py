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

"""This script include some common sampler methods."""
import numpy as np


class SubsetRandomSampler(object):
    """This is a random sampler method for subset."""

    def __init__(self, indices):
        """Construct the DistributedSampler class."""
        self.indices = indices

    def __iter__(self):
        """Provide a way to iterate over indices of dataset elements.

        :return: an iterator object
        :rtype: object
        """
        return (self.indices[i] for i in np.random.permutation(len(self.indices)))

    def __len__(self):
        """Get the length.

        :return: the length of the iterators.
        :rtype: int
        """
        return len(self.indices)
