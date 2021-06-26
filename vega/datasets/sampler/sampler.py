# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
