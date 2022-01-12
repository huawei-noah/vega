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
from __future__ import division
import math
import torch
import numpy as np
from torch.utils.data import Sampler
from torch.utils.data import DistributedSampler as _DistributedSampler
from vega.common.utils_torch import get_dist_info


class DistributedSampler(_DistributedSampler):
    """This is a distributed sampler method for distributed train.

    :param dataset: Dataset used for sampling
    :type dataset: a class which contains `__len()__`  and `__getitem()`
    :param num_replicas: Number of processes participating in distributed training, defaults to None
    :type num_replicas: int, optional
    :param rank: Rank of the current process within num_replicas, defaults to None
    :type rank: int, optional
    :param shuffle: whether the shuffle is used, defaults to True
    :type shuffle: bool, optional
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        """Construct the DistributedSampler class."""
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        """Provide a way to iterate over indices of dataset elements.

        :return: an iterator object
        :rtype: object
        """
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if not len(indices) == self.total_size:
            raise ValueError('the length of the indices should be equal to total_size')

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if not len(indices) == self.num_samples:
            raise ValueError("the length of the indices should be equal to num_samples in subsample")

        return iter(indices)


class DistributedGroupSampler(Sampler):
    """This is a distributed sampler method for distributed train.

    :param dataset: Dataset used for sampling
    :type dataset: a class which contains `__len()__`  and `__getitem()`
    :param samples_per_gpu: equal to images per gpu , defaults to 1
    :type samples_per_gpu: int, optional
    :param num_replicas: Number of processes participating in distributed training, defaults to None
    :type num_replicas: int, optional
    :param rank: Rank of the current process within num_replicas, defaults to None
    :type rank: int, optional
    :param shuffle: whether the shuffle is used, defaults to True
    :type shuffle: bool, optional
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        """Construct the DistributedGroupSampler class."""
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if not hasattr(self.dataset, 'flag'):
            raise ValueError("the dataset doesn't have a atrribute named 'flag'")
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(math.ceil(
                self.group_sizes[i] * 1.0 / self.samples_per_gpu / self.num_replicas
            )) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """Provide a way to iterate over indices of dataset elements.

        :return: an iterator object
        :rtype: object
        """
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                if not len(indice) == size:
                    raise ValueError("the length of the indice should be equal to size")
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        if not len(indices) == self.total_size:
            raise ValueError("the length of the indices should be equal to total_size")

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        if not len(indices) == self.num_samples:
            raise ValueError("the length of the indices should be equal to num_samplers in subsample")

        return iter(indices)

    def __len__(self):
        """Get the length.

        :return: the length of the returned iterators.
        :rtype: int
        """
        return self.num_samples

    def set_epoch(self, epoch):
        """Set the epoch.

        :param epoch: the epoch value to set
        :type epoch: int
        """
        self.epoch = epoch


class GroupSampler(Sampler):
    """This is a non-distributed sampler method for non-distributed train.

    :param dataset: Dataset used for sampling
    :type dataset: a class which contains `__len()__`  and `__getitem()`
    :param samples_per_gpu: equal to images per gpu , defaults to 1
    :type samples_per_gpu: int, optional
    """

    def __init__(self, dataset, samples_per_gpu=1):
        """Construct the GroupSampler class."""
        if not hasattr(dataset, 'flag'):
            raise ValueError("the dataset doesn't have an attribute named 'flag ")
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        """Provide a way to iterate over indices of dataset elements.

        :return: an iterator object
        :rtype: object
        """
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            if not len(indice) == size:
                raise ValueError('the length of the indice should be equal to the size')
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = torch.from_numpy(indices).long()
        if not len(indices) == self.num_samples:
            raise ValueError("the length of the indices should be equal to num_samples")
        return iter(indices)

    def __len__(self):
        """Get the length.

        :return: the length of the returned iterators.
        :rtype: int
        """
        return self.num_samples
