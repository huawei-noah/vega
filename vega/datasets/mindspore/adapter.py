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

"""This is a base class of the dataset."""

import logging
from typing import Iterator
import numpy as np
from mindspore.dataset import GeneratorDataset, DistributedSampler, SubsetRandomSampler
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as vision
import mindspore.common.dtype as mstype
from mindspore.communication.management import get_rank, get_group_size


class MsAdapter(object):
    """This is the base class of the dataset, which is a subclass of `TaskOps`.

    The Dataset provide several basic attribute like dataloader, transform and sampler.
    """

    invalid_dtype = ("float64", "int64", "torch.float64", "torch.int64")
    dtype_map = {"float64": mstype.float32,
                 "int64": mstype.int32,
                 "torch.float64": mstype.float32,
                 "torch.int64": mstype.int32}

    def __init__(self, dataset):
        self.dataset = dataset
        self.args = dataset.args
        self.sampler = self._init_sampler()

    def convert_dtype(self, ms_dataset):
        """Convert the dataset dtype if the dtype is invalid.

        :param ms_dataset: a dataset object of mindspore
        :return: a dataset object of mindspore after dtype convert
        """
        item = self.dataset[0]
        image, label = item[0], item[1]
        try:
            image_dtype = str(image.dtype)
        except Exception:
            logging.debug('Falied to get image dtype.')
        try:
            label_dtype = str(label.dtype)
        except Exception:
            label_dtype = "int64"
        if image_dtype in self.invalid_dtype:
            type_cast_op = C2.TypeCast(self.dtype_map[image_dtype])
            ms_dataset = ms_dataset.map(input_columns="image", operations=type_cast_op)

        if label_dtype in self.invalid_dtype:
            type_cast_op = C2.TypeCast(self.dtype_map[label_dtype])
            ms_dataset = ms_dataset.map(input_columns="label", operations=type_cast_op)

        return ms_dataset

    def _init_sampler(self):
        """Initialize sampler method.

        :return: if the distributed is True, return a sampler object, else return None
        :rtype: an object or None
        """
        if self.dataset.world_size > 1:
            sampler = DistributedSampler(num_shards=self.dataset.world_size,
                                         shard_id=self.dataset.rank,
                                         shuffle=self.args.shuffle)
            self.args.shuffle = False
        elif not hasattr(self.args, "train_portion"):
            sampler = None
        elif self.dataset.mode == 'test' or self.args.train_portion == 1:
            sampler = None
        else:
            self.args.shuffle = False
            num_train = len(self.dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.args.train_portion * num_train))
            if self.dataset.mode == 'train':
                sampler = SubsetRandomSampler(indices[:split])
            elif self.dataset.mode == 'val':
                sampler = SubsetRandomSampler(indices[split:num_train])
            else:
                raise ValueError('the mode should be train, val or test')
        return sampler

    @property
    def loader(self):
        """Dataloader arrtribute which is a unified interface to generate the data.

        :return: a batch data
        :rtype: dict, list, optional
        """
        from mindspore.dataset.engine.datasets import BatchDataset, MapDataset
        BatchDataset.__len__ = BatchDataset.get_dataset_size
        MapDataset.__len__ = MapDataset.get_dataset_size
        GeneratorDataset.__len__ = GeneratorDataset.get_dataset_size
        Iterator.__len__ = lambda x: x.dataset.get_dataset_size()
        if hasattr(self.dataset, "data_loader"):
            return self.dataset.data_loader
        rank_size = 1
        rank_id = 0
        if self.dataset.world_size > 1:
            rank_size = get_group_size()
            rank_id = get_rank()
            self.sampler = None
        ms_dataset = GeneratorDataset(self.dataset, ["image", "label"], sampler=self.sampler, num_shards=rank_size,
                                      shard_id=rank_id)
        ms_dataset = self.convert_dtype(ms_dataset)
        if self.args.shuffle:
            buffer_size = self.args.get("buffer_size", len(self.dataset))
            ms_dataset = ms_dataset.shuffle(buffer_size=buffer_size)

        if self.args.get("mixup", False):
            num_class = self.args.get("num_class")
            one_hot_op = C2.OneHot(num_classes=num_class)
            ms_dataset = ms_dataset.map(operations=one_hot_op, input_columns=["label"])

            mixup_batch_op = vision.MixUpBatch(2)
            ms_dataset = ms_dataset.batch(self.args.batch_size)
            ms_dataset = ms_dataset.map(operations=mixup_batch_op, input_columns=["image", "label"])
        else:
            ms_dataset = ms_dataset.batch(self.args.batch_size)

        return ms_dataset
