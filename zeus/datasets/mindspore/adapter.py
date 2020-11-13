# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a base class of the dataset."""
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.common.dtype as mstype


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

    def convert_dtype(self, ms_dataset):
        """Convert the dataset dtype if the dtype is invalid.

        :param ms_dataset: a dataset object of mindspore
        :return: a dataset object of mindspore after dtype convert
        """
        item = self.dataset[0]
        image, label = item[0], item[1]
        try:
            image_dtype = str(image.dtype)
        except:
            pass
        try:
            label_dtype = str(label.dtype)
        except:
            label_dtype = "int64"
        if image_dtype in self.invalid_dtype:
            type_cast_op = C2.TypeCast(self.dtype_map[image_dtype])
            ms_dataset = ms_dataset.map(input_columns="image", operations=type_cast_op)

        if label_dtype in self.invalid_dtype:
            type_cast_op = C2.TypeCast(self.dtype_map[label_dtype])
            ms_dataset = ms_dataset.map(input_columns="label", operations=type_cast_op)

        return ms_dataset

    @property
    def loader(self):
        """Dataloader arrtribute which is a unified interface to generate the data.

        :return: a batch data
        :rtype: dict, list, optional
        """
        ms_dataset = GeneratorDataset(self.dataset, ["image", "label"])
        # ms_dataset.set_dataset_size(len(self.dataset))  # TODO delete, only mindspore 0.5 need
        ms_dataset = self.convert_dtype(ms_dataset)
        if self.args.shuffle:
            ms_dataset = ms_dataset.shuffle(buffer_size=len(self.dataset))
        ms_dataset = ms_dataset.batch(self.args.batch_size)

        return ms_dataset
