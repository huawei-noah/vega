# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Nasbench101 dataset."""
from zeus.common import ClassFactory, ClassType
from zeus.common import FileOps
from .utils.dataset import Dataset
from nas_201_api import NASBench201API as API

VALID_OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
VALID_DATASET = ["cifar10", "cifar100", "ImageNet16-120"]


@ClassFactory.register(ClassType.DATASET)
class Nasbench201(Dataset):
    """Nasbench201 Dataset."""

    def __init__(self):
        """Construct the Nasbench201 class."""
        super(Nasbench201, self).__init__()
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        self.nasbench201_api = API('self.args.data_path')

    def query(self, arch_str, dataset):
        """Query an item from the dataset according to the given arch_str and dataset .

        :arch_str: arch_str to define the topology of the cell
        :type path: str
        :dataset: dataset type
        :type dataset: str
        :return: an item of the dataset, which contains the network info and its results like accuracy, flops and etc
        :rtype: dict
        """
        if dataset not in VALID_DATASET:
            raise ValueError("Only cifar10, cifar100, and Imagenet dataset is supported.")
        ops_list = self.nasbench201_api.str2lists(arch_str)
        for op in ops_list:
            if op not in VALID_OPS:
                raise ValueError("{} is not in the nasbench201 space.".format(op))
        index = self.nasbench201_api.query_index_by_arch(arch_str)

        results = self.nasbench201_api.query_by_index(index, dataset)
        return results
