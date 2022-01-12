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

"""This is a class for Nasbench dataset."""
import random
import numpy as np
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.datasets.conf.nasbench import NasbenchConfig
from nasbench import api
from .dataset import Dataset

VALID_OPS = ['input', 'output', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']


@ClassFactory.register(ClassType.DATASET)
class Nasbench(Dataset):
    """NASbench Dataset."""

    config = NasbenchConfig()

    def __init__(self, **kwargs):
        """Construct the Nasbench101 class."""
        Dataset.__init__(self, **kwargs)
        self.node_size = 7
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        self.nasbench = api.NASBench(self.args.data_path)
        self.keys = list(self.nasbench.hash_iterator())
        random.shuffle(self.keys)
        split_num = int(len(self.keys) * self.args.portion)
        self.mean, self.std = self._get_mean_std(split_num)
        if self.mode == 'train':
            self.keys = self.keys[:split_num]
        elif self.mode == 'val':
            self.keys = self.keys[split_num:]
        self.global_node = self.args.get('global_node', True)

    def _get_mean_std(self, length):
        """Get the mean and standard of NASBench target of train dataset part."""
        metrics_list = []
        for i in range(length):
            key = self.keys[i]
            _, computed_stat = self.nasbench.get_metrics_from_hash(key)
            computed_stat_epoch = computed_stat[self.args.epochs]
            target = sum([item[self.args.computed_key] for item in computed_stat_epoch]) / len(computed_stat_epoch)
            metrics_list.append(target)
        metrics_list = np.array(metrics_list, np.float32)
        mean = np.mean(metrics_list)
        std = np.std(metrics_list)
        return mean, std

    def __getitem__(self, index):
        """Get an item of the dataset according to the index."""
        fixed_stat, computed_stat = self.nasbench.get_metrics_from_hash(self.keys[index])
        adjacency_matrix = fixed_stat['module_adjacency']
        operations = fixed_stat['module_operations']
        computed_stat_epoch = computed_stat[self.args.epochs]
        target_list = [item[self.args.computed_key] for item in computed_stat_epoch]
        target = np.array([sum(target_list) / len(target_list)], dtype=np.float32)
        target = (target - self.mean) / self.std

        adjacency_matrix = np.array(adjacency_matrix).astype(np.float32)
        feature_matrix = np.zeros((len(operations), len(VALID_OPS)), dtype=np.float32)
        for i, op in enumerate(operations):
            pos = VALID_OPS.index(op)
            feature_matrix[i][pos] = 1.
        while adjacency_matrix.shape[0] < self.node_size:
            feature_matrix = np.row_stack((feature_matrix,
                                           np.zeros(feature_matrix.shape[1], dtype=np.float32)))
            adjacency_matrix = np.row_stack((adjacency_matrix,
                                             np.zeros(adjacency_matrix.shape[1], dtype=np.float32)))
            adjacency_matrix = np.column_stack((adjacency_matrix,
                                                np.zeros(adjacency_matrix.shape[0], dtype=np.float32)))
        if self.global_node:
            feature_matrix = self._add_global_node(feature_matrix, if_adj=False)
            adjacency_matrix = self._add_global_node(adjacency_matrix, if_adj=True)
        input = np.column_stack((adjacency_matrix, feature_matrix))

        return input, target

    def _add_global_node(self, mx, if_adj):
        """Add global node information to matrix."""
        if if_adj:
            mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
            np.fill_diagonal(mx, 1)
            mx = mx.T
        else:
            mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
            mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
        return mx

    def __len__(self):
        """Length of dataset."""
        return len(self.keys)
