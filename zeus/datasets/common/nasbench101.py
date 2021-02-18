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
from nasbench import api

VALID_OPS = ['input', 'output', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']


@ClassFactory.register(ClassType.DATASET)
class Nasbench101(Dataset):
    """Nasbench101 Dataset."""

    def __init__(self):
        """Construct the Nasbench101 class."""
        super(Nasbench101, self).__init__()
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        self.nasbench = api.NASBench(self.args.data_path)

    def query(self, adjacency_matrix, ops_list):
        """Query an item from the dataset according to the given adjacency_matrix and ops_list .

        :adjacency_matrix: adjacency_matrix to define the topology of the cell
        :type adjacency_matrix: lsit(list)
        :ops_list: the ops in the cell
        :type ops_list: lsit
        :return: an item of the dataset, which contains the network info and its results like accuracy, flops and etc
        :rtype: dict
        """
        if not (isinstance(adjacency_matrix, list) and isinstance(adjacency_matrix[0], list)):
            raise ValueError("The matrix must be a 2D list.")
        if ops_list[0] != "input":
            raise ValueError("The first op must be input.")
        if ops_list[-1] != "output":
            raise ValueError("The last op must be output.")
        for op in ops_list:
            if op not in VALID_OPS:
                raise ValueError("{} is not in the nasbench101 space.".format(op))
        model_spec = api.ModelSpec(matrix=adjacency_matrix, ops=ops_list)
        return self.nasbench.query(model_spec)
