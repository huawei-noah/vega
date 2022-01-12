# -*- coding:utf-8 -*-

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
"""This is DAG Cell for network."""
from vega.modules.module import Module
from vega.common.dag import DAG
import numpy as np
from vega.modules.operators import ops
from vega.modules.connections import Sequential
from vega.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.NETWORK)
class DagGraphCell(Module):
    """Merge and process inputs to the middle-level graph."""

    def __init__(self, adj_matrix, nodes, in_channels=64, out_channels=64):
        super(DagGraphCell, self).__init__()
        self.adj_matrix = adj_matrix
        self.nodes = nodes
        self.c_in = in_channels
        self.c_out = out_channels
        self._add_nodes()

    def _add_nodes(self):
        for node_id, node_name in enumerate(self.nodes):
            module = ClassFactory.get_instance(ClassType.NETWORK, node_name, in_channels=self.c_in,
                                               out_channels=self.c_out)
            self.add_module(str(node_id), module)

    def _create_dag(self):
        dag = DAG()
        for name, modules in self.named_children():
            dag.add_node(int(name))
        frontier = [0]
        num_vertices = np.shape(self.adj_matrix)[0]
        while frontier:
            node_id = frontier.pop()
            for v in range(num_vertices):
                if self.adj_matrix[node_id][v]:
                    dag.add_edge(node_id, v)
                    frontier.append(v)
        self.out_tensors = {}
        return dag

    def forward(self, x, *args, **kwargs):
        """Forward x."""
        dag = self._create_dag()
        node = dag.ind_nodes()[0]
        out = self._forward_module(x, node, dag)
        return out

    def _forward_module(self, x, parent, dag):
        parent_nodes = dag.pre_nodes(parent)
        if len(parent_nodes) <= 1:
            next_input = self._modules.get(str(parent))(x)
        elif self.out_tensors.get(parent) and len(self.out_tensors.get(parent)) == len(parent_nodes) - 1:
            out = self.out_tensors.pop(parent)
            out.append(x)
            next_input = self._modules.get(str(parent))(out)
        else:
            if parent not in self.out_tensors:
                self.out_tensors[parent] = []
            self.out_tensors[parent].append(x)
            return None
        children = dag.next_nodes(parent)
        for child in children:
            out = self._forward_module(next_input, child, dag)
            if out is not None:
                next_input = out
        return next_input


class ConvBnRelu(Module):
    """Conv bn Relu class."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.conv_bn_relu = Sequential(
            ops.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            ops.BatchNorm2d(out_channels),
            ops.Relu(inplace=True)
        )

    def call(self, x):
        """Call forward function."""
        return self.conv_bn_relu(x)


@ClassFactory.register(ClassType.NETWORK)
class Conv3x3BnRelu(Module):
    """The Class of 3x3 convolution with batch norm and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super(Conv3x3BnRelu, self).__init__()
        self.conv3x3 = ConvBnRelu(in_channels, out_channels, 3, 1, 1)

    def call(self, x):
        """Call forward function."""
        return self.conv3x3(x)


@ClassFactory.register(ClassType.NETWORK)
class Conv1x1BnRelu(Module):
    """The Class of 1x1 convolution with batch norm and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super(Conv1x1BnRelu, self).__init__()
        self.conv1x1 = ConvBnRelu(in_channels, out_channels, 1, 1, 0)

    def call(self, x):
        """Call forward function."""
        return self.conv1x1(x)


@ClassFactory.register(ClassType.NETWORK)
class MaxPool3x3(Module):
    """The class of 3x3 max pool with no subsampling."""

    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(MaxPool3x3, self).__init__()
        self.maxpool = ops.MaxPool2d(kernel_size, stride, padding)

    def call(self, x):
        """Call forward function."""
        return self.maxpool(x)


@ClassFactory.register(ClassType.NETWORK)
class Input(Module):
    """Input Class."""

    def __init__(self, size=None):
        super(Input, self).__init__()
        self.size = size

    def call(self, x):
        """Call forward function."""
        return x


@ClassFactory.register(ClassType.NETWORK)
class Output(Module):
    """Output Class."""

    def __init__(self, size=None):
        super(Output, self).__init__()
        self.size = size

    def call(self, x, **kwargs):
        """Call forward function."""
        return ops.concat(x, 1)
