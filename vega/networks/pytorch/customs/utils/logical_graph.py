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

"""LogicalGraph for NAGO."""
import time
import collections
import logging
from dataclasses import dataclass
from typing import List
import numpy as np
import networkx as nx


Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])


def _has_all_keys(op_dict, all_ops):
    """Check all keys."""
    return all([k in op_dict.keys() for k in all_ops]) and len(all_ops) == len(op_dict)


def get_graph_info(graph):
    """Label and sort nodes in a graph."""
    input_nodes = []
    output_nodes = []
    Nodes = []
    n_nodes = graph.number_of_nodes()
    if n_nodes == 1:
        node = Node(0, [], -1)
        return [node], [0], []
    for node in range(n_nodes):
        tmp = list(graph.neighbors(node))
        tmp.sort()
        type = -1
        if node < tmp[0]:
            input_nodes.append(node)
            type = 0
        if node > tmp[-1]:
            output_nodes.append(node)
            type = 1
        Nodes.append(Node(node, [n for n in tmp if n < node], type))
    return Nodes, input_nodes, output_nodes


def build_graph(graphparam, seed):
    """Build a graph using network x based on graphparamters."""
    graph_model_name = graphparam[0]
    if graph_model_name == 'ER':
        graph_model, nodes, P = graphparam
        return nx.random_graphs.erdos_renyi_graph(int(nodes), P, seed)
    elif graph_model_name == 'BA':
        graph_model, nodes, M = graphparam
        return nx.random_graphs.barabasi_albert_graph(int(nodes), int(M), seed)
    elif graph_model_name == 'WS':
        graph_model, nodes, P, K = graphparam
        return nx.random_graphs.connected_watts_strogatz_graph(int(nodes), int(K), P, tries=200, seed=seed)


def sample_merging_strategy(inputs, merge_distribution, role):
    """Sample merging options from a categorical distribution."""
    if role == NodeRoles.INPUT or len(inputs) == 1:
        return EdgeMerge.SINGLE
    return np.random.choice(EdgeMerge.merging_options, p=merge_distribution)


def compute_input_planes(input_channels, merging_strategy, inputs, abs_nodes):
    """Compute the number of input planes."""
    if len(inputs) == 0:  # this is an input node
        return input_channels
    inplanes = 0
    for i in inputs:
        if merging_strategy == EdgeMerge.CAT:
            inplanes += abs_nodes[i].outplanes
        else:  # Residual or Attention
            # this assumes that later stages are what determine the actual number of planes
            inplanes = abs_nodes[i].outplanes
    return inplanes


def get_stage_list(n_stages, stage_ratios):
    """Get stage label."""
    if n_stages == 3:
        return [0, 1, 2]
    stage_values = []
    stage = 0
    threshold = stage_ratios[stage] * n_stages
    for x in range(n_stages):
        if x >= threshold:
            stage += 1
            threshold += stage_ratios[stage] * n_stages
        stage_values.append(stage)
    return stage_values


class EdgeMerge:
    """Class Edge merge."""

    SUM = "SUM"
    CAT = "CAT"
    ATTENTION = "ATTENTION"
    SINGLE = "SINGLE"
    merging_options = [SUM, CAT, ATTENTION]


class Ops:
    """Class Ops."""

    POOL3 = "pool3x3"
    POOL5 = "pool5x5"
    C13 = "1x3"
    C31 = "3x1"
    C1 = "1x1"
    C5 = "5x5"
    C3 = "3x3"
    pooling_ops = [POOL3, POOL5]
    conv_ops = [C3, C5, C1, C31, C13]
    all_ops = conv_ops + pooling_ops
    ops_to_num_params = {C3: 9, C5: 25, C31: 3, C13: 3, C1: 1, POOL3: 0, POOL5: 0}
    ops_to_kernel_size = {C3: (3, 3), C5: (5, 5), C31: (3, 1), C13: (1, 3), C1: (1, 1), POOL3: (3, 3), POOL5: (5, 5)}
    if not _has_all_keys(ops_to_num_params, all_ops) or not _has_all_keys(ops_to_kernel_size, all_ops):
        raise ValueError("Ops must match.")


@dataclass
class GeneratorSolution:
    """Specify all the hyperparameters needed to define the generator."""

    master_params: List[float]
    cell_params: List[float]
    op_graph_params: List[float]
    stage_ratios: List[float]
    channel_ratios: List[int]
    op_dist: List[float]
    conv_type: str
    master_merge_dist: List[float]
    cell_merge_dist: List[float]
    op_merge_dist: List[float]


class NodeRoles:
    """Class node roles."""

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    TRANSIT = "TRANSIT"
    MASTER = "MASTER"


class BasicNode:
    """Define a basic graph in our generator."""

    tollerance = 30

    def __init__(self, node_inputs, role, merging_strategy, inplanes, outplanes, incoming_planes):
        """Initialize BasicNode."""
        self.inputs = node_inputs
        self.role = role
        self.merging_strategy = merging_strategy
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.incoming_planes = incoming_planes

    def _get_role_for_node(self, i):
        role = NodeRoles.TRANSIT
        if i in self.input_nodes:
            role = NodeRoles.INPUT
        elif i in self.output_nodes:
            role = NodeRoles.OUTPUT
        return role

    def _init_graph(self, graphparams):
        init_seed = 0
        for x in range(BasicNode.tollerance):
            graph = build_graph(graphparams, int(x + init_seed))
            self.graph = graph.copy()
            if nx.is_connected(self.graph) is not True:
                continue
            try:
                self.nodes, self.input_nodes, self.output_nodes = get_graph_info(self.graph)
                return graph
            except Exception:
                logging.debug('Failed to get graph.')
                continue
        self.nodes, self.input_nodes, self.output_nodes = ([], [], [])
        return graph


class LogicalOperation(BasicNode):
    """Each operation node in the bottom-level graph."""

    def __init__(self, op_type, conv_type, node_inputs, role, merging_strategy, inplanes, outplanes, incoming_planes):
        """Initialize LogicalOperation."""
        super().__init__(node_inputs, role, merging_strategy, inplanes, outplanes, incoming_planes)
        self.type = op_type
        self.conv_type = conv_type
        if op_type in Ops.pooling_ops:
            self.outplanes = self.inplanes

    def __repr__(self):
        """Overide __repr__."""
        return "%s, inputs: %s\t%s {%s:%d,%d}" % (
            self.type, self.inputs, self.role, self.merging_strategy[:2], self.inplanes, self.outplanes)

    def _get_param_count(self):
        if self.conv_type == 'depthwise_separable':
            if Ops.ops_to_num_params[self.type] == 0:
                param_count = 0
            else:
                param_count = Ops.ops_to_num_params[self.type] * self.inplanes + 1 * 1 * self.inplanes * self.outplanes
            return param_count

        else:
            return Ops.ops_to_num_params[self.type] * self.inplanes * self.outplanes


class LogicalOpGraph(BasicNode):
    """
    Define the logic bottom-level graph and define the details.

    (operation type, inputs, node type, input merging option,
    input/output channel number and parameter count) for in each operation node.
    """

    def __init__(self, role, inputs, merging_strategy, solution: GeneratorSolution, inplanes, outplanes,
                 incoming_planes):
        """Initialize LogicalOpGraph."""
        super().__init__(inputs, role, merging_strategy, inplanes, outplanes, incoming_planes)
        self.depth = "\t\t"
        self.bottomlvl_graph = self._init_graph(solution.op_graph_params)
        self._init_nodes(solution)

    def _init_nodes(self, solution: GeneratorSolution):
        self.child_nodes = []
        for i, node in enumerate(self.nodes):
            role, merging_strategy, inplanes, outplanes = self._get_node_details(i, node, solution.op_merge_dist)
            if role == NodeRoles.OUTPUT:
                p = solution.op_dist[:-len(Ops.pooling_ops)]
                op = np.random.choice(Ops.conv_ops, p=p / p.sum())
            else:
                op = np.random.choice(Ops.all_ops, p=solution.op_dist)
            incoming_planes = [self.child_nodes[i].outplanes for i in node.inputs]
            self.child_nodes.append(LogicalOperation(
                op, solution.conv_type, node.inputs, role,
                merging_strategy, inplanes, outplanes, incoming_planes))

    def _get_node_details(self, i, node, merge_dist):
        role = self._get_role_for_node(i)
        merging_strategy = sample_merging_strategy(node.inputs, merge_dist, role)
        outplanes = self.outplanes
        inplanes = compute_input_planes(self.inplanes, merging_strategy, node.inputs, self.child_nodes)
        return role, merging_strategy, inplanes, outplanes

    def _get_param_count(self):
        cost = [node._get_param_count() for node in self.child_nodes]
        return sum(cost)

    def _get_merging_cost(self):
        fixed_costs = 0
        for node in self.child_nodes:
            inplanes = node.inplanes
            for i in node.inputs:
                incoming_outplanes = self.child_nodes[i].outplanes
                if (node.merging_strategy in [EdgeMerge.ATTENTION, EdgeMerge.SUM]) and inplanes != incoming_outplanes:
                    fixed_costs += inplanes * incoming_outplanes
        return fixed_costs

    def __repr__(self):
        """Implement __repr__."""
        self_rep = []
        for i, node in enumerate(self.child_nodes):
            self_rep.append("%s%d: %s" % (self.depth, i, node))
        return "{" + "\n ".join(self_rep) + "} [%s] - %s" % (self.inputs, self.role)


class LogicalCellGraph(LogicalOpGraph):
    """Define the logic midlle-level graph."""

    def __init__(self, stage: int, role: NodeRoles, inputs: List[int], merging_strategy: str,
                 solution: GeneratorSolution, inplanes: int, outplanes: int, incoming_planes: List[int]):
        """Initialize LogicalCellGraph."""
        BasicNode.__init__(self, inputs, role, merging_strategy, inplanes, outplanes, incoming_planes)
        self.depth = "\t"
        self.stage = stage
        self.midlvl_graph = self._init_graph(solution.cell_params)
        self._init_nodes(solution)

    def _get_merging_cost(self):
        cost = [node._get_merging_cost() for node in self.child_nodes]
        return sum(cost)

    def _init_nodes(self, solution: GeneratorSolution):
        self.child_nodes = []
        for i, node in enumerate(self.nodes):
            role = self._get_role_for_node(i)
            merging_strategy = sample_merging_strategy(node.inputs, solution.cell_merge_dist, role)
            outplanes = self.outplanes
            inplanes = compute_input_planes(self.inplanes, merging_strategy, node.inputs, self.child_nodes)
            incoming_planes = [self.child_nodes[i].outplanes for i in node.inputs]
            self.child_nodes.append(
                LogicalOpGraph(role, node.inputs, merging_strategy, solution, inplanes, outplanes, incoming_planes))


class LogicalMasterGraph(LogicalOpGraph):
    """Define the logic top-level graph (i.e. the architecture) from the GeneratorSolution hyperparameters."""

    def __init__(self, solution: GeneratorSolution):
        """Initialize LogicalMasterGraph."""
        if len(solution.stage_ratios) == len(solution.channel_ratios):
            self.child_nodes = []
            self.depth = ""
            self.inputs = []
            self.inplanes = solution.channel_ratios[0]
            self.role = NodeRoles.MASTER
            self.toplvl_graph = self._init_graph(solution.master_params)
            self._init_nodes(solution)
        else:
            raise ValueError("Ratios should have same length.")

    def _get_merging_cost(self):  # TODO fix this, it's unprecise
        cost = [node._get_merging_cost() for node in self.child_nodes]
        return sum(cost)

    def _init_nodes(self, solution: GeneratorSolution):
        self.child_nodes = []
        n_nodes = len(self.nodes)
        stage_values = get_stage_list(n_nodes, solution.stage_ratios)
        for i, node in enumerate(self.nodes):
            role = self._get_role_for_node(i)
            merging_strategy = sample_merging_strategy(node.inputs, solution.master_merge_dist, role)
            stage = stage_values[i]
            outplanes = solution.channel_ratios[stage]
            inplanes = compute_input_planes(self.inplanes, merging_strategy, node.inputs, self.child_nodes)
            incoming_planes = [self.child_nodes[i].outplanes for i in node.inputs]
            self.child_nodes.append(
                LogicalCellGraph(stage, role, node.inputs, merging_strategy, solution, inplanes, outplanes,
                                 incoming_planes))

    def _get_graphs(self):
        return [self.toplvl_graph, self.child_nodes[0].midlvl_graph, self.child_nodes[0].child_nodes[0].bottomlvl_graph]
