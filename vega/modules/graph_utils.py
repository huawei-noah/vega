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

"""Graph utils to Modules."""
import logging
from collections import OrderedDict
import re
from vega.common.dag import DAG
from .nodes import Node
from .nodes import Sequential, Add


def graph2desc(graph):
    """Parse graph to Network Desc."""
    ops = get_ops_from_graph(graph)
    dag = ops2dag(ops)
    desc = Dag2Module(dag, ops).parse()
    logging.info("Success to create desc form graph")
    logging.debug(desc)
    return desc


def get_ops_from_graph(graph):
    """Get ops from graph and convert Node."""
    ops = graph.get_operations()
    merged_ops = OrderedDict()
    for op in ops:
        support_ops_name = None
        scope_name = None
        for _support_ops_name in Node.__support_ops__:
            if re.findall(_support_ops_name, op.name) or op.name.endswith(_support_ops_name):
                support_ops_name = _support_ops_name
        if op.type in Node.__support_ops_types__:
            support_ops_name = op.name
            scope_name = op.name
        if not support_ops_name:
            continue
        scope_name = scope_name or op.name[:op.name.index(support_ops_name)]
        all_ops_in_scope = [op for op in ops if op.name.startswith(scope_name + '/') or op.name == scope_name]
        if not all_ops_in_scope and len(op.inputs) == 0:
            continue
        inputs = op.inputs
        if inputs and inputs[0].op.type == 'Identity':
            all_ops_in_scope.insert(0, inputs)
            inputs = op.inputs[0].op.inputs
        type_name = op.type if op.type != 'Const' else op.name.split('/')[-1]
        if op.type == 'Const':
            continue
        node = Node(inputs=inputs, outputs=op.outputs[0], type_name=type_name, op_name=op.name,
                    op_list=all_ops_in_scope)
        merged_ops[node.op_name] = node
        if op.name.endswith('Softmax'):
            break
    return merged_ops


def ops2dag(merged_ops):
    """Load ops dict into dag."""
    dag = DAG()
    dot = DagGraphVisual()
    dot.node(name='root', label='root')
    outs = {op['outputs'].name: op for name, op in merged_ops.items() if op['outputs'] is not None}
    outs = {k.replace('Conv2D:0', 'BiasAdd:0'): v for k, v in outs.items()}
    for name, node in merged_ops.items():
        inps = node['inputs']
        pre_node_name = 'root'
        dag.add_node(name)
        dot.node(name=name, label=name)
        if inps is not None:
            for inp in inps:
                pre_node = outs.get(inp.name)
                if pre_node is not None:
                    pre_node_name = pre_node.op_name
                    dag.add_edge(pre_node_name, name)
                    dot.edge(pre_node_name, name)
        else:
            dag.add_edge(pre_node_name, name)
            dot.edge(pre_node_name, name)
    dot.show()
    return dag


class Dag2Module(object):
    """Parse dag to module desc."""

    def __init__(self, dag, ops):
        self.g = dag.nodes
        self.ops = ops
        self.e = self._convert_edge_list()
        self.muti_edges = [k for k, v in self.g.items() if len(v) > 1]
        self.muti_node = [k for k, v in self.e.items() if len(v) > 1]

    def parse(self):
        """Parse graph to Sequential desc."""
        result = Sequential()
        while self.g:
            k, v = self.g.popitem(False)
            if self._is_connection_node(k):
                continue
            result.append(self.ops.get(k))
            if self._is_branch_node(k):
                branch_seq = []
                for _ in v:
                    seq = self._parse_branch_graph(self.g)
                    branch_seq.append(seq)
                branch = Add(*branch_seq)
                result.append(branch)
        return result.to_json()

    def _convert_edge_list(self):
        e = OrderedDict()
        for node, edge in self.g.items():
            for v in edge:
                e[v] = [node] if v not in e else e[v] + [node]
        return e

    def _is_branch_node(self, node):
        return node in self.muti_edges

    def _is_connection_node(self, node):
        if not node:
            return False
        if isinstance(node, set):
            return node.issubset(self.muti_node)
        else:
            return node in self.muti_node

    def _parse_branch_graph(self, g):
        seq = Sequential()
        k, v = g.popitem(False)
        if self._is_connection_node(k):
            return seq
        seq.append(self.ops.get(k))
        while not self._is_connection_node(v):
            k, v = g.popitem(False)
            seq.append(self.ops.get(k))
        return seq


class DagGraphVisual(object):
    """Dag Graph Visual."""

    def __init__(self, show_dag=False):
        if show_dag:
            from graphviz import Digraph
            self.dot = Digraph(name="Root", comment="network", format="png")
        else:
            self.dot = None

    def node(self, name, label):
        """Add node to dot."""
        if self.dot:
            self.dot.node(name=name, label=label, color='green')

    def edge(self, pre_node_name, name):
        """Add edge to dot."""
        if self.dot:
            self.dot.edge(pre_node_name, name)

    def show(self):
        """Show dot."""
        if self.dot:
            self.dot.view()
