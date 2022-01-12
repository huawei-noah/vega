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

"""These are some tool function."""
import logging
import numpy as np
from .ops import constant_values, compute_funcs


def get_upstreams(dag, node):
    """Return the upstream nodes of this node."""
    nodes = dag.nodes
    res = []
    for key, value in nodes.items():
        if node in value:
            res.append(key)
    return res


def dag2compute(dag, input_data):
    """Calculate the output according to the dag."""
    wait_cal_list = []
    has_cal_list = []
    res = {}
    all_nodes = dag.topological_sort()
    for node in all_nodes:
        values = {"status": 0, "data": None}
        res[node] = values

    logging.debug("The calculation nodes is :{}.".format(dag.nodes))
    for node in all_nodes:
        if node.split('-')[0] == 'in':
            cal_res = input_data
            res[node]['data'] = cal_res
            has_cal_list.append(node)
        elif node.startswith('const'):
            constant_type = node.split('-')[0]
            cal_res = constant_values[constant_type]
            res[node]['data'] = cal_res
            has_cal_list.append(node)
        elif node == 'out':
            upstream_node = get_upstreams(dag, node)[0]
            cal_res = res[upstream_node]['data']
            res[node]['data'] = cal_res
            has_cal_list.append(node)
        else:
            upstream_nodes = get_upstreams(dag, node)
            upstream_done = True
            inputs = []
            for upstream_node in upstream_nodes:
                if res[upstream_node]['data'] is None:
                    upstream_done = False
                    wait_cal_list.append(dag.next_nodes(node=upstream_node))
                else:
                    inputs.append(res[upstream_node]['data'])

            if upstream_done:
                node_type = node.split('-')[0]
                if len(inputs) == 1:
                    if node_type in ['rec'] and not isinstance(inputs[0], np.ndarray) and inputs[0] == 0:
                        logging.debug("To avoid zero div, y will be added a smller number.")
                        inputs[0] = inputs[0] + 1e-15
                    cal_res = compute_funcs[node_type](inputs[0])
                elif len(inputs) == 2:
                    if node_type in ['div'] and not isinstance(inputs[1], np.ndarray) and inputs[1] == 0:
                        logging.debug("To avoid zero div, y will be added a smller number.")
                        inputs[1] = inputs[1] + 1e-15
                    cal_res = compute_funcs[node_type](inputs[0], inputs[1])
                else:
                    raise ValueError("The op {} only support one or two inputs, but got {}.".format(node, len(inputs)))
                res[node]['data'] = cal_res
                has_cal_list.append(node)

    return res['out']['data']


def cal_mish(x):
    """Calculate mish."""
    return x * np.tanh(np.log(np.exp(x) + 1))


def cal_gelu(x):
    """Calculate gelu."""
    w = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + np.tanh(w * (x + 0.044715 * x * x * x)))


def cal_tanh(x):
    """Calculate tanh."""
    return np.tanh(x)


def cal_sqrt(x):
    """Calculate sqrt."""
    return np.sqrt(x)


def cal_softplus(x):
    """Calculate softplus."""
    return np.log(np.exp(x) + 1)


def cal_error_threshold(a, b):
    """Calculate error threshold."""
    threshold = np.abs(a - b) / (np.abs(b) + 1)
    return threshold.max()


def is_close(arr1, arr2):
    """Check if two array is close or not."""
    return isinstance(arr1, np.ndarray) and len(arr1) == len(arr2) and np.allclose(arr1, arr2, atol=1, rtol=1)


def is_close_l2(arr1, arr2):
    """Check if two array is close or not."""
    return len(arr1) == len(arr2) and np.allclose(arr1, arr2, atol=1e-4, rtol=1e-4)


def cal_fitness(output, real_value):
    """Calculate the relative error of two array."""
    diff = output - real_value
    r_error = np.abs(diff) / np.abs(real_value)
    mean_error = np.mean(r_error)
    return mean_error
