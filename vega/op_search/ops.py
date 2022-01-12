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

"""Define the operation."""
import logging
import numpy as np

input_nodes = ["in"]
constant_nodes = ['const1', 'const2', 'const4', 'const8', 'const16', 'const32']
output_nodes = ["out"]
unary_ops = ["rec", 'power2', 'power3', 'power4', 'power6', 'power8', 'power16', 'power32', 'negative', 'sin', 'cos',
             'tan', 'tanh', 'exp', 'log', 'abs']
binary_ops = ["add", "sub", "mul", "div"]

MAX_LEN_OF_FORMULA = 20

constant_values = {'const1': 1,
                   'const2': 2,
                   'const3': 3,
                   'const4': 4,
                   'const5': 5,
                   'const6': 6,
                   'const7': 7,
                   'const8': 8,
                   'const16': 16,
                   'const32': 32,
                   'const.5': 0.5}


def _factorial(n):
    if isinstance(n, int) and n >= 1:
        res = 1
        for i in range(1, n + 1):
            res = res * i
        return res
    else:
        logging.debug("Only int number support factorial.")
        return n


compute_funcs = {'abs': lambda x: np.abs(x),
                 'exp': lambda x: np.exp(x),
                 'log': lambda x: np.log(x),
                 'sin': lambda x: np.sin(x),
                 'cos': lambda x: np.cos(x),
                 'rec': lambda x: 1 / x,
                 'tanh': lambda x: np.tanh(x),
                 'tan': lambda x: np.tan(x),
                 'power2': lambda x: np.power(x, 2),
                 'power3': lambda x: np.power(x, 3),
                 'power4': lambda x: np.power(x, 4),
                 'power5': lambda x: np.power(x, 5),
                 'power6': lambda x: np.power(x, 6),
                 'power7': lambda x: np.power(x, 7),
                 'power8': lambda x: np.power(x, 8),
                 'power16': lambda x: np.power(x, 16),
                 'power32': lambda x: np.power(x, 32),
                 'factorial': lambda x: _factorial(x),
                 'negative': lambda x: -1 * x,
                 # binary ops
                 'add': lambda x, y: x + y,
                 'sub': lambda x, y: x - y,
                 'mul': lambda x, y: x * y,
                 'div': lambda x, y: x / y,
                 }

invalid_conditions = {'in': [],
                      'const1': ['abs', 'rec', 'power2', 'power3', 'div', 'mul', 'factorial'],
                      'const2': ['abs'],
                      'const3': ['abs'],
                      'const4': ['abs'],
                      'const5': ['abs'],
                      'const6': ['abs'],
                      'const7': ['abs'],
                      'const8': ['abs'],
                      'const.5': ['abs', 'factorial'],
                      'exp': ['abs'],
                      'power2': ['abs'],
                      'power4': ['abs'],
                      'power6': ['abs'],
                      'abs': ['abs'],
                      'rec': ['rec'],
                      'negative': ['abs', 'negative'],
                      }

mish_dict = {'in': ['exp', 'mul'],
             'const1': ['add'],
             'exp': ['add'],
             'add': ['log'],
             'log': ['tanh'],
             'tanh': ['mul'],
             'mul': ['out'],
             'out': []
             }

init_dict = {'in': ['out'],
             'out': []
             }


def filter_rules(code):
    """Filter the valid sample."""
    for op_type in invalid_conditions.keys():
        if not _check_downstream(code, op_type, invalid_conditions[op_type]):
            return False
    return True


def _check_downstream(code, op_type, invalid_downstreams):
    for node in code.keys():
        if node.split('-')[0] == op_type:
            if code[node] == invalid_downstreams:
                return False
            for edge in code[node]:
                edge_type = edge.split('-')[0]
                if edge_type in invalid_downstreams:
                    return False

    return True


def _check_op_exist(code, op_type):
    flag = False
    for key in code.keys():
        if key.startswith(op_type):
            flag = True
            break
    return flag
