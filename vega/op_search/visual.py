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
import subprocess


def _write_graph(dict):
    with open("test.dot", "w") as file:
        file.write("digraph {} {{\n".format("graph_name"))
        for node, edges in dict.items():
            node = node.replace('-', '_')
            for edge in edges:
                edge = edge.replace('-', '_')
                file.write("{}->{};\n".format(node, edge))
        file.write("}\n")


def visual_dag(dict, save_name):
    """Visulize the compute graph."""
    _write_graph(dict)
    subprocess.call(f"dot -Tpdf test.dot -o {save_name}.pdf".split(" "))
    subprocess.call(f"dot -Tpng test.dot -o {save_name}.png".split(" "))
