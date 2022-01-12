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

"""RASP-based metrics."""

from typing import List, Any, Union
from torch.nn.modules.module import Module
from modnas.registry.metrics import register, build
from modnas.arch_space.mixed_ops import MixedOp
from modnas.registry import SPEC_TYPE
from ..base import MetricsBase

try:
    import rasp
    import rasp.frontend as F
    from rasp.profiler.tree import StatTreeNode
except ImportError:
    rasp = None
    StatTreeNode = None


@register
class RASPStatsMetrics(MetricsBase):
    """RASP node statistics metrics class."""

    def __init__(self, item: str) -> None:
        super().__init__()
        self.item = item

    def __call__(self, node: StatTreeNode) -> int:
        """Return metrics output."""
        return node[self.item]


@register
class RASPTraversalMetrics(MetricsBase):
    """RASP model traversal metrics class."""

    def __init__(self,
                 input_shape: List[int],
                 metrics: SPEC_TYPE,
                 compute: bool = True,
                 timing: bool = False,
                 device: str = 'cuda',
                 mixed_only: bool = False,
                 keep_stats: bool = True,
                 traversal_type: str = 'tape_nodes') -> None:
        super().__init__()
        if rasp is None:
            raise ValueError('package RASP is not found')
        self.metrics = build(metrics)
        self.eval_compute = compute
        self.eval_timing = timing
        self.input_shape = input_shape
        self.device = device
        self.mixed_only = mixed_only
        self.keep_stats = keep_stats
        if traversal_type == 'tape_leaves':
            self.traverse = self._traverse_tape_leaves
        elif traversal_type == 'tape_nodes':
            self.traverse = self._traverse_tape_nodes
        else:
            raise ValueError('invalid traversal type')
        self.excluded = set()

    def _traverse_tape_nodes(self, node: StatTreeNode) -> Union[float, int]:
        ret = 0
        if node in self.excluded:
            return ret
        if node.num_children == 0:
            ret = self.metrics(node)
            return ret
        if node.tape is None:
            return ret
        for cur_node in node.tape.items:
            if cur_node['cand_type']:
                n_ret = self.metrics(cur_node)
            else:
                n_ret = self._traverse_tape_nodes(cur_node)
            if n_ret is None:
                n_ret = 0
            ret += n_ret
        return ret

    def _traverse_tape_leaves(self, node: StatTreeNode) -> Union[float, int]:
        ret = 0
        for cur_node in node.tape.items_all:
            if cur_node in self.excluded:
                continue
            n_ret = self.metrics(cur_node)
            if n_ret is None:
                n_ret = 0
            ret += n_ret
        return ret

    def _stat(self, module: Module, input_shape: List[int]) -> None:
        """Run profiling on given module."""
        if self.eval_compute:
            F.hook_compute(module)
        if self.eval_timing:
            F.hook_timing(module)
        F.run(module, F.get_random_data(input_shape), self.device)
        F.unhook_compute(module)
        F.unhook_timing(module)

    def __call__(self, net: Module) -> Any:
        """Return metrics output."""
        self.excluded.clear()
        root = F.get_stats_node(net)
        if root is None:
            root = F.reg_stats_node(net)
            self._stat(net, self.input_shape)
        mt = 0.
        for m in net.modules():
            if not isinstance(m, MixedOp):
                continue
            mixop_node = F.get_stats_node(m)
            self.excluded.add(mixop_node)
            if mixop_node['in_shape'] is None:
                raise ValueError('Inshape of mixop is None.')
            mixop_mt = 0
            m_in, _ = mixop_node['in_shape'], mixop_node['out_shape']
            for p, (pn, op) in zip(m.prob(), m.named_candidates()):
                if not p:
                    continue
                subn = F.get_stats_node(op)
                if subn['cand_type'] is None:
                    subn['cand_type'] = pn
                if subn['compute_updated'] is None:
                    if subn['in_shape'] is None:
                        subn['in_shape'] = m_in
                    self._stat(subn.module, subn['in_shape'])
                    subn['compute_updated'] = True
                subn_mt = self.metrics(subn)
                if subn_mt is None:
                    subn_mt = self.traverse(subn)
                if subn_mt is None:
                    self.logger.warning('unresolved node: {} type: {}'.format(subn['name'], subn['type']))
                    subn_mt = 0
                mixop_mt = mixop_mt + subn_mt * p
            mt += mixop_mt
        if not self.mixed_only:
            mt = mt + self.traverse(root)
        if not self.keep_stats:
            F.unreg_stats_node(net)
        return mt


@register
class RASPRootMetrics(MetricsBase):
    """RASP root node metrics class."""

    def __init__(self, input_shape, metrics, compute=True, timing=False, device=None):
        super().__init__()
        if rasp is None:
            raise ValueError('package RASP is not found')
        self.metrics = build(metrics)
        self.eval_compute = compute
        self.eval_timing = timing
        self.input_shape = input_shape
        self.device = device

    def __call__(self, net):
        """Return metrics output."""
        root = F.get_stats_node(net)
        if root is None:
            root = F.reg_stats_node(net)
            if self.eval_compute:
                F.hook_compute(net)
            if self.eval_timing:
                F.hook_timing(net)
            inputs = F.get_random_data(self.input_shape)
            F.run(net, inputs, self.device)
            F.unhook_compute(net)
            F.unhook_timing(net)
        return self.metrics(root)
