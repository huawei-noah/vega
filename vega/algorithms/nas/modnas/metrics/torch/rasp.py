# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""RASP-based metrics."""
from ..base import MetricsBase
from modnas.registry.metrics import register, build
from modnas.arch_space.mixed_ops import MixedOp
try:
    import rasp
    import rasp.frontend as F
except ImportError:
    rasp = None


@register
class RASPStatsMetrics(MetricsBase):
    """RASP node statistics metrics class."""

    def __init__(self, item):
        super().__init__()
        self.item = item

    def __call__(self, node):
        """Return metrics output."""
        return node[self.item]


@register
class RASPTraversalMetrics(MetricsBase):
    """RASP model traversal metrics class."""

    def __init__(self,
                 input_shape,
                 metrics,
                 compute=True,
                 timing=False,
                 device='cuda',
                 mixed_only=False,
                 keep_stats=True,
                 traversal_type='tape_nodes'):
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

    def _traverse_tape_nodes(self, node):
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

    def _traverse_tape_leaves(self, node):
        ret = 0
        for cur_node in node.tape.items_all:
            if cur_node in self.excluded:
                continue
            n_ret = self.metrics(cur_node)
            if n_ret is None:
                n_ret = 0
            ret += n_ret
        return ret

    def _stat(self, module, input_shape):
        """Run profiling on given module."""
        if self.eval_compute:
            F.hook_compute(module)
        if self.eval_timing:
            F.hook_timing(module)
        F.run(module, F.get_random_data(input_shape), self.device)
        F.unhook_compute(module)
        F.unhook_timing(module)

    def __call__(self, net):
        """Return metrics output."""
        self.excluded.clear()
        root = F.get_stats_node(net)
        if root is None:
            root = F.reg_stats_node(net)
            self._stat(net, self.input_shape)
        mt = 0
        for m in net.modules():
            if not isinstance(m, MixedOp):
                continue
            mixop_node = F.get_stats_node(m)
            self.excluded.add(mixop_node)
            assert mixop_node['in_shape'] is not None
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
