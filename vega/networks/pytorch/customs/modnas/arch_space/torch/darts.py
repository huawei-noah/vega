# -*- coding:utf-8 -*-

# This file is adapted from the pt.darts library at
# https://github.com/khanrc/pt.darts

# 2020.6.29-Changed for Modular-NAS search space.
#         Huawei Technologies Co., Ltd. <linyunfeng5@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.

"""Cell-based architecture in DARTS."""

from functools import partial
import torch.nn as nn
from modnas.registry.construct import DefaultMixedOpConstructor
from modnas.registry.construct import register as register_constructor
from modnas.registry.arch_space import register
from modnas.registry import parse_spec, to_spec
from ..layers import DAGLayer
from ..ops import FactorizedReduce, StdConv
from ..slot import Slot


class PreprocLayer(StdConv):
    """Preprocessing layer."""

    def __init__(self, C_in, C_out):
        super(PreprocLayer, self).__init__(C_in, C_out, 1, 1, 0)


class AuxiliaryHead(nn.Module):
    """Auxiliary head in 2/3 place of network to let the gradient flow well."""

    def __init__(self, input_size, C, n_classes):
        # assuming input size 7x7 or 8x8
        if input_size not in [7, 8]:
            raise ValueError('unknown input_size: %s' % repr(input_size))
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size - 5, padding=0, count_include_pad=False),  # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False),  # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        """Return module output."""
        out = self.net(x)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


class DARTSLikeNet(nn.Module):
    """Cell-based architecture in DARTS."""

    def __init__(self, chn_in, chn, n_classes, layers, shared_a,
                 channel_multiplier, auxiliary, cell_cls, cell_kwargs):
        super().__init__()
        self.chn_in = chn_in
        self.chn = chn
        self.aux_pos = 2 * layers // 3 if auxiliary else -1
        self.shared_a = shared_a

        chn_cur = self.chn * channel_multiplier
        self.stem0, self.stem1, reduction_p = self.get_stem(chn_in, chn_cur)

        chn_pp, chn_p, chn_cur = chn_cur, chn_cur, self.chn

        self.cells = nn.ModuleList()
        self.cell_group = [[], [] if shared_a else []]
        for i in range(layers):
            stride = 1
            cell_kwargs['preproc'] = (PreprocLayer, PreprocLayer)
            if i in [layers // 3, 2 * layers // 3]:
                reduction = True
                stride = 2
                chn_cur *= 2
            else:
                reduction = False
            if reduction_p:
                cell_kwargs['preproc'] = (FactorizedReduce, PreprocLayer)
            cell_kwargs['chn_in'] = (chn_pp, chn_p)
            cell_kwargs['stride'] = stride
            cell_kwargs['name'] = 'reduce' if reduction else 'normal'
            cell_kwargs['edge_kwargs']['_chn_in'] = (chn_cur, )
            cell = cell_cls(**cell_kwargs)
            self.cells.append(cell)
            self.cell_group[1 if reduction else 0].append(cell)
            chn_out = chn_cur * cell_kwargs['n_nodes']
            chn_pp, chn_p = chn_p, chn_out
            reduction_p = reduction
            if i == self.aux_pos:
                fm_size = 32
                self.aux_head = AuxiliaryHead(fm_size // 4, chn_p, n_classes)

        self.conv_last = nn.Sequential(nn.AdaptiveAvgPool2d(1), )
        self.fc = nn.Linear(chn_p, n_classes)

    def get_stem(self, chn_in, chn_cur):
        """Return stem layers."""
        stem = nn.Sequential(
            nn.Conv2d(chn_in, chn_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(chn_cur),
        )
        return stem, lambda x: x, False

    def forward(self, x):
        """Return model output."""
        s0 = self.stem0(x)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell([s0, s1])
            if i == self.aux_pos:
                self.aux_out = self.aux_head(s1)
        y = self.conv_last(s1)
        y = y.view(y.size(0), -1)  # flatten
        return self.fc(y)

    def forward_aux(self, x):
        """Return model auxiliary output."""
        if not self.training or self.aux_pos == -1:
            return None
        return self.aux_out

    def build_from_arch_desc(self, desc, *args, **kwargs):
        """Build network from archdesc."""
        if len(self.cell_group) == len(desc):
            for cells, g in zip(self.cell_group, desc):
                for c in cells:
                    c.build_from_arch_desc(g, *args, **kwargs)
        else:
            raise ValueError('Failed to build network from archdesc.')

    def to_arch_desc(self, k=2):
        """Return archdesc from parameters."""
        desc = []
        for cells in self.cell_group:
            desc.append(cells[0].to_arch_desc(k))
        return desc

    def dags(self):
        """Return generator over dag layers."""
        for cell in self.cells:
            yield cell


@register_constructor
class DARTSSearchConstructor(DefaultMixedOpConstructor):
    """DARTS Search Constructor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_map = {}

    def convert(self, slot):
        """Convert Slot to mixed operator."""
        arch_params = self.param_map.get(slot.name, None)
        old_conf = self.mixed_op_conf
        reg_id, args = parse_spec(old_conf)
        if arch_params is not None:
            args['arch_param'] = arch_params
        self.mixed_op_conf = to_spec(reg_id, args)
        ent = super().convert(slot)
        self.mixed_op_conf = old_conf
        args.pop('arch_param', None)
        if slot.name not in self.param_map:
            self.param_map[slot.name] = ent.arch_param
        return ent


class ImageNetDARTSLikeNet(DARTSLikeNet):
    """Return architecture built from config."""

    def get_stem(self, chn_in, chn_cur):
        """Return stem layers."""
        stem0 = nn.Sequential(
            nn.Conv2d(chn_in, chn_cur // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(chn_cur // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(chn_cur // 2, chn_cur, 3, 2, 1, bias=False),
            nn.BatchNorm2d(chn_cur),
        )
        stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(chn_cur, chn_cur, 3, 2, 1, bias=False),
            nn.BatchNorm2d(chn_cur),
        )
        return stem0, stem1, True


def build_from_config(darts_cls=DARTSLikeNet, **kwargs):
    """Return architecture built from config."""
    n_nodes = 4
    if 'nodes' in kwargs:
        n_nodes = kwargs.pop('nodes')
    darts_kwargs = {
        'cell_cls': DAGLayer,
        'cell_kwargs': {
            'chn_in': None,
            'chn_out': None,
            'stride': None,
            'n_nodes': n_nodes,
            'allocator': 'ReplicateAllocator',
            'merger_state': 'SumMerger',
            'merger_out': 'ConcatMerger',
            'enumerator': 'CombinationEnumerator',
            'preproc': None,
            'edge_cls': Slot,
            'edge_kwargs': {
                '_chn_in': None,
                '_chn_out': None,
                '_stride': None,
            },
        },
    }
    darts_kwargs.update(kwargs)
    return darts_cls(**darts_kwargs)


register(build_from_config, 'DARTS')
register(partial(build_from_config, ImageNetDARTSLikeNet), 'ImageNet-DARTS')
