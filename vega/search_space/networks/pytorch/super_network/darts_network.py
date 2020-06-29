# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Darts Network."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vega.core.common import Config
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks import NetTypes, NetTypesMap, NetworkFactory


@NetworkFactory.register(NetTypes.SUPER_NETWORK)
class DartsNetwork(Network):
    """Base Darts Network of classification.

    :param desc: darts description
    :type desc: Config
    """

    def __init__(self, desc):
        """Init DartsNetwork."""
        super(DartsNetwork, self).__init__()
        self.desc = desc
        self.network = desc.network
        self._C = desc.init_channels
        self._classes = desc.num_classes
        self.input_size = desc.input_size
        self._auxiliary = desc.auxiliary
        self.search = desc.search
        self.drop_path_prob = 0
        if self._auxiliary:
            self._aux_size = desc.aux_size
            self._auxiliary_layer = desc.auxiliary_layer
        self.build_network()

    def build_network(self):
        """Build Darts Network."""
        C_curr = self._network_stems(self.network[0])
        C_prev, C_aux = self._network_cells(self.network[1:], C_curr)
        if not self.search and self._auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_aux, self._classes,
                                                self._aux_size)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self._classes)
        if self.search:
            self._initialize_alphas()

    def _network_stems(self, stem):
        """Build stems part.

        :param stem: stem part of network
        :type stem: torch.nn.Module
        :return: stem's output channel
        :rtype: int
        """
        stem_desc = {'C': self._C, 'stem_multi': 3}
        stem_class = NetworkFactory.get_network(NetTypesMap['block'], stem)
        self.stem = stem_class(Config(stem_desc))
        return self.stem.C_curr

    def _network_cells(self, network_list, C_curr):
        """Build cells part.

        :param network_list: list of cell's name
        :type network_list: list of str
        :param C_curr: input channel of cells
        :type C_curr: int
        """
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
        self.cells = nn.ModuleList()
        reduction_prev = True if C_curr == C_prev else False
        for i, name in enumerate(network_list):
            if name == 'reduce':
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = self.build_cell(name, C_prev_prev, C_prev, C_curr,
                                   reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            multiplier = len(self.desc[name]['concat'])
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if self._auxiliary and i == self._auxiliary_layer:
                C_aux = C_prev
        return C_prev, C_aux if self._auxiliary else None

    def build_cell(self, name, C_prev_prev, C_prev, C_curr, reduction, reduction_prev):
        """Build cell for Darts Network.

        :param name: cell name
        :type name: str
        :param C_prev_prev: channel of previous of previous cell
        :type C_prev_prev: int
        :param C_prev: channel of previous cell
        :type C_prev: int
        :param C_curr: channel of current cell
        :type C_curr: int
        :param reduction: whether to reduce resolution in this cell
        :type reduction: bool
        :param reduction_prev: whether to reduce resolution in previous cell
        :return: object of cell
        :rtype: class type of cell
        """
        cell_desc = {
            'genotype': self.desc[name]['genotype'],
            'steps': self.desc[name]['steps'],
            'concat': self.desc[name]['concat'],
            'C_prev_prev': C_prev_prev,
            'C_prev': C_prev,
            'C': C_curr,
            'reduction': reduction,
            'reduction_prev': reduction_prev,
            'search': self.search
        }
        cell_type = self.desc[name]['type']
        cell_name = self.desc[name]['name']
        cell_class = NetworkFactory.get_network(
            NetTypesMap[cell_type], cell_name)
        return cell_class(Config(cell_desc))

    def _initialize_alphas(self):
        """Initialize architecture parameters."""
        k = len(self.desc.normal.genotype)
        num_ops = len(self.desc.normal.genotype[0][0])
        self.register_buffer('alphas_normal',
                             (1e-3 * torch.randn(k, num_ops)).cuda().requires_grad_())
        self.register_buffer('alphas_reduce',
                             (1e-3 * torch.randn(k, num_ops)).cuda().requires_grad_())
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        """Abstract base function of getting arch parameters."""
        return self._arch_parameters

    @property
    def arch_weights(self):
        """Get arch weights."""
        weights_normal = F.softmax(
            self.alphas_normal, dim=-1).data.cpu().numpy()
        weights_reduce = F.softmax(
            self.alphas_reduce, dim=-1).data.cpu().numpy()
        return [weights_normal, weights_reduce]

    def set_arch_parameters(self, weights):
        """Set arch parameters."""
        raise NotImplementedError

    def forward(self, input):
        """Forward function of Darts Network."""
        s0, s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if self.search:
                if self.desc.network[i + 1] == 'reduce':
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            else:
                weights = None
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            if not self.search:
                if self._auxiliary and i == self._auxiliary_layer:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self._auxiliary and not self.search:
            return logits, logits_aux
        else:
            return logits


class AuxiliaryHead(nn.Module):
    """Auxiliary Head of Network.

    :param C: input channels
    :type C: int
    :param num_classes: numbers of classes
    :type num_classes: int
    :param input_size: input size
    :type input_size: int
    """

    def __init__(self, C, num_classes, input_size):
        """Init AuxiliaryHead."""
        super(AuxiliaryHead, self).__init__()
        s = input_size - 5
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=s, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        """Forward function of Auxiliary Head."""
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
