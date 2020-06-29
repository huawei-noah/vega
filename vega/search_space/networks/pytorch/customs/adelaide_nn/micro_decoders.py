# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Micro encoder."""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_factory import conv_bn_relu, conv3x3, OPS


class AggregateCell(nn.Module):
    """Aggregate two cells and sum or concat them up."""

    def __init__(self, size_1, size_2, agg_size, pre_transform=True, concat=False):
        """Construct AggregateCell.

        :param size_1: channel of first input
        :param size_2: channel of second input
        :param agg_size: channel of aggregated tensor
        :param pre_transform: whether to do a transform on two inputs
        :param concat: concat the result if set to True, otherwise add the result
        """
        super(AggregateCell, self).__init__()
        self.pre_transform = pre_transform
        self.concat = concat
        if self.pre_transform:
            self.branch_1 = conv_bn_relu(size_1, agg_size, 1, 1, 0)
            self.branch_2 = conv_bn_relu(size_2, agg_size, 1, 1, 0)
        if self.concat:
            self.conv1x1 = conv_bn_relu(agg_size * 2, agg_size, 1, 1, 0)

    def forward(self, x1, x2):
        """Do an inference on AggregateCell.

        :param x1: first input
        :param x2: second input
        :return: output
        """
        if self.pre_transform:
            x1 = self.branch_1(x1)
            x2 = self.branch_2(x2)
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode='bilinear', align_corners=True)(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode='bilinear', align_corners=True)(x1)
        if self.concat:
            return self.conv1x1(torch.cat([x1, x2], 1))
        else:
            return x1 + x2


class ContextualCell_v1(nn.Module):
    """New contextual cell design."""

    def __init__(self, op_names, config, inp, repeats=1, concat=False):
        """Construct ContextualCell_v1 class.

        :param op_names: list of operation indices
        :param config: list of config numbers
        :param inp: input channel
        :param repeats: number of repeated times
        :param concat: concat the result if set to True, otherwise add the result
        """
        super(ContextualCell_v1, self).__init__()
        self._ops = nn.ModuleList()
        self._pos = []
        self._collect_inds = [0]
        self._pools = ['x']
        for ind, op in enumerate(config):
            # first op is always applied on x
            if ind == 0:
                pos = 0
                op_id = op
                self._collect_inds.remove(pos)
                op_name = op_names[op_id]
                self._ops.append(OPS[op_name](inp, 1, True, repeats))  # turn-off scaling in batch norm
                self._pos.append(pos)
                self._collect_inds.append(ind + 1)
                self._pools.append('{}({})'.format(op_name, self._pools[pos]))
            else:
                pos1, pos2, op_id1, op_id2 = op
                # drop op_id from loose ends
                for ind2, (pos, op_id) in enumerate(zip([pos1, pos2], [op_id1, op_id2])):
                    if pos in self._collect_inds:
                        self._collect_inds.remove(pos)
                    op_name = op_names[op_id]
                    self._ops.append(OPS[op_name](inp, 1, True, repeats))  # turn-off scaling in batch norm
                    self._pos.append(pos)
                    # self._collect_inds.append(ind * 3 + ind2 - 1) # Do not collect intermediate
                    self._pools.append('{}({})'.format(op_name, self._pools[pos]))
                # summation
                op_name = 'sum'
                self._ops.append(AggregateCell(size_1=None, size_2=None, agg_size=inp, pre_transform=False,
                                               concat=concat))  # turn-off convbnrelu
                self._pos.append([ind * 3 - 1, ind * 3])
                self._collect_inds.append(ind * 3 + 1)
                self._pools.append('{}({},{})'.format(op_name, self._pools[ind * 3 - 1], self._pools[ind * 3]))

    def forward(self, x):
        """Do an inference on ContextualCell_v1.

        :param x: input tensor
        :return: output tensor
        """
        feats = [x]
        for pos, op in zip(self._pos, self._ops):
            if isinstance(pos, list):
                assert len(pos) == 2, "Two ops must be provided"
                feats.append(op(feats[pos[0]], feats[pos[1]]))
            else:
                feats.append(op(feats[pos]))
        out = 0
        for i in self._collect_inds:
            out += feats[i]
        return out

    def prettify(self):
        """Format printing.

        :return: formatted string of the module
        """
        return ' + '.join(self._pools[i] for i in self._collect_inds)


class MergeCell(nn.Module):
    """Pass two inputs through ContextualCell, and aggregate their results."""

    def __init__(self, op_names, ctx_config, conn, inps, agg_size, ctx_cell, repeats=1,
                 cell_concat=False):
        """Construct MergeCell class.

        :param op_names: list of operation indices
        :param ctx_config: list of config numbers
        :param conn: list of two indices
        :param inps: channel of first and second input
        :param agg_size: number of aggregation channel
        :param ctx_cell: ctx module
        :param repeats: number of repeats
        :param cell_concat: whether to concat or add cells
        """
        super(MergeCell, self).__init__()
        self.index_1, self.index_2 = conn
        inp_1, inp_2 = inps
        self.op_1 = ctx_cell(op_names, ctx_config, inp_1, repeats=repeats)
        self.op_2 = ctx_cell(op_names, ctx_config, inp_2, repeats=repeats)
        self.agg = AggregateCell(inp_1, inp_2, agg_size, concat=cell_concat)

    def forward(self, x1, x2):
        """Do an inference on MergeCell.

        :param x1: input tensor 1
        :param x2: input tensor 2
        :return: output tensor
        """
        x1 = self.op_1(x1)
        x2 = self.op_2(x2)
        return self.agg(x1, x2)

    def prettify(self):
        """Format printing.

        :return: formatted string of the module
        """
        return self.op_1.prettify()


class MicroDecoder(nn.Module):
    """Parent class for MicroDecoders."""

    def __init__(self, op_names, backbone_out_sizes, num_classes, config, agg_size=64, num_pools=4,
                 ctx_cell=ContextualCell_v1, aux_cell=False, sep_repeats=1, agg_concat=False, cell_concat=False,
                 **params):
        """Construct MicroDecoder class.

        :param op_names: list of operation candidate names
        :param backbone_out_sizes: backbone output channels
        :param num_classes: number of classes
        :param config: config list
        :param agg_size: number of channels in aggregation cells
        :param num_pools: number of pools
        :param ctx_cell: ctx module
        :param aux_cell: aux cells
        :param sep_repeats: number of repeats
        :param agg_concat: whether to concat or add agg results
        :param cell_concat: whether to concat or add cells
        :param params: other parameters
        """
        super(MicroDecoder, self).__init__()
        cells = []
        aux_clfs = []
        self.aux_cell = aux_cell
        self.collect_inds = []
        # for description of the structure
        self.pool = ['l{}'.format(i + 1) for i in range(num_pools)]
        self.info = []
        self.agg_size = agg_size
        self.agg_concat = agg_concat
        self.op_names = op_names

        # NOTE: bring all outputs to the same size
        for out_idx, size in enumerate(backbone_out_sizes):
            setattr(self,
                    'adapt{}'.format(out_idx + 1),
                    conv_bn_relu(size, agg_size, 1, 1, 0, affine=True))
            backbone_out_sizes[out_idx] = agg_size

        if sys.version_info[0] < 3:
            backbone_out_sizes = list(backbone_out_sizes)
        else:
            backbone_out_sizes = backbone_out_sizes.copy()
        cell_config, conns = config
        self.conns = conns
        self.ctx = cell_config
        self.repeats = sep_repeats
        self.collect_inds = []
        self.ctx_cell = ctx_cell
        for block_idx, conn in enumerate(conns):
            for ind in conn:
                if ind in self.collect_inds:
                    # remove from outputs if used by pool cell
                    self.collect_inds.remove(ind)
            ind_1, ind_2 = conn
            cells.append(MergeCell(self.op_names, cell_config, conn,
                                   (backbone_out_sizes[ind_1], backbone_out_sizes[ind_2]),
                                   agg_size,
                                   ctx_cell, repeats=sep_repeats,
                                   cell_concat=cell_concat))
            aux_clfs.append(nn.Sequential())
            if self.aux_cell:
                aux_clfs[block_idx].add_module('aux_cell',
                                               ctx_cell(self.op_names, self.ctx, agg_size, repeats=sep_repeats))
            aux_clfs[block_idx].add_module('aux_clf', conv3x3(agg_size, num_classes, stride=1, bias=True))
            self.collect_inds.append(block_idx + num_pools)
            backbone_out_sizes.append(agg_size)
            # for description
            self.pool.append('({} + {})'.format(self.pool[ind_1], self.pool[ind_2]))
        self.cells = nn.ModuleList(cells)
        self.aux_clfs = nn.ModuleList(aux_clfs)
        self.pre_clf = conv_bn_relu(agg_size * (len(self.collect_inds) if self.agg_concat else 1),
                                    agg_size, 1, 1, 0)
        self.conv_clf = conv3x3(agg_size, num_classes, stride=1, bias=True)
        self.info = ' + '.join(self.pool[i] for i in self.collect_inds)
        self.num_classes = num_classes

    def _reset_clf(self, num_classes):
        if num_classes != self.num_classes:
            del self.conv_clf
            self.conv_clf = conv3x3(self.agg_size, num_classes, stride=1, bias=True).cuda()
            del self.aux_clfs
            self.aux_clfs = nn.ModuleList()
            for block_idx, conn in enumerate(self.conns):
                self.aux_clfs.append(nn.Sequential())
                if self.aux_cell:
                    self.aux_clfs[block_idx].add_module('aux_cell',
                                                        self.ctx_cell(self.op_names, self.ctx, self.agg_size,
                                                                      self.repeats))
                self.aux_clfs[block_idx].add_module('aux_clf', conv3x3(self.agg_size, num_classes, stride=1, bias=True))
            self.aux_clfs = self.aux_clfs.cuda()
            self.num_classes = num_classes

    def prettify(self, n_params):
        """Format printing.

        :param n_params: number parameters
        :return: formatted printing of the module
        """
        header = '#PARAMS\n\n {:3.2f}M'.format(n_params / 1e6)
        ctx_desc = '#Contextual:\n' + self.cells[0].prettify()
        conn_desc = '#Connections:\n' + self.info
        return header + '\n\n' + ctx_desc + '\n\n' + conn_desc

    def forward(self, x):
        """Do an inference on MicroDecoder.

        :param x: input tensor
        :return: output tensor
        """
        x = list(x)
        for out_idx in range(len(x)):
            x[out_idx] = getattr(self, 'adapt{}'.format(out_idx + 1))(x[out_idx])
        for cell, aux_clf, conn in zip(self.cells, self.aux_clfs, self.conns):
            cell_out = cell(x[conn[0]], x[conn[1]])
            x.append(cell_out)
        out = x[self.collect_inds[0]]
        for i in range(1, len(self.collect_inds)):
            collect = x[self.collect_inds[i]]
            if out.size()[2] > collect.size()[2]:
                # upsample collect
                collect = nn.Upsample(size=out.size()[2:], mode='bilinear', align_corners=True)(collect)
            elif collect.size()[2] > out.size()[2]:
                out = nn.Upsample(size=collect.size()[2:], mode='bilinear', align_corners=True)(out)
            if self.agg_concat:
                out = torch.cat([out, collect], 1)
            else:
                out += collect

        out = F.relu(out)
        out = self.pre_clf(out)
        out = self.conv_clf(out)
        return out
