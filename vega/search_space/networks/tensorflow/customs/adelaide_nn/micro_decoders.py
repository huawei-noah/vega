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
from .layer_factory import Conv, OPS, resize_bilinear
import tensorflow as tf


class AggregateCell(object):
    """Aggregate two cells and sum or concat them up."""

    def __init__(self, agg_size, pre_transform=True, concat=False, data_format='channels_first'):
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
        self.agg_size = agg_size
        self.concat = concat
        self.data_format = data_format

    def __call__(self, x1, x2, training):
        """Do an inference on AggregateCell.

        :param x1: first input
        :param x2: second input
        :return: output
        """
        if self.pre_transform:
            x1 = Conv(self.agg_size, 1, 1, data_format=self.data_format)(x1, training)
            x2 = Conv(self.agg_size, 1, 1, data_format=self.data_format)(x2, training)
        if tuple(x1.get_shape()[2:]) > tuple(x2.get_shape()[2:]):
            x2 = resize_bilinear(x2, x1.get_shape()[2:])
        elif tuple(x1.get_shape()[2:]) < tuple(x2.get_shape()[2:]):
            x1 = resize_bilinear(x1, x2.get_shape()[2:])
        if self.concat:
            return Conv(self.agg_size, 1, 1, data_format=self.data_format)(tf.concat([x1, x2], 1), training)
        else:
            return x1 + x2


class ContextualCell_v1(object):
    """New contextual cell design."""

    def __init__(self, op_names, config, inp, repeats=1, concat=False, data_format='channels_first'):
        """Construct ContextualCell_v1 class.

        :param op_names: list of operation indices
        :param config: list of config numbers
        :param inp: input channel
        :param repeats: number of repeated times
        :param concat: concat the result if set to True, otherwise add the result
        """
        super(ContextualCell_v1, self).__init__()
        self.op_names = op_names
        self.config = config
        self.inp = inp
        self.repeats = repeats
        self.concat = concat
        self.data_format = data_format

    def build(self):
        """Build contextual cell."""
        self._ops = []
        self._pos = []
        self._collect_inds = [0]
        self._pools = ['x']
        for ind, op in enumerate(self.config):
            # first op is always applied on x
            if ind == 0:
                pos = 0
                op_id = op
                self._collect_inds.remove(pos)
                op_name = self.op_names[op_id]
                self._ops.append(OPS[op_name](self.inp, 1, True, self.repeats, self.data_format))
                self._pos.append(pos)
                self._collect_inds.append(ind + 1)
                self._pools.append('{}({})'.format(op_name, self._pools[pos]))
            else:
                pos1, pos2, op_id1, op_id2 = op
                # drop op_id from loose ends
                for ind2, (pos, op_id) in enumerate(zip([pos1, pos2], [op_id1, op_id2])):
                    if pos in self._collect_inds:
                        self._collect_inds.remove(pos)
                    op_name = self.op_names[op_id]
                    self._ops.append(OPS[op_name](self.inp, 1, True, self.repeats, self.data_format))
                    self._pos.append(pos)
                    # self._collect_inds.append(ind * 3 + ind2 - 1) # Do not collect intermediate
                    self._pools.append('{}({})'.format(op_name, self._pools[pos]))
                # summation
                op_name = 'sum'
                self._ops.append(AggregateCell(agg_size=self.inp, pre_transform=False,
                                               concat=self.concat))  # turn-off convbnrelu
                self._pos.append([ind * 3 - 1, ind * 3])
                self._collect_inds.append(ind * 3 + 1)
                self._pools.append('{}({},{})'.format(op_name, self._pools[ind * 3 - 1], self._pools[ind * 3]))

    def __call__(self, x, training):
        """Do an inference on ContextualCell_v1.

        :param x: input tensor
        :return: output tensor
        """
        self.build()

        feats = [x]
        for pos, op in zip(self._pos, self._ops):
            if isinstance(pos, list):
                assert len(pos) == 2, "Two ops must be provided"
                feats.append(op(feats[pos[0]], feats[pos[1]], training))
            else:
                feats.append(op(feats[pos], training))
        out = 0
        for i in self._collect_inds:
            out += feats[i]
        return out

    def prettify(self):
        """Format printing.

        :return: formatted string of the module
        """
        return ' + '.join(self._pools[i] for i in self._collect_inds)


class MergeCell(object):
    """Pass two inputs through ContextualCell, and aggregate their results."""

    def __init__(self, op_names, ctx_config, conn, inps, agg_size, ctx_cell, repeats=1,
                 cell_concat=False, data_format='channels_first'):
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
        self.inp_1, self.inp_2 = inps
        self.op_names = op_names
        self.ctx_config = ctx_config
        self.conn = conn
        self.agg_size = agg_size
        self.ctx_cell = ctx_cell
        self.repeats = repeats
        self.cell_concat = cell_concat
        self.data_format = data_format

    def __call__(self, x1, x2, training):
        """Do an inference on MergeCell.

        :param x1: input tensor 1
        :param x2: input tensor 2
        :return: output tensor
        """
        x1 = self.ctx_cell(self.op_names, self.ctx_config, self.inp_1, repeats=self.repeats,
                           data_format=self.data_format)(x1, training)
        x2 = self.ctx_cell(self.op_names, self.ctx_config, self.inp_2, repeats=self.repeats,
                           data_format=self.data_format)(x2, training)
        return AggregateCell(self.agg_size, concat=self.cell_concat, data_format=self.data_format)(x1, x2, training)

    def prettify(self):
        """Format printing.

        :return: formatted string of the module
        """
        return self.op_1.prettify()


class MicroDecoder(object):
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
        self.aux_cell = aux_cell
        self.collect_inds = []
        # for description of the structure
        self.pool = ['l{}'.format(i + 1) for i in range(num_pools)]
        self.num_pools = num_pools
        self.info = []
        self.agg_size = agg_size
        self.agg_concat = agg_concat
        self.op_names = op_names
        self.cell_concat = cell_concat
        self.num_classes = num_classes

        # NOTE: bring all outputs to the same size
        for out_idx, size in enumerate(backbone_out_sizes):
            backbone_out_sizes[out_idx] = agg_size
        if sys.version_info[0] < 3:
            backbone_out_sizes = list(backbone_out_sizes)
        else:
            backbone_out_sizes = backbone_out_sizes.copy()
        self.cell_config, conns = config
        self.conns = conns
        self.ctx = self.cell_config
        self.repeats = sep_repeats
        self.collect_inds = []
        self.ctx_cell = ctx_cell
        self.backbone_out_sizes = backbone_out_sizes

    def build(self):
        """Build MicroDecoder."""
        cells = []
        aux_clfs = []
        for block_idx, conn in enumerate(self.conns):
            for ind in conn:
                if ind in self.collect_inds:
                    # remove from outputs if used by pool cell
                    self.collect_inds.remove(ind)
            ind_1, ind_2 = conn
            cells.append(MergeCell(self.op_names, self.cell_config, conn,
                                   (self.backbone_out_sizes[ind_1], self.backbone_out_sizes[ind_2]),
                                   self.agg_size,
                                   self.ctx_cell, repeats=self.repeats,
                                   cell_concat=self.cell_concat))
            aux_clfs.append({})
            if self.aux_cell:
                aux_clfs[block_idx]['aux_cell'] = self.ctx_cell(self.op_names, self.ctx, self.agg_size,
                                                                repeats=self.repeats)
            aux_clfs[block_idx]['aux_clf'] = Conv(self.num_classes, 3, strides=1, use_bias=True)
            self.collect_inds.append(block_idx + self.num_pools)
            self.backbone_out_sizes.append(self.agg_size)
            # for description
            self.pool.append('({} + {})'.format(self.pool[ind_1], self.pool[ind_2]))
        self.cells = cells
        self.aux_clfs = aux_clfs
        self.pre_clf = Conv(self.agg_size, 1, strides=1)
        self.conv_clf = Conv(self.num_classes, 3, strides=1, use_bias=True)
        self.info = ' + '.join(self.pool[i] for i in self.collect_inds)

    def __call__(self, x, training):
        """Do an inference on MicroDecoder.

        :param x: input tensor
        :return: output tensor
        """
        self.build()

        x = list(x)
        for out_idx in range(len(x)):
            x[out_idx] = Conv(self.agg_size, 1, strides=1, affine=True)(x[out_idx], training)
        for cell, aux_clf, conn in zip(self.cells, self.aux_clfs, self.conns):
            cell_out = cell(x[conn[0]], x[conn[1]], training)
            x.append(cell_out)
        out = x[self.collect_inds[0]]
        for i in range(1, len(self.collect_inds)):
            collect = x[self.collect_inds[i]]
            if out.get_shape()[2] > collect.get_shape()[2]:
                # upsample collect
                collect = resize_bilinear(collect, out.get_shape()[2:])
            elif collect.get_shape()[2] > out.get_shape()[2]:
                out = resize_bilinear(out, collect.get_shape()[2:])
            if self.agg_concat:
                out = tf.concat([out, collect], 1)
            else:
                out += collect

        out = tf.nn.relu(out)
        out = self.pre_clf(out, training)
        out = self.conv_clf(out, training)
        return out
