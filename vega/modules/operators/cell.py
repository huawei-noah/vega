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

"""Import all torch operators."""

from vega.common import ClassType, ClassFactory
from vega.modules.operators import ops
from vega.modules.operators.mix_ops import MixedOp, OPS
from vega.modules.operators.conv import conv_bn_relu, Seq, FactorizedReduce, ReLUConvBN


@ClassFactory.register(ClassType.NETWORK)
class Cell(ops.Module):
    """Cell structure according to desc."""

    concat_size = 0

    def __init__(self, genotype, steps, concat, reduction, reduction_prev=None, C_prev_prev=None, C_prev=None, C=None):
        """Init Cell."""
        super(Cell, self).__init__()
        self.genotype = genotype
        self.steps = steps
        self.concat = concat
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        self.C_prev_prev = C_prev_prev
        self.C_prev = C_prev
        self.C = C
        self.concat_size = 0
        affine = True
        if isinstance(self.genotype[0][0], list):
            affine = False
        if self.reduction_prev:
            self.preprocess0 = FactorizedReduce(self.C_prev_prev, self.C, affine)
        else:
            self.preprocess0 = ReLUConvBN(self.C_prev_prev, self.C, 1, 1, 0, affine)
        self.preprocess1 = ReLUConvBN(self.C_prev, self.C, 1, 1, 0, affine)
        op_names, indices_out, indices_inp = zip(*self.genotype)
        self.build_ops(self.C, op_names, indices_out, indices_inp, self.concat, self.reduction)
        self.concat_size = len(self.concat)

    def build_ops(self, C, op_names, indices_out, indices_inp, concat, reduction):
        """Compile the cell.

        :param C: channels of this cell
        :type C: int
        :param op_names: list of all the operations in description
        :type op_names: list of str
        :param indices_out: list of all output nodes
        :type indices_out: list of int
        :param indices_inp: list of all input nodes link to output node
        :type indices_inp: list of int
        :param concat: cell concat list of output node
        :type concat: list of int
        :param reduction: whether to reduce
        :type reduction: bool
        """
        self._concat = concat
        self._multiplier = len(concat)
        self.out_inp_list = []
        temp_list = []
        idx_cmp = 2
        _op_list = []
        for i in range(len(op_names)):
            if indices_out[i] == idx_cmp:
                temp_list.append(indices_inp[i])
            elif indices_out[i] > idx_cmp:
                self.out_inp_list.append(temp_list.copy())
                temp_list = []
                idx_cmp += 1
                temp_list.append(indices_inp[i])
            else:
                raise Exception("input index should not less than idx_cmp")
            stride = 2 if reduction and indices_inp[i] < 2 else 1
            op = MixedOp(C=C, stride=stride, ops_cands=op_names[i])
            _op_list.append(op)
        self.op_list = Seq(*tuple(_op_list))
        self.oplist = list(self.op_list.children())
        self.out_inp_list.append(temp_list.copy())
        if len(self.out_inp_list) != self.steps:
            raise Exception("out_inp_list length should equal to steps")

    def call(self, s0, s1, weights=None, drop_path_prob=0, selected_idxs=None):
        """Forward function of Cell.

        :param s0: feature map of previous of previous cell
        :type s0: torch tensor
        :param s1: feature map of previous cell
        :type s1: torch tensor
        :param weights: weights of operations in cell
        :type weights: torch tensor, 2 dimension
        :return: cell output
        :rtype: torch tensor
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        idx = 0
        for i in range(self.steps):
            hlist = []
            for j, inp in enumerate(self.out_inp_list[i]):
                op = self.oplist[idx + j]
                if selected_idxs is None:
                    if weights is None:
                        h = op(states[inp])
                    else:
                        h = op(states[inp], weights[idx + j])
                    if drop_path_prob > 0. and not isinstance(list(op.children())[0], ops.Identity):
                        h = ops.drop_path(h, drop_path_prob)
                    hlist.append(h)
                elif selected_idxs[idx + j] == -1:
                    # undecided mix edges
                    h = op(states[inp], weights[idx + j])
                    hlist.append(h)
                elif selected_idxs[idx + j] == 0:
                    # zero operation
                    continue
                else:
                    h = self.oplist[idx + j](states[inp], None, selected_idxs[idx + j])
                    hlist.append(h)
            s = ops.add_n(hlist)
            states.append(s)
            idx += len(self.out_inp_list[i])
        states_list = ()
        for i in self._concat:
            states_list += (states[i],)
        return ops.concat(states_list)


@ClassFactory.register(ClassType.NETWORK)
class NormalCell(Cell):
    """Normal Cell structure according to desc."""

    def __init__(self, genotype, steps, concat, reduction_prev=None, C_prev_prev=None, C_prev=None, C=None):
        super(NormalCell, self).__init__(genotype, steps, concat, False, reduction_prev, C_prev_prev, C_prev, C)


@ClassFactory.register(ClassType.NETWORK)
class ReduceCell(Cell):
    """Reduce Cell structure according to desc."""

    def __init__(self, genotype, steps, concat, reduction_prev=None, C_prev_prev=None, C_prev=None, C=None):
        super(ReduceCell, self).__init__(genotype, steps, concat, True, reduction_prev, C_prev_prev, C_prev, C)


@ClassFactory.register(ClassType.NETWORK)
class ContextualCell_v1(ops.Module):
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
        self.ops = ops.MoudleList()
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
                # turn-off scaling in batch norm
                self.ops.append(OPS[op_name](inp, 1, True, repeats))
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
                    # turn-off scaling in batch norm
                    self.ops.append(OPS[op_name](inp, 1, True, repeats))
                    self._pos.append(pos)
                    self._pools.append('{}({})'.format(
                        op_name, self._pools[pos]))
                # summation
                op_name = 'sum'
                self.ops.append(AggregateCell(size_1=None, size_2=None, agg_size=inp, pre_transform=False,
                                              concat=concat))  # turn-off convbnrelu
                self._pos.append([ind * 3 - 1, ind * 3])
                self._collect_inds.append(ind * 3 + 1)
                self._pools.append('{}({},{})'.format(
                    op_name, self._pools[ind * 3 - 1], self._pools[ind * 3]))

    def call(self, x):
        """Do an inference on ContextualCell_v1.

        :param x: input tensor
        :return: output tensor
        """
        feats = [x]
        for pos, op in zip(self._pos, self.ops):
            if isinstance(pos, list):
                if len(pos) == 2:
                    feats.append(op(feats[pos[0]], feats[pos[1]]))
                else:
                    raise ValueError("Two ops must be provided")
            else:
                feats.append(op(feats[pos]))
        out = 0
        for i in self._collect_inds:
            out += feats[i]
        return out


@ClassFactory.register(ClassType.NETWORK)
class AggregateCell(ops.Module):
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

    def call(self, x1, x2):
        """Do an inference on AggregateCell.

        :param x1: first input
        :param x2: second input
        :return: output
        """
        if self.pre_transform:
            x1 = self.branch_1(x1)
            x2 = self.branch_2(x2)
        if tuple(ops.get_shape(x1)[2:]) > tuple(ops.get_shape(x2)[2:]):
            x2 = ops.interpolate(x2, size=ops.get_shape(
                x1)[2:], mode='bilinear', align_corners=True)
        elif tuple(ops.get_shape(x1)[2:]) < tuple(ops.get_shape(x2)[2:]):
            x1 = ops.interpolate(x1, size=ops.get_shape(
                x2)[2:], mode='bilinear', align_corners=True)
        if self.concat:
            return self.conv1x1(ops.concat([x1, x2]))
        else:
            return x1 + x2
