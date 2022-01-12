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

"""Common module in NAGO."""
import torch
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d
from vega.common import ClassType, ClassFactory
from vega.networks.pytorch.heads.auxiliary_head import AuxiliaryHead
from .ops import depthwise_separable_conv_general, Triplet_unit, PassThrough, BoundedScalarMultiply
from .logical_graph import LogicalMasterGraph, LogicalCellGraph, LogicalOpGraph, EdgeMerge, \
    LogicalOperation, Ops


def diff_size(x, size):
    """Return size is same as shape or not."""
    return x.shape[2] != size


def get_operation(op, inplanes, outplanes, stride, conv_type):
    """Set up conv and pool operations."""
    kernel_size = Ops.ops_to_kernel_size[op]
    padding = [(k - 1) // 2 for k in kernel_size]
    if op in Ops.pooling_ops:
        if inplanes == outplanes:
            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        else:
            return nn.Sequential(nn.Conv2d(inplanes, outplanes, 1, 1, 0),
                                 nn.AvgPool2d(kernel_size, stride=stride, padding=padding))
    else:
        if conv_type == 'depthwise_separable':
            return depthwise_separable_conv_general(inplanes, outplanes, stride, kernel_size, padding)
        else:
            return nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding=padding)


@ClassFactory.register(ClassType.NETWORK)
class MasterNetwork(nn.Module):
    """Define the network model based on the logic top-level graph.

    Process inputs to the top-level graph and merge its output.
    """

    def __init__(self, master_node: LogicalMasterGraph, plane_multiplier: int, image_size: int, num_classes=10,
                 drop_path=None, aux_head=False, dropout_p=0):
        """Initialize MasterNetwork."""
        super().__init__()
        self.nodes = master_node.nodes
        self.input_nodes = master_node.input_nodes
        self.output_nodes = master_node.output_nodes
        self.drop_path = drop_path
        self.dropout_p = dropout_p

        plane_multiplier = int(plane_multiplier)
        out_channels = int(master_node.inplanes * plane_multiplier)
        if image_size < 40:
            large_images = False
            self.first_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
            stage_mult = 2
        else:
            large_images = True
            self.first_conv = nn.Sequential(
                depthwise_separable_conv_general(3, out_channels // 2, 2),
                nn.BatchNorm2d(out_channels // 2),
                Triplet_unit(out_channels // 2, out_channels, dropout_p=dropout_p, stride=2),
            )
            stage_mult = 0.5
        self.image_size = image_size
        self.node_ops = nn.ModuleList()
        self.aux_head_node = None
        for node in master_node.child_nodes:
            if aux_head and node.stage == 2 and self.aux_head_node is None:
                aux_in_channels = node.child_nodes[0].child_nodes[0].inplanes * plane_multiplier
                self.aux_head = AuxiliaryHead(aux_in_channels, num_classes, large_images)
                cell = CellGraph(node, plane_multiplier, image_size * stage_mult,
                                 drop_path=drop_path, aux_head=self.aux_head, dropout_p=dropout_p)
                self.aux_head_node = cell
            else:
                cell = CellGraph(node, plane_multiplier, image_size * stage_mult,
                                 drop_path=drop_path, dropout_p=dropout_p)
            self.node_ops.append(cell)

        self._prepare_output_merge(plane_multiplier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        n_count = self._get_final_stage_planes(plane_multiplier)

        self.conv_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_count, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)
        # init the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _prepare_output_merge(self, multi):
        out_channels = self._get_final_stage_planes(multi)
        if len(self.output_nodes) > 1:
            mergers = []
            for i in self.output_nodes:
                node_out = int(self.node_ops[i].logic.outplanes * multi)
                if node_out != out_channels:
                    op = nn.Conv2d(node_out, out_channels, kernel_size=1, padding=0, stride=1)
                else:
                    op = PassThrough()
                mergers.append(op)
            self.out_mergers = nn.ModuleList(mergers)
        else:
            self.out_mergers = nn.ModuleList([PassThrough()])

    def _get_final_stage_planes(self, multi):
        return int(self.node_ops[self.output_nodes[-1]].logic.outplanes * multi)

    def _parse_outputs(self, results):
        if len(results.values()) == 1:
            return results.values()[0]
        outputs = [results[res] for i, res in enumerate(results.keys()) if i in self.output_nodes]
        ref_size = min([out.shape[-1] for out in outputs])
        outputs = [adaptive_avg_pool2d(feat, (ref_size, ref_size)) if diff_size(feat, ref_size) else feat for feat in
                   outputs]
        result = self.out_mergers[0](outputs[0])
        for i, out in enumerate(outputs[1:]):
            result = result + self.out_mergers[i + 1](out)
        return result / len(outputs)

    def _process_graph(self, x):
        results = {}
        # all input nodes receive the same input x
        for id in self.input_nodes:
            results[id] = self.node_ops[id](x)

        # non-input nodes nodes
        for id, node in enumerate(self.nodes):
            if id not in self.input_nodes:
                results[id] = self.node_ops[id](*[results[_id] for _id in node.inputs])

        # get the first output node's result and sum over all the output nodes' results
        result = self._parse_outputs(results)
        return result

    def forward(self, x):
        """Implement forward."""
        x = self.first_conv(x)
        x = self._process_graph(x)
        x = self.conv_out(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        aux_logits = None
        if self.training and self.aux_head_node:
            aux_logits = self.aux_head_node.aux_logits
        return self.fc(x), aux_logits


@ClassFactory.register(ClassType.NETWORK)
class CellGraph(MasterNetwork):
    """Merge and process inputs to the middle-level graph."""

    def __init__(self, cell: LogicalCellGraph, plane_multiplier: int, image_size: int,
                 drop_path=None, aux_head=None, dropout_p=0):
        nn.Module.__init__(self)
        self.aux_head = aux_head
        self.dropout_p = dropout_p
        self.logic = cell
        self.nodes = cell.nodes
        self.input_nodes = cell.input_nodes
        self.output_nodes = cell.output_nodes
        self.node_ops = nn.ModuleList()
        self.image_size = int(image_size / 2 ** (self.logic.stage + 1))
        self.prepare_merger_edges(plane_multiplier)
        self.drop_path = drop_path
        for node in cell.child_nodes:
            self.node_ops.append(OpGraph(node, plane_multiplier, drop_path=drop_path, dropout_p=dropout_p))

    def _parse_outputs(self, results):
        if len(results.values()) == 1:
            return list(results.values())[0]
        # get the first output node's result and sum over all the output nodes' results
        result = results[self.output_nodes[0]]
        for id in self.output_nodes[1:]:
            result = result + results[id]
        result = result / len(self.output_nodes)
        return result

    def prepare_merger_edges(self, multi):
        """Prepare merge edges."""
        if len(self.logic.inputs) > 1 and self.logic.merging_strategy in [EdgeMerge.SUM, EdgeMerge.ATTENTION]:
            mergers = []
            for i in self.logic.incoming_planes:
                if i == self.logic.inplanes:
                    op = BoundedScalarMultiply()
                else:
                    op = nn.Conv2d(int(i * multi), int(self.logic.inplanes * multi), kernel_size=1, padding=0, stride=1)
                mergers.append(op)
            if self.logic.merging_strategy == EdgeMerge.ATTENTION:
                mergers.append(BoundedScalarMultiply())
            self.merger = nn.ModuleList(mergers)

    def forward(self, *input):
        """Forward method."""
        ref_size = int(self.image_size)
        # downscaling to right size if needed
        input = [adaptive_avg_pool2d(feat, (ref_size, ref_size)) if diff_size(feat, ref_size) else feat for feat in
                 input]
        out = self.merge_inputs(input)
        if self.training and self.aux_head:
            self.aux_logits = self.aux_head(out)
        out = self._process_graph(out)
        return out

    def compute_active_input_indices(self, l_input, merge_mode):
        """Compute active input indices."""
        return range(l_input)

    def merge_inputs(self, input):
        """Merge inputs."""
        if len(input) > 1:
            idxs = self.compute_active_input_indices(len(input), self.logic.merging_strategy)
            if self.logic.merging_strategy == EdgeMerge.SUM:
                out = sum([self.merger[id](input[id]) for id in idxs])
            elif self.logic.merging_strategy == EdgeMerge.ATTENTION:
                id0 = idxs[0]
                attention_mask = self.merger[id0](input[id0])
                attention_mask = (attention_mask - attention_mask.min()) / (attention_mask.max() - attention_mask.min())
                out = sum([self.merger[id](input[id]) for id in idxs[1:]])
                out = self.merger[-1](out) + out * attention_mask  # weighted sum
            else:
                out = torch.cat(input, 1)
        else:
            out = input[0]
        return out


@ClassFactory.register(ClassType.NETWORK)
class OpGraph(CellGraph):
    """Merge and process inputs to the bottom-level graph."""

    def __init__(self, op_graph: LogicalOpGraph, plane_multiplier: int, drop_path=None, dropout_p=0):
        nn.Module.__init__(self)
        self.logic = op_graph
        self.nodes = op_graph.nodes
        self.input_nodes = op_graph.input_nodes
        self.output_nodes = op_graph.output_nodes
        self.node_ops = nn.ModuleList()
        self.prepare_merger_edges(plane_multiplier)
        self.drop_path = drop_path
        for node in op_graph.child_nodes:
            self.node_ops.append(OpNode(node, plane_multiplier, dropout_p))

    def forward(self, *input):
        """Forward method."""
        out = self.merge_inputs(input)
        return self._process_graph(out)


@ClassFactory.register(ClassType.NETWORK)
class TorchOperation(nn.Module):
    """Operation at each node in the bottom-level graph."""

    def __init__(self, node: LogicalOperation, plane_multiplier, dropout_p, stride=1):
        super(TorchOperation, self).__init__()
        self.dropout_p = dropout_p
        outplanes = int(node.outplanes * plane_multiplier)
        inplanes = int(node.inplanes * plane_multiplier)
        self.op = get_operation(node.type, inplanes, outplanes, stride, node.conv_type)
        self.type = node.type
        if node.type in Ops.conv_ops:
            self.relu = nn.ReLU(inplace=False)
            self.bn = nn.BatchNorm2d(outplanes)
            if self.dropout_p > 0:
                self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        """Forward method."""
        if self.type in Ops.conv_ops:
            out = self.relu(x)
            out = self.op(out)
            out = self.bn(out)
        else:
            out = self.op(x)

        if self.type in Ops.conv_ops and self.dropout_p > 0:
            out = self.dropout(out)
        return out


@ClassFactory.register(ClassType.NETWORK)
class OpNode(CellGraph):
    """Merge and process inputs to each node in the bottom-level graph."""

    def __init__(self, node: LogicalOperation, plane_multiplier, dropout_p=0, drop_path=False):
        nn.Module.__init__(self)
        self.logic = node
        self.input_nums = len(node.inputs)
        self.prepare_merger_edges(plane_multiplier)
        self.operation = TorchOperation(node, plane_multiplier, dropout_p)
        self.drop_path = drop_path

    def compute_active_input_indices(self, l_input, merge_mode):
        """Compute active input indices."""
        if self.training or self.drop_path == 0 or l_input == 1:
            return range(l_input)
        else:
            if merge_mode in [EdgeMerge.SUM, EdgeMerge.CAT]:
                idx = [i for i in range(l_input) if torch.rand(1) >= self.drop_path]
                if len(idx) == 0:
                    idx = [int(torch.randint(0, l_input, (1,)))]
                return idx
            elif EdgeMerge.ATTENTION:
                if l_input == 2:  # when using attention you need at least 2 inputs
                    return range(l_input)
                idx = [i for i in range(1, l_input) if torch.rand(1) >= self.drop_path]
                if len(idx) == 0:
                    idx = [int(torch.randint(1, l_input, (1,)))]
                return [0] + idx

    def forward(self, *input):
        """Forward method."""
        out = self.merge_inputs(input)
        return self.operation(out)
