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

"""This is FasterRCNN network."""

from collections import OrderedDict
from vega.common import ClassFactory, ClassType
from vega.modules.connections import Sequential
from vega.modules.module import Module
from torchvision.models.detection import FasterRCNN


@ClassFactory.register(ClassType.NETWORK)
class FasterRCNN(FasterRCNN, Module):
    """Create ResNet Network."""

    def __init__(self, num_classes=91, backbone='ResNetBackbone', neck='FPN', convert_pretrained=False,
                 freeze_swap_keys=None, **kwargs):
        """Create layers.

        :param num_class: number of class
        :type num_class: int
        """
        self.convert_pretrained = convert_pretrained
        self.freeze_swap_keys = freeze_swap_keys
        backbone_cls = ClassFactory.get_instance(ClassType.NETWORK, backbone)
        neck_cls = ClassFactory.get_instance(ClassType.NETWORK, neck, in_channels=backbone_cls.out_channels)
        backbone_neck = Sequential()
        backbone_neck.append(backbone_cls, 'body')
        backbone_neck.append(neck_cls, 'fpn')
        import torchvision
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0, 1, 2, 3], output_size=7, sampling_ratio=2)
        super(FasterRCNN, self).__init__(backbone_neck, num_classes, box_roi_pool=roi_pooler, **kwargs)

    def load_state_dict(self, state_dict=None, strict=None, exclude_weight_prefix=None):
        """Load State dict."""
        if self.convert_pretrained:
            if exclude_weight_prefix:
                state_dict = self._exclude_checkpoint_by_prefix(state_dict)
            state_dict = self._convert_state_dict(self.state_dict(), state_dict)
            return super().load_state_dict(state_dict)
        if not self.freeze_swap_keys:
            return super().load_state_dict(state_dict, strict or False)
        not_swap_keys = super().load_state_dict(state_dict, False)
        need_freeze_layers = [name for name, parameter in self.named_parameters() if name not in not_swap_keys]
        for name, parameter in self.named_parameters():
            if not all([not name.startswith(layer) for layer in need_freeze_layers]):
                parameter.requires_grad_(False)
            else:
                parameter.requires_grad_(True)

    def _exclude_checkpoint_by_prefix(self, states):
        if self.exclude_weight_prefix:
            if not isinstance(self.exclude_weight_prefix, list):
                self.exclude_weight_prefix = [self.exclude_weight_prefix]
            for prefix in self.exclude_weight_prefix:
                states = {k: v for k, v in states.items() if not k.startswith(prefix)}
            self.strict = False
        return states

    def _convert_state_dict(self, own_state, state_dict):
        own_state_copy = OrderedDict({k: v for k, v in own_state.items() if 'num_batches_tracked' not in k})
        state_dict_copy = OrderedDict({k: v for k, v in state_dict.items() if 'num_batches_tracked' not in k})
        while state_dict_copy:
            name, weight = state_dict_copy.popitem(0)
            own_name, own_weight = own_state_copy.popitem(0)
            if weight.shape != own_weight.shape:
                raise ValueError("Unexpected key(s) in state_dict for ""convert: {} {} --> {} {}".format(
                    name, weight.shape, own_name, own_weight.shape))
            own_state[own_name] = weight
        return own_state
