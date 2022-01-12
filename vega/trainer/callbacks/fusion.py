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

"""Callbacks called at certain points of trainer."""
import logging
import copy
import vega
from vega.common.class_factory import ClassFactory, ClassType
from vega.trainer.callbacks.callback import Callback
from vega.modules.operators.ops import Identity

if vega.is_torch_backend():
    import torch
    from torch.nn.utils.fusion import fuse_conv_bn_weights


@ClassFactory.register(ClassType.CALLBACK)
class OperatorFusionCallback(Callback):
    """Callback that fuse operators when valid model."""

    def __init__(self):
        """Construct a OperatorFusionCallback callback."""
        super(OperatorFusionCallback, self).__init__()

    def after_train(self, logs=None):
        """Be called before the validation."""
        if not vega.is_torch_backend() or self.trainer.model.__class__.__name__ != 'DagNetwork':
            return
        logging.info("Start operator fusion.")
        for name, node in self.trainer.model.module_map.items():
            module = node.module
            if isinstance(node.module, torch.nn.Conv2d):
                next_nodes = node.child_nodes
                if next_nodes and isinstance(next_nodes[0].module, torch.nn.BatchNorm2d):
                    node.module = self._fuse_conv_bn(module, next_nodes[0].module)
                    next_nodes[0].module = Identity()
        self._save_model()

    def _fuse_conv_bn(self, conv, bn):
        fused_conv = copy.deepcopy(conv)
        fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
            fused_conv.weight, fused_conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
        return fused_conv

    def _save_model(self):
        if vega.is_torch_backend():
            torch.save(self.trainer.model.state_dict(), self.trainer.weights_file)
