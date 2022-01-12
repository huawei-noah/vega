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

"""ResNetVariant for Detection."""
from collections import OrderedDict
from vega.common import ClassType, ClassFactory
from vega.modules.module import Module
from vega.modules.connections import Sequential


@ClassFactory.register(ClassType.NETWORK)
class MultiOutputGetter(Module):
    """Get output layer by layer names and connect into a OrderDict."""

    def __init__(self, model, layer_names):
        super(MultiOutputGetter, self).__init__(model)
        if not layer_names or not set(layer_names).issubset([name for name, _ in model.named_children()]):
            raise ValueError("layer_names are not present in model")
        if isinstance(layer_names, list):
            layer_names = {v: k for k, v in enumerate(layer_names)}
        self.output_layers = OrderedDict()
        for name, module in model.named_children():
            if not layer_names:
                break
            self.add_module(name, Module.from_module(module))
            if name in layer_names:
                layer_name = layer_names.pop(name)
                self.output_layers[name] = layer_name

    def call(self, inputs):
        """Override call function, connect models into a OrderedDict."""
        output = inputs
        outs = OrderedDict()
        for name, model in self.named_children():
            output = model(output)
            if name in self.output_layers:
                outs[self.output_layers[name]] = output
        return outs

    @property
    def out_channels(self):
        """Output Channel for Module."""
        layers = [layer for layer in self.children() if isinstance(layer, Sequential)]
        return [layer.out_channels for layer in layers]


@ClassFactory.register(ClassType.NETWORK)
class BackboneGetter(MultiOutputGetter):
    """Backbone Getter form torchvision ResNet."""

    def __init__(self, backbone_name, layer_names=None, **kwargs):
        backbone = ClassFactory.get_cls(ClassType.NETWORK, backbone_name)
        backbone = backbone(**kwargs) if kwargs else backbone()
        if hasattr(backbone, "layers_name"):
            layer_names = backbone.layers_name()

        super(BackboneGetter, self).__init__(backbone, layer_names)


@ClassFactory.register(ClassType.NETWORK)
class ResNetBackbone(BackboneGetter):
    """Get Resnet Backbone form torchvision ResNet."""

    def __init__(self, backbone_name='torchvision_resnet50', layer_names=None, trainable_layers=3, **kwargs):
        layer_names = layer_names or ['layer1', 'layer2', 'layer3', 'layer4']
        self.trainable_layers = trainable_layers
        super(ResNetBackbone, self).__init__(backbone_name, layer_names, **kwargs)
        self.freeze(['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:self.trainable_layers])
