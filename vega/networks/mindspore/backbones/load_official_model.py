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

"""Load the official model from mindspore modelzoo."""
import operator
import importlib
from official_model_conf import output_layer_map, location_map
import mindspore.nn as nn
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class OffcialModelLoader(nn.Cell):
    """Load torchvison model and change it."""

    def __init__(self, model_name, output_layer_names=None, optional_conv_channels=None, **kwargs):
        super(OffcialModelLoader, self).__init__()
        model = self.load_package(model_name)
        self.model = model(**kwargs) if kwargs else model()
        if output_layer_names is None:
            self.output_layer_names = output_layer_map[model_name]
        else:
            self.output_layer_names = output_layer_names

        if optional_conv_channels is not None:
            if not isinstance(optional_conv_channels, dict):
                raise ValueError("The optional_conv_channels must be a dict, but got type of {}".
                                 format(type(optional_conv_channels)))
            all_need_changed_layer = self.get_changed_layer(optional_conv_channels)
            for need_change_layer in all_need_changed_layer:
                layer_name, new_channel, _type = need_change_layer
                self.change_layer(layer_name, new_channel, _type)

    def construct(self, inputs):
        """Forward of the network."""
        output = inputs
        for name, module in self.model.name_cells().items():
            output = module(output)
            if name == self.output_layer_names:
                return output
        return output

    def load_package(self, model_name):
        """Load the model class from the name.

        :param model_name: the specified model name
        :type model_name: str
        """
        package_name = "model_zoo.official." + location_map[model_name][0]
        class_name = location_map[model_name][1]
        module = importlib.import_module(package_name)
        return getattr(module, class_name)

    def is_sub_list(self, list1, list2):
        """Check whether the list1 is sun list of list2 or not.

        :param list1: the given list1.
        :type list1: list
        :param list2: the given list2.
        :type list2: list
        """
        if len(list1) >= len(list2):
            return False
        for index in range(len(list1)):
            if list1[index] != list2[index]:
                return False
        return True

    def get_all_layer_names(self):
        """Get all the layers name excluding the parent."""
        names_list = [name for name, _ in self.model.cells_and_names()]
        valid_names = []
        for index in range(len(names_list) - 1):
            cur_name = names_list[index]
            next_name = names_list[index + 1]
            if cur_name != "" and not self.is_sub_list(cur_name.split("."), next_name.split(".")):
                valid_names.append(cur_name)
        valid_names.append(next_name)
        return valid_names

    def get_changed_layer(self, specified_layers):
        """Get all the layers needed to change according to the specified layer.

        :param specified_layer: the given layer name, which need to be changed.
        :type specified_layer: str
        """
        valid_names = self.get_all_layer_names()
        specified_layers_names = [name for name, _ in specified_layers.items()]
        if not set(specified_layers_names).issubset(set(valid_names)):
            raise ValueError("The specified layer is not valid because it is not in the init model.")
        layer_index_map = {}
        need_change_layers = []
        for index, name in enumerate(valid_names):
            layer_index_map.update({name: index})

        for (specified_layer, new_channel) in specified_layers.items():
            init_index = layer_index_map[specified_layer]
            need_change_layers.append((specified_layer, new_channel, "out"))
            for i in range(init_index + 1, len(valid_names)):
                next_layer = valid_names[i]
                if next_layer.split(".")[-1].startswith("bn"):
                    need_change_layers.append((next_layer, new_channel, "in"))
                elif next_layer.split(".")[-2] == "downsample":
                    need_change_layers.append((next_layer, new_channel, "out"))
                elif next_layer.split(".")[-1].startswith("conv"):
                    need_change_layers.append((next_layer, new_channel, "in"))
                    break

        return need_change_layers

    def change_layer(self, need_changed_layer, new_channels, _type):
        """Change the layer according to the given name.

        :param layer_name: the given layer name, which must be a valid name in init model
        :type layer_name: str
        :param new_channels: the new channel after changed
        :type new_channels: int
        :param _type: "in" or "out"
        :type _type: str
        """
        layer_getter = operator.attrgetter(need_changed_layer)
        changed_layer = layer_getter(self.model)
        if isinstance(changed_layer, nn.Conv2d):
            in_channels = changed_layer.in_channels
            out_channels = changed_layer.out_channels
            kernel_size = changed_layer.kernel_size
            stride = changed_layer.stride
            pad_mode = changed_layer.pad_mode
            padding = changed_layer.padding
            has_bias = changed_layer.has_bias
            if _type == "out":
                new_layer = nn.Conv2d(in_channels=in_channels, out_channels=new_channels,
                                      kernel_size=kernel_size, stride=stride, pad_mode=pad_mode, padding=padding,
                                      has_bias=has_bias)
            else:
                new_layer = nn.Conv2d(in_channels=new_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride, pad_mode=pad_mode, padding=padding,
                                      has_bias=has_bias)
        elif isinstance(changed_layer, nn.BatchNorm2d):
            new_layer = nn.BatchNorm2d(num_features=new_channels)

        layer_name = need_changed_layer.split(".")
        layer_name = [name if not name.isdigit() else int(name) for name in layer_name]
        self.model._cells[layer_name[0]][layer_name[1]]._cells[layer_name[2]] = new_layer
