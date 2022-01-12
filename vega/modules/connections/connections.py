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

"""This is SearchSpace for connection."""
from copy import deepcopy
from collections import OrderedDict
from vega.common import ClassFactory, ClassType, SearchSpaceType
from vega.common import update_dict_with_flatten_keys
from vega.modules.module import Module
from vega.modules.operators import ops


class ConnectionsDecorator(Module):
    """Base class for Connections."""

    def __init__(self, *models, **kwargs):
        super(ConnectionsDecorator, self).__init__(*models, **kwargs)
        if kwargs:
            for key, model in kwargs.items():
                self.__add_module(key, model)
        else:
            for idx, model in enumerate(models):
                self.__add_module(str(idx), model)

    def __add_module(self, key, model):
        if isinstance(model, OrderedDict):
            for name, value in model.items():
                if not isinstance(value, Module) and isinstance(value, dict):
                    value = self.from_desc(value)
                self.add_module(name, value)
        else:
            if not isinstance(model, Module) and isinstance(model, dict):
                model = self.from_desc(model)
            self.add_module(key, model)

    def to_desc(self, recursion=True):
        """Convert module to desc."""
        if not recursion:
            return self.desc
        desc = {"type": self.__class__.__name__}
        for name, module in self.named_children():
            if hasattr(module, 'to_desc'):
                sub_desc = module.to_desc()
                desc[name] = sub_desc
        return desc

    def call(self, inputs):
        """Override call function."""
        raise NotImplementedError


@ClassFactory.register(SearchSpaceType.CONNECTIONS)
class Add(ConnectionsDecorator):
    """Create Lambda for forward x."""

    def __init__(self, *models):
        super(Add, self).__init__(*models)

    def call(self, x):
        """Forward x."""
        output = None
        for model in self.children():
            if model is not None:
                if output is None:
                    output = model(x)
                else:
                    output += model(x)
        return output

    @property
    def out_channels(self):
        """Get out channels."""
        return [k.out_channels for k in self.children() if hasattr(k, 'out_channels')]


@ClassFactory.register(SearchSpaceType.CONNECTIONS)
class Sequential(ConnectionsDecorator):
    """Sequential SearchSpace."""

    def __init__(self, *models, **kwargs):
        super(Sequential, self).__init__(*models, **kwargs)

    def append(self, module, name=None):
        """Append new module."""
        name = name or str(len(self._modules))
        self.add_module(name, module)
        return self

    def call(self, inputs):
        """Override call function, connect models into a seq."""
        output = inputs
        models = self.children()
        for model in models:
            output = model(output)
        return output

    @classmethod
    def from_module(cls, module):
        """Convert Sequential."""
        model = cls()
        for name, module in module.named_children():
            model.append(module, name)
        return model

    def to_desc(self, recursion=True):
        """Convert module to desc."""
        if not recursion:
            return self.desc
        desc = {"type": self.__class__.__name__}
        modules = []
        for name, module in self.named_children():
            if hasattr(module, 'to_desc'):
                sub_desc = module.to_desc()
                desc[name] = sub_desc
                modules.append(name)
        if modules:
            desc["modules"] = modules
        return desc


class ModuleList(Module):
    """Class of LeakyReLU."""

    def __init__(self):
        super(ModuleList, self).__init__()

    def append(self, moudle):
        """Append new moudle."""
        index = len(self._modules)
        self.add_module(str(index), moudle)
        return self

    def __getitem__(self, idx):
        """Get item by idx."""
        return list(self.children())[idx]


@ClassFactory.register(SearchSpaceType.CONNECTIONS)
class OutlistSequential(ConnectionsDecorator):
    """Sequential SearchSpace."""

    def __init__(self, *models, out_list=None):
        super(OutlistSequential, self).__init__(*models)
        self.out_list = out_list

    def append(self, module):
        """Append new module."""
        self.add_module(str(len(self._modules)), module)
        return self

    def call(self, inputs):
        """Override compile function, conect models into a seq."""
        output = inputs
        models = self.children()
        outputs = []
        for idx, model in enumerate(models):
            output = model(output)
            if self.out_list and idx not in self.out_list:
                continue
            outputs.append(output)
        return outputs

    @property
    def out_channels(self):
        """Output Channel for Module."""
        return [layer.out_channels for layer in self.children()]


@ClassFactory.register(SearchSpaceType.CONNECTIONS)
class OutDictSequential(ConnectionsDecorator):
    """Sequential SearchSpace."""

    def __init__(self, *models, out_list=None):
        super(OutDictSequential, self).__init__(*models)
        self.out_list = out_list

    def append(self, module):
        """Append new module."""
        self.add_module(str(len(self._modules)), module)
        return self

    def call(self, inputs):
        """Override compile function, conect models into a seq."""
        output = inputs
        models = self.children()
        outputs = {}
        for idx, model in enumerate(models):
            output = model(output)
            if self.out_list and idx not in self.out_list:
                continue
            outputs[idx] = output
        return outputs

    @property
    def out_channels(self):
        """Output Channel for Module."""
        return [layer.out_channels for layer in self.children()]


@ClassFactory.register(SearchSpaceType.CONNECTIONS)
class MultiOutput(ConnectionsDecorator):
    """MultiOutput Connections."""

    def __init__(self, *models, out_func=None):
        super(MultiOutput, self).__init__(*models)
        self.out_func = out_func

    def add(self, module):
        """Add a module into MultiOutput."""
        self.add_module(str(len(self._modules.values())), module)

    def call(self, inputs):
        """Override compile function, connect models into a seq."""
        models = list(self.children())
        if not models:
            return None
        input_model = models.pop(0)
        x = input_model(inputs)
        outputs = []
        for idx, model in enumerate(models):
            outputs.append(model(x))
        if self.out_func is not None:
            outputs = self.out_func(outputs)
        return outputs


@ClassFactory.register(SearchSpaceType.CONNECTIONS)
class Concat(MultiOutput):
    """Create Lambda for forward x."""

    def __init__(self, *models):
        super(Concat, self).__init__(*models)
        self.out_func = ops.concat

    def call(self, inputs):
        """Override compile function, connect models into a seq."""
        models = list(self.children())
        if not models:
            return None
        outputs = []
        for idx, model in enumerate(models):
            outputs.append(model(inputs))
        return ops.concat(outputs)


@ClassFactory.register(SearchSpaceType.CONNECTIONS)
class ProcessList(ConnectionsDecorator):
    """Process_list."""

    def __init__(self, *models, out_list=None):
        super(ProcessList, self).__init__(*models)
        self.out_list = out_list

    def call(self, inputs):
        """Call list of input train forward function."""
        if self.out_list is None:
            if isinstance(inputs, list):
                outputs = []
                for model, idx in zip(self.children(), [i for i in range(len(inputs))]):
                    output = model(inputs[idx])
                    outputs.append(output)
                output = outputs
            else:
                raise ValueError("Input must list!")
        else:
            inputs = list(inputs)
            for model, idx in zip(self.children(), self.out_list):
                if isinstance(idx, list):
                    if len(idx) == 2:
                        output = model(inputs[idx[0]], inputs[idx[1]])
                        inputs.append(output)
                    else:
                        raise ValueError('Idx must be 2.')
                else:
                    inputs.append(model(inputs[idx]))
            output = inputs
        return output


@ClassFactory.register(ClassType.NETWORK)
class Repeat(Module):
    """Repeat SearchSpace."""

    def __init__(self, num_reps, items, ref):
        """Make a search space obj repeat with nums.

        :param num_reps: repeat num.
        :param items: search space items.
        :return: ref
        """
        super(Repeat, self).__init__()
        repeat_items = deepcopy(items)
        ref_copy = deepcopy(ref)
        for idx in range(num_reps):
            params = {}
            for key, values in repeat_items.items():
                if not isinstance(values, list):
                    params[key] = values
                else:
                    v_idx = idx if len(values) > idx else -1
                    params[key] = values[v_idx]
            params = update_dict_with_flatten_keys(ref_copy, params)
            name, module = _create_module(params)
            self.add_module('{}{}'.format(name, idx), module)


def _create_module(model):
    """Create search space from model or desc."""
    if isinstance(model, Module):
        return model.__class__.__name__, model
    elif isinstance(model, dict):
        module_type = model.get('type')
        module_param = deepcopy(model)
        module_param.pop('type')
        module = ClassFactory.get_cls(ClassType.NETWORK, module_type)
        return module_type, module(**module_param)


@ClassFactory.register(SearchSpaceType.CONNECTIONS)
class Cells(Module):
    """Cells Connection."""

    def __init__(self, desc, C_curr, C, auxiliary=False, auxiliary_layer=0):
        super(Cells, self).__init__()
        self.C_prev_prev = C_curr
        self.C_prev = C_curr
        self.C_curr = C
        self.C_aux = None
        self.auxiliary = auxiliary
        if auxiliary:
            self.auxiliary_layer = auxiliary_layer
        normal_info = desc.get('normal')
        if normal_info:
            self.k = len(normal_info.genotype)
            self.num_ops = len(normal_info.genotype[0][0])
            self.len_alpha = len(normal_info.genotype)
            self.steps = normal_info.steps
        self._build(desc)

    def _build(self, desc):
        """Build cell."""
        reduction_prev = True if self.C_curr == self.C_prev else False
        for idx, model_name in enumerate(desc.get('modules')):
            params = deepcopy(desc.get(model_name))
            if model_name == 'reduce':
                self.C_curr *= 2
                reduction = True
            else:
                reduction = False
            params['reduction_prev'] = reduction_prev
            params['C_prev_prev'] = self.C_prev_prev
            params['C_prev'] = self.C_prev
            params['C'] = self.C_curr
            reduction_prev = reduction
            model = ClassFactory.get_instance(ClassType.NETWORK, params)
            self.add_module(str(idx), model)
            concat_size = model.concat_size if hasattr(model, 'concat_size') else 1
            self.C_prev_prev, self.C_prev = self.C_prev, concat_size * self.C_curr
            if self.auxiliary and idx == self.auxiliary_layer:
                self.C_aux = self.C_prev

    def output_channels(self):
        """Get output channels."""
        return self.C_prev, self.C_aux if self.auxiliary else None
