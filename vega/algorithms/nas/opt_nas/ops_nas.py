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
"""This is Operator SearchSpace."""
from vega.common import ClassFactory, ClassType, SearchableRegister, Searchable, space, change_space
from vega.core.search_space import SearchSpace
from vega.networks.network_desc import NetworkDesc
from vega.core.pipeline.conf import PipeStepConfig
from vega.trainer.callbacks import Callback


@ClassFactory.register(ClassType.SEARCHSPACE)
class OperatorSearchSpace(SearchSpace):
    """Operator SearchSpace."""

    @classmethod
    def get_space(self, desc):
        """Get model and input."""
        for hp in desc.get("hyperparameters") or []:
            change_space(hp)
        model = NetworkDesc(PipeStepConfig.model.model_desc).to_model()
        searchable = create_searchable_decorator(model)
        return {"hyperparameters": searchable.search_space()}


@ClassFactory.register(ClassType.CALLBACK)
class OperatorReplaceCallback(Callback):
    """Operator Replace callback."""

    def before_train(self, logs=None):
        """Call before train."""
        searchable = create_searchable_decorator(self.trainer.model)
        searchable.update(self.trainer.hps)


def create_searchable_decorator(model):
    """Create searchable class from model."""
    searchable = SearchableRegister().init()
    searchable.register(Conv2dSearchable)
    searchable.add_search_event(change_module)
    for name, m in model.named_modules():
        searchable.add_space(name, m)
    return searchable


def change_module(model, name, entity):
    """Change module."""
    if not entity:
        return
    tokens = name.split('.')
    attr_name = tokens[-1]
    parent_names = tokens[:-1]
    for s in parent_names:
        model = getattr(model, s)
    setattr(model, attr_name, entity)


@space(
    key='conv',
    type='CATEGORY',
    range=['Conv2d', 'GhostConv2d', 'SeparableConv2d'])
class Conv2dSearchable(Searchable):
    """Searchable class of Conv2d."""

    def search_on(self, module):
        """Call search on function."""
        return module.__class__.__name__ == 'Conv2d'

    def __call__(self, module):
        """Call searchable."""
        cls = ClassFactory.get_cls(ClassType.NETWORK, self.desc)
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
        return cls(in_channels, out_channels, kernel_size, stride, padding)
