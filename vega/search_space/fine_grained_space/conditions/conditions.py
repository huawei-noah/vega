# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for conditions."""
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.fine_grained_space.fine_grained_space import FineGrainedSpace, FineGrainedSpaceFactory
from vega.core.common.utils import update_dict_with_flatten_keys


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Repeat(FineGrainedSpace):
    """Repeat SearchSpace."""

    def __init__(self, num_reps, items, ref):
        """Make a search space obj repeat with nums.

        :param num_reps: repeat num.
        :param items: search space items.
        :return: ref
        """
        super(Repeat, self).__init__(num_reps, items, ref)
        repeat_items = deepcopy(items)
        ref_copy = deepcopy(ref)
        for idx in range(num_reps):
            # TODO: using desc language
            params = {}
            for key, values in repeat_items.items():
                if not isinstance(values, list):
                    params[key] = values
                else:
                    v_idx = idx if len(values) > idx else -1
                    params[key] = values[v_idx]
            params = update_dict_with_flatten_keys(ref_copy, params)
            name, search_space_obj = create_search_space(params)
            self.add('layers_{}'.format(idx), search_space_obj)


def create_search_space(model):
    """Create search space from model or desc."""
    if isinstance(model, FineGrainedSpace):
        return model.__class__.__name__, model.to_model()
    elif isinstance(model, dict):
        cls = FineGrainedSpaceFactory.from_desc(model)
        if isinstance(cls, FineGrainedSpace):
            cls = cls.to_model()
        return model.get('type'), cls
    else:
        return model.__class__.__name__, model


class Condition(nn.Module):
    """Base module for conditions."""

    def __init__(self, *models):
        self.models = models if isinstance(models, tuple) else tuple(models)
        super(Condition, self).__init__()
        self.build()

    def build(self):
        """Build models."""
        for _idx, model in enumerate(self.models):
            model_name, model = create_search_space(model)
            self.add_module('{}_{}'.format(model_name, _idx), model)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Sequential(Condition):
    """Sequential SearchSpace."""

    def __init__(self, *models, out_list=None):
        super(Sequential, self).__init__(*models)
        self.out_list = out_list

    def forward(self, x):
        """Forward x."""
        output = x
        models = self.children()
        if self.out_list is None:
            for model in models:
                output = model(output)
        else:
            outputs = []
            models = list(models)
            for idx, model in enumerate(models):
                output = model(output)
                if idx in self.out_list:
                    outputs.append(output)
            output = outputs
        return output


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Add(Condition):
    """Create Lambda for forward x."""

    def __init__(self, *models):
        super(Add, self).__init__(*models)

    def forward(self, x):
        """Forward x."""
        output = None
        for model in self.children():
            if output is None:
                output = model(x)
            else:
                output += model(x)
        return output


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Append(Condition):
    """Create Lambda for forward x."""

    def __init__(self, *models):
        super(Append, self).__init__(*models)

    def forward(self, x):
        """Forward x."""
        output = x
        outputs = []
        for model in self.children():
            output = model(output)
            outputs.append(output)
        return tuple(outputs)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Concat(Condition):
    """Create Lambda for forward x."""

    def __init__(self, *models):
        super(Concat, self).__init__(*models)

    def forward(self, x):
        """Forward x."""
        outputs = []
        for model in self.children():
            outputs.append(model(x))
        return torch.cat(outputs, 1)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Interpolate(Condition):
    """Create Lambda for forward x."""

    def __init__(self, *models):
        super(Interpolate, self).__init__(*models)

    def forward(self, x):
        """Forward x."""
        output = input
        for model in self.children():
            output = model(output)
        output = F.interpolate(output, size=x.size()[2:], mode='bilinear', align_corners=True)
        return output


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Merge(Condition):
    """Create Lambda for forward x."""

    def __init__(self, *models):
        super(Merge, self).__init__(*models)

    def forward(self, x):
        """Forward x."""
        output = x
        outputs = []
        for model in self.children():
            model_out = model(output)
            if isinstance(model_out, tuple):
                model_out = list(model_out)
                outputs.extend(model_out)
            else:
                outputs.append(model_out)
            output = tuple(outputs)
        return output


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Map(Condition):
    """Create Lambda for forward x."""

    def __init__(self, *models):
        super(Map, self).__init__(*models)

    def forward(self, x):
        """Forward x."""
        models = self.children()
        model = next(models)
        map_results = [model(data) for data in x]
        output = tuple(map_results)
        return output


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Tuple(Condition):
    """Create Lambda for forward x."""

    def __init__(self, *models):
        super(Tuple, self).__init__(*models)

    def forward(self, x):
        """Forward x."""
        models = self.children()
        outputs = []
        for model in models:
            outputs.append(model(x))
        return tuple(outputs)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Process_list(Condition):
    """Process_list Condition."""

    def __init__(self, *models, out_list=None):
        super(Process_list, self).__init__(*models)
        self.out_list = out_list

    def forward(self, input):
        """Call list of input train forward function."""
        if self.out_list is None:
            if isinstance(input, list):
                outputs = []
                for model, idx in zip(self.children(), [i for i in range(len(input))]):
                    output = model(input[idx])
                    outputs.append(output)
                output = outputs
            else:
                raise ValueError("Input must list!")
        else:
            input = list(input)
            for model, idx in zip(self.children(), self.out_list):
                if isinstance(idx, list):
                    assert len(idx) == 2
                    output = model(input[idx[0]], input[idx[1]])
                    input.append(output)
                else:
                    input.append(model(input[idx]))
            output = input
        return output


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Cells(Condition):
    """Cells Condition."""

    def __init__(self, items, ref, C_curr, C, genotype_size, num_ops):
        models = [ref.get(item) for item in items]
        self.C_prev_prev = C_curr
        self.C_prev = C_curr
        self.C_curr = C
        self.k = genotype_size
        self.num_ops = num_ops
        super(Cells, self).__init__(*tuple(models))
        self._initialize_alphas()

    def build(self):
        """Build cell."""
        reduction_prev = True if self.C_curr == self.C_prev else False
        for idx, model in enumerate(self.models):
            model_name, model = create_search_space(model)
            model.name = model_name
            if model_name == 'reduce':
                self.C_curr *= 2
                reduction = True
            else:
                reduction = False
            model.reduction_prev = reduction_prev
            model.C_prev_prev = self.C_prev_prev
            model.C_prev = self.C_prev
            model.C = self.C_curr
            model.reduction = reduction
            model = model.build()
            reduction_prev = reduction
            multiplier = model.concat_size
            self.C_prev_prev, self.C_prev = self.C_prev, multiplier * self.C_curr
            self.add_module('{}_{}'.format(model_name, idx), model)

    def _initialize_alphas(self):
        """Initialize architecture parameters."""
        self.register_buffer('alphas_normal',
                             (1e-3 * torch.randn(self.k, self.num_ops)).cuda().requires_grad_())
        self.register_buffer('alphas_reduce',
                             (1e-3 * torch.randn(self.k, self.num_ops)).cuda().requires_grad_())
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    @property
    def arch_parameters(self):
        """Abstract base function of getting arch parameters."""
        return self._arch_parameters

    @property
    def arch_weights(self):
        """Get arch weights."""
        weights_normal = F.softmax(
            self.alphas_normal, dim=-1).data.cpu().numpy()
        weights_reduce = F.softmax(
            self.alphas_reduce, dim=-1).data.cpu().numpy()
        return [weights_normal, weights_reduce]

    def forward(self, input):
        """Call list of input train forward function."""
        s0, s1 = input
        for model in self.children():
            if model.search:
                if model.name == 'reduce':
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            else:
                weights = None
            s0, s1 = s1, model(s0, s1, weights)
        return s1
