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

"""Elastic spatial (width) transformations."""
from typing import Callable, Iterator, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.module import Module
from .modifier import modify_param, modify_buffer, modify_attr,\
    restore_module_states, get_ori_buffer


def _conv2d_fan_out_trnsf(m: nn.Conv2d, idx: Tensor) -> None:
    modify_param(m, 'weight', m.weight[idx, :, :, :])
    if m.bias is not None:
        modify_param(m, 'bias', m.bias[idx])
    if m.groups != 1 and m.weight.shape[1] == 1:
        width = idx.stop - idx.start if isinstance(idx, slice) else len(idx)
        modify_attr(m, 'groups', width)


def _conv2d_fan_in_trnsf(m: nn.Conv2d, idx: Tensor) -> None:
    bias_idx = None
    if m.groups == 1:
        modify_param(m, 'weight', m.weight[:, idx, :, :])
    elif m.weight.shape[1] == 1:
        width = idx.stop - idx.start if isinstance(idx, slice) else len(idx)
        modify_param(m, 'weight', m.weight[idx, :, :, :])
        modify_attr(m, 'groups', width)
        bias_idx = idx
    else:
        raise NotImplementedError
    if m.bias is not None and bias_idx is not None:
        modify_param(m, 'bias', m.bias[bias_idx])


def _batchnorm2d_fan_in_out_trnsf(m: nn.BatchNorm2d, idx: Tensor) -> None:
    if m.weight is not None:
        modify_param(m, 'weight', m.weight[idx])
    if m.bias is not None:
        modify_param(m, 'bias', m.bias[idx])
    modify_buffer(m, 'running_mean', m.running_mean[idx])
    modify_buffer(m, 'running_var', m.running_var[idx])


def _batchnorm2d_fan_in_out_post_trnsf(m: nn.BatchNorm2d, idx: Tensor) -> None:
    if isinstance(idx, slice):
        return
    get_ori_buffer(m, 'running_mean')[idx] = m.running_mean
    get_ori_buffer(m, 'running_var')[idx] = m.running_var


_fan_out_transform = {
    nn.Conv2d: _conv2d_fan_out_trnsf,
    nn.BatchNorm2d: _batchnorm2d_fan_in_out_trnsf,
}

_fan_in_transform = {
    nn.Conv2d: _conv2d_fan_in_trnsf,
    nn.BatchNorm2d: _batchnorm2d_fan_in_out_trnsf,
}

_fan_out_post_transform = {
    nn.BatchNorm2d: _batchnorm2d_fan_in_out_post_trnsf,
}

_fan_in_post_transform = {
    nn.BatchNorm2d: _batchnorm2d_fan_in_out_post_trnsf,
}


def get_fan_out_transform(mtype: Type) -> Callable:
    """Return the fan out transform of a module type."""
    return _fan_out_transform.get(mtype, None)


def get_fan_in_transform(mtype: Type) -> Callable:
    """Return the fan in transform of a module type."""
    return _fan_in_transform.get(mtype, None)


def get_fan_out_post_transform(mtype: Type) -> Optional[Callable]:
    """Return the fan out post transform of a module type."""
    return _fan_out_post_transform.get(mtype, None)


def get_fan_in_post_transform(mtype: Type) -> Optional[Callable]:
    """Return the fan in post transform of a module type."""
    return _fan_in_post_transform.get(mtype, None)


def set_fan_out_transform(mtype, transf):
    """Set the fan out transform of a module type."""
    _fan_out_transform[mtype] = transf


def set_fan_in_transform(mtype, transf):
    """Set the fan in transform of a module type."""
    _fan_in_transform[mtype] = transf


def set_fan_out_post_transform(mtype, transf):
    """Set the fan out post transform of a module type."""
    _fan_out_post_transform[mtype] = transf


def set_fan_in_post_transform(mtype, transf):
    """Set the fan in post transform of a module type."""
    _fan_in_post_transform[mtype] = transf


def _hook_module_in(module: Module, inputs: Tuple[Tensor]) -> None:
    fan_in_idx, fan_out_idx = ElasticSpatial.get_spatial_idx(module)
    mtype = type(module)
    trnsf = get_fan_in_transform(mtype)
    if trnsf is not None and fan_in_idx is not None:
        trnsf(module, fan_in_idx)
    trnsf = get_fan_out_transform(mtype)
    if trnsf is not None and fan_out_idx is not None:
        trnsf(module, fan_out_idx)


def _hook_module_out(module: Module, inputs: Tuple[Tensor], outputs: Tensor) -> None:
    fan_in_idx, fan_out_idx = ElasticSpatial.get_spatial_idx(module)
    mtype = type(module)
    trnsf = get_fan_in_post_transform(mtype)
    if trnsf is not None and fan_in_idx is not None:
        trnsf(module, fan_in_idx)
    trnsf = get_fan_out_post_transform(mtype)
    if trnsf is not None and fan_out_idx is not None:
        trnsf(module, fan_out_idx)
    restore_module_states(module)


class ElasticSpatialGroup():
    """Module group with elastic spatial dimensions."""

    def __init__(
        self, fan_out_modules: List[Module], fan_in_modules: List[Module],
        max_width: Optional[int] = None, rank_fn: Optional[Callable] = None
    ) -> None:
        super().__init__()
        if fan_in_modules is None:
            fan_in_modules = []
        if fan_out_modules is None:
            fan_out_modules = []
        self.fan_out_modules = fan_out_modules
        self.fan_in_modules = fan_in_modules
        self.idx_mapping = dict()
        self.rank_fn = rank_fn
        self.cur_rank = None
        self.max_width = max_width
        self.enable_spatial_transform()
        ElasticSpatial.add_group(self)

    def destroy(self):
        """Destroy group."""
        self.reset_spatial_idx()
        self.disable_spatial_transform()

    def add_fan_in_module(self, module):
        """Add fan in module to group."""
        self.fan_in_modules.append(module)

    def add_fan_out_module(self, module):
        """Add fan out module to group."""
        self.fan_out_modules.append(module)

    def add_idx_mapping(self, dest, map_fn):
        """Add spatial index mapping to group."""
        self.idx_mapping[dest] = map_fn

    def map_index(self, idx: Tensor, dest: Module) -> Tensor:
        """Return mapped spatial index to target module."""
        map_fn = self.idx_mapping.get(dest, None)
        if map_fn is None:
            return idx
        return [map_fn(i) for i in idx]

    def enable_spatial_transform(self) -> None:
        """Enable spatial transformation of group modules."""
        for m in self.fan_in_modules + self.fan_out_modules:
            ElasticSpatial.enable_spatial_transform(m)

    def disable_spatial_transform(self):
        """Disable spatial transformation of group modules."""
        for m in self.fan_in_modules + self.fan_out_modules:
            ElasticSpatial.disable_spatial_transform(m)

    def set_width_ratio(self, ratio: float, rank: Optional[List[int]] = None) -> None:
        """Set group width by ratio of the max width."""
        if ratio is None:
            self.reset_spatial_idx()
            return
        if self.max_width is None:
            raise ValueError('max_width not specified')
        width = int(self.max_width * ratio)
        self.set_width(width, rank)

    def set_width(self, width: int, rank: Optional[List[int]] = None) -> None:
        """Set group width."""
        if width is None:
            self.reset_spatial_idx()
            return
        if self.cur_rank is None:
            self.set_spatial_rank()
        rank = rank or self.cur_rank
        if rank is None:
            idx = slice(0, width)
        else:
            idx = rank[:width]
        self.set_spatial_idx(idx)

    def reset_spatial_rank(self):
        """Reset ranking of group spatial dimension."""
        self.cur_rank = None

    def set_spatial_rank(self, rank: Optional[List[int]] = None) -> None:
        """Rank group spatial dimension."""
        if rank is None and self.rank_fn is not None:
            rank = self.rank_fn()
        self.cur_rank = rank

    def set_spatial_idx(self, idx: Tensor) -> None:
        """Set group spatial index."""
        if idx is None:
            self.reset_spatial_idx()
            return
        if isinstance(idx, int):
            idx = [idx]
        for m in self.fan_in_modules:
            m_idx = self.map_index(idx, m)
            ElasticSpatial.set_spatial_fan_in_idx(m, m_idx)
        for m in self.fan_out_modules:
            m_idx = self.map_index(idx, m)
            ElasticSpatial.set_spatial_fan_out_idx(m, m_idx)

    def reset_spatial_idx(self):
        """Reset group spatial index."""
        for m in self.fan_in_modules:
            ElasticSpatial.reset_spatial_fan_in_idx(m)
        for m in self.fan_out_modules:
            ElasticSpatial.reset_spatial_fan_out_idx(m)


def conv2d_rank_weight_l1norm_fan_in(module: nn.Conv2d) -> Tensor:
    """Return the rank of Conv2d weight by L1 norm along fan in dimension."""
    if module.groups == 1:
        sum_dim = 0
    elif module.weight.shape[1] == 1:
        sum_dim = 1
    else:
        raise NotImplementedError
    _, idx = torch.sort(torch.sum(torch.abs(module.weight.data), dim=(sum_dim, 2, 3)), dim=0, descending=True)
    return idx


def conv2d_rank_weight_l1norm_fan_out(module: nn.Conv2d):
    """Return the rank of Conv2d weight by L1 norm along fan out dimension."""
    _, idx = torch.sort(torch.sum(torch.abs(module.weight.data), dim=(1, 2, 3)), dim=0, descending=True)
    return idx


def batchnorm2d_rank_weight_l1norm(module: nn.BatchNorm2d):
    """Return the rank of BatchNorm2d weight by L1 norm."""
    _, idx = torch.sort(torch.abs(module.weight.data), dim=0, descending=True)
    return idx


class ElasticSpatial():
    """Elastic spatial group manager."""

    _module_hooks = dict()
    _groups = list()

    @staticmethod
    def add_group(group: ElasticSpatialGroup) -> None:
        """Add a group."""
        ElasticSpatial._groups.append(group)

    @staticmethod
    def remove_group(group):
        """Remove a group."""
        idx = ElasticSpatial._groups.index(group)
        if not idx == -1:
            group.destroy()
            del ElasticSpatial._groups[idx]

    @staticmethod
    def groups() -> Iterator[ElasticSpatialGroup]:
        """Return an iterator over groups."""
        for g in ElasticSpatial._groups:
            yield g

    @staticmethod
    def num_groups() -> int:
        """Return the number of groups."""
        return len(ElasticSpatial._groups)

    @staticmethod
    def enable_spatial_transform(module: Module) -> None:
        """Enable spatial transformation on a module."""
        if module not in ElasticSpatial._module_hooks:
            h_in = module.register_forward_pre_hook(_hook_module_in)
            h_out = module.register_forward_hook(_hook_module_out)
            ElasticSpatial._module_hooks[module] = (h_in, h_out)

    @staticmethod
    def disable_spatial_transform(module):
        """Disable spatial transformation on a module."""
        if module in ElasticSpatial._module_hooks:
            m_hooks = ElasticSpatial._module_hooks.pop(module)
            for h in m_hooks:
                h.remove()
            del module._spatial_idx

    @staticmethod
    def set_spatial_fan_in_idx(module: Module, idx: Tensor) -> None:
        """Set spatial fan in index of a module."""
        ElasticSpatial.get_spatial_idx(module)[0] = idx

    @staticmethod
    def set_spatial_fan_out_idx(module: Module, idx: Tensor) -> None:
        """Set spatial fan out index of a module."""
        ElasticSpatial.get_spatial_idx(module)[1] = idx

    @staticmethod
    def reset_spatial_fan_in_idx(module):
        """Reset spatial fan in index of a module."""
        ElasticSpatial.get_spatial_idx(module)[0] = None

    @staticmethod
    def reset_spatial_fan_out_idx(module):
        """Reset spatial fan out index of a module."""
        ElasticSpatial.get_spatial_idx(module)[1] = None

    @staticmethod
    def reset_spatial_idx(module):
        """Reset spatial index of a module."""
        module._spatial_idx = [None, None]

    @staticmethod
    def get_spatial_idx(module: Module) -> Union[List[Optional[Tensor]], List[None]]:
        """Get spatial index of a module."""
        if not hasattr(module, '_spatial_idx'):
            module._spatial_idx = [None, None]
        return module._spatial_idx

    @staticmethod
    def set_spatial_idx(module, fan_in, fan_out):
        """Set spatial index of a module."""
        module._spatial_idx = [fan_in, fan_out]
