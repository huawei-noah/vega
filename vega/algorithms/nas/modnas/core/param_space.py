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

"""Architecture Parameter Space."""
from collections import OrderedDict
from modnas.utils.logging import get_logger
from . import singleton


logger = get_logger(__name__)


@singleton
class ParamSpace():
    """Parameter Space class."""

    def __init__(self):
        self.reset()

    def reset(self, ):
        """Reset parameter space."""
        self._param_id = -1
        self._params_map = OrderedDict()
        self._categorical_length = None

    def register(self, param, name):
        """Register a new parameter."""
        param.pid = self.new_param_id()
        if name is None:
            reg_name = self.new_param_name(param)
        else:
            reg_name = name
            idx = 0
            while reg_name in self._params_map:
                idx += 1
                reg_name = '{}_{}'.format(name, idx)
        param.name = reg_name
        self.add_param(reg_name, param)
        if param.TYPE == 'C':
            self._categorical_length = None

    def new_param_id(self, ):
        """Return a new parameter id."""
        self._param_id += 1
        return self._param_id

    def new_param_name(self, param):
        """Return a new parameter name."""
        prefix = param.__class__.__name__[0].lower()
        return '{}_{}'.format(prefix, self._param_id)

    def params(self, ):
        """Return an iterator over parameters."""
        for p in self._params_map.values():
            yield p

    def named_params(self, ):
        """Return an iterator over named parameters."""
        for n, p in self._params_map.items():
            yield n, p

    def named_param_values(self, ):
        """Return an iterator over named parameter values."""
        for n, p in self._params_map.items():
            yield n, p.value()

    def add_param(self, name, param):
        """Add a parameter to space."""
        self._params_map[name] = param

    def get_param(self, name):
        """Return a parameter by name."""
        return self._params_map.get(name, None)

    def categorical_size(self, ):
        """Return size of the categorical parameter space."""
        if self._categorical_length is None:
            prod = 1
            for x in self.categorical_params():
                prod *= len(x)
            self._categorical_length = prod
        return self._categorical_length

    def categorical_params(self, ):
        """Return an iterator over categorical parameters."""
        for p in self.params():
            if p.TYPE == 'C':
                yield p

    def tensor_params(self, ):
        """Return an iterator over tensor parameters."""
        for p in self.params():
            if p.TYPE == 'T':
                yield p

    def categorical_values(self, ):
        """Return an iterator over categorical parameters values."""
        for p in self.params():
            if p.TYPE == 'C':
                yield p.value()

    def tensor_values(self, ):
        """Return an iterator over tensor parameters values."""
        for p in self.params():
            if p.TYPE == 'T':
                yield p.value()

    def get_categorical_params(self, idx):
        """Return a set of parameter values from a categorical space index."""
        arch_param = OrderedDict()
        for ap in self.categorical_params():
            ap_dim = len(ap)
            arch_param[ap.name] = ap.get_value(idx % ap_dim)
            idx //= ap_dim
        return arch_param

    def get_categorical_index(self, param):
        """Return a categorical space index from a set of parameter values."""
        idx = 0
        base = 1
        for p in self.categorical_params():
            p_dim = len(p)
            p_idx = p.get_index(param[p.name])
            idx += base * p_idx
            base *= p_dim
        return idx

    def update_params(self, pmap):
        """Update parameter values from a dict."""
        for k, v in pmap.items():
            p = self.get_param(k)
            if p is None:
                logger.error('parameter \'{}\' not found'.format(k))
            p.set_value(v)

    def on_update_tensor_params(self):
        """Invoke handlers on tensor parameter updates."""
        for ap in self.tensor_params():
            ap.on_update()
