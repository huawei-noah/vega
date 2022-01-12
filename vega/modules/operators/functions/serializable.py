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

"""Convert class to desc."""
from copy import deepcopy
import hashlib
import json
from inspect import signature as sig
from collections import OrderedDict
from vega.common.config import Config
from vega.common import ClassFactory, ClassType, SearchSpaceType


class Serializable(object):
    """Base class of Serializable."""

    def __new__(cls, *args, **kwargs):
        """Record params."""
        desc = {}
        params_sig = sig(cls.__init__).parameters
        param_names = list(params_sig.keys())
        if len(param_names) > len(args):
            # not dynamic parameter for connections
            for idx, arg in enumerate(args):
                arg_name = param_names[idx + 1]
                desc[arg_name] = arg
        if kwargs:
            desc.update(kwargs)
        instance = super(Serializable, cls).__new__(cls)
        instance.desc = Config(desc) if desc else {}
        return instance

    def to_desc(self, recursion=None):
        """Convert Module to desc dict."""
        raise NotImplementedError

    @classmethod
    def from_desc(cls, desc):
        """Create Model from desc."""
        raise NotImplementedError

    @classmethod
    def from_module(cls, module):
        """Create Model from module."""
        raise NotImplementedError

    @property
    def sha256(self):
        """SHA256 value of network description."""
        return self.get_sha256(self.to_desc(False))

    @classmethod
    def get_sha256(cls, desc):
        """Get desc's short sha256 code.

        :param desc: network description.
        :type desc: str.
        :return: short sha256 code.
        :rtype: str.

        """
        code = hashlib.sha256(json.dumps(desc, sort_keys=True).encode('utf-8')).hexdigest()
        return code[:8]

    @property
    def model_name(self):
        """Return model name."""
        return self.__class__.__name__

    @property
    def _arch_params(self):
        """Get Arch params."""
        return ArchParams._values

    def set_arch_params(self, value):
        """Set Arch params."""
        ArchParams._values = Config(value)

    @property
    def _arch_params_type(self):
        """Get Arch params."""
        return ArchParams._arch_type

    def clear_arch_params(self):
        """Clear Arch params."""
        return ArchParams.clear()

    @property
    def module_arch_params(self):
        """Get Arch params."""
        return Config(
            {k.split('.')[-1]: v for k, v in ArchParams._values.items() if '.'.join(k.split('.')[:-1]) == self.name})

    @staticmethod
    def clear_module_arch_params(module, fea_in, fea_out):
        """Get Arch params."""
        ArchParams._values = {k: v for k, v in ArchParams._values.items() if '.'.join(k.split('.')[:-1]) != module.name}
        return None

    def define_arch_params(self, key, default_value, dtype=None, params=None):
        """Define a prop and get value."""
        value = self.desc.get(key) or default_value
        return ArchParams(key, value, dtype, params).value


class OperatorSerializable(Serializable):
    """Seriablizable for Operator class."""

    def to_desc(self, recursion=None):
        """Convert Operator to desc dict.

        :param level: Specifies witch level to convert. all conversions are performed as default.
        """
        desc = dict(type=self.model_name)
        if hasattr(self, 'name') and getattr(self, 'name'):
            desc.update({'name': getattr(self, 'name')})
        for k, v in self.desc.items():
            if hasattr(self, k):
                src_v = getattr(self, k)
                if src_v and src_v != v:
                    self.desc[k] = src_v
        desc.update(self.desc)
        return desc

    @classmethod
    def from_desc(cls, desc):
        """Create Operator class by desc."""
        return ClassFactory.get_instance(ClassType.NETWORK, desc)

    @classmethod
    def from_module(cls, module):
        """Create Operator class by module."""
        return module


class ModuleSerializable(Serializable):
    """Seriablizable Module class."""

    def to_desc(self, recursion=True):
        """Convert Module to desc dict.

        :param level: Specifies witch level to convert. all conversions are performed as default.
        """
        if not recursion:
            if 'modules' in self.desc:
                return self.desc
            else:
                return dict(self.desc, **dict(type=self.model_name))
        desc = {}
        if getattr(self, 'loss') and self.loss:
            desc['loss'] = self.loss
        for name, module in self.named_children():
            if hasattr(module, 'to_desc'):
                sub_desc = module.to_desc()
            else:
                sub_desc = {'type': module.__class__.__name__}
            if 'modules' in desc:
                desc['modules'].append(name)
            else:
                desc['modules'] = [name]
            desc[name] = sub_desc
        return desc

    def update_from_desc(self, desc):
        """Update desc according to desc."""
        for key, value in desc.items():
            if key == 'type' or not hasattr(self, key):
                continue
            child_module = getattr(self, key)
            if hasattr(child_module, 'add_module'):
                self.add_module(key, value)
            else:
                child_module.update_from_desc(value)

    @classmethod
    def from_desc(cls, desc):
        """Create Model from desc."""
        desc = deepcopy(desc)
        module_groups = desc.get('modules', [])
        module_type = desc.get('type', 'Sequential')
        loss = desc.get('loss')
        if '_arch_params' in desc:
            arch_params = desc.pop('_arch_params')
            arch_type = list(arch_params.keys())[0]
            ArchParams._arch_type = arch_type
            ArchParams.update(arch_params.get(arch_type))
        modules = OrderedDict()
        for group_name in module_groups:
            module_desc = deepcopy(desc.get(group_name))
            if not module_desc:
                continue
            if 'modules' in module_desc:
                module = cls.from_desc(module_desc)
            else:
                cls_name = module_desc.get('type')
                if not ClassFactory.is_exists(ClassType.NETWORK, cls_name):
                    raise ValueError("Network {} not exists.".format(cls_name))
                module = ClassFactory.get_instance(ClassType.NETWORK, module_desc)
            modules[group_name] = module
            module.name = str(group_name)
        if not module_groups and module_type:
            model = ClassFactory.get_instance(ClassType.NETWORK, desc)
        else:
            if ClassFactory.is_exists(SearchSpaceType.CONNECTIONS, module_type):
                connections = ClassFactory.get_cls(SearchSpaceType.CONNECTIONS, module_type)
            else:
                connections = ClassFactory.get_cls(SearchSpaceType.CONNECTIONS, 'Sequential')
            model = list(modules.values())[0] if len(modules) == 1 else connections(modules)
        if loss:
            model.add_loss(ClassFactory.get_cls(ClassType.LOSS, loss))
        return model

    @classmethod
    def from_module(cls, module):
        """From Model."""
        name = module.__class__.__name__
        if ClassFactory.is_exists(ClassType.NETWORK, name):
            module_cls = ClassFactory.get_cls(ClassType.NETWORK, name)
            if hasattr(module_cls, "from_module"):
                return module_cls.from_module(module)
        return module


class ArchParams(object):
    """Set proxy property in module.

    When a variable is changed by an external program, the value of the invoker can be changed synchronously.
    """

    _values = {}
    _arch_type = None

    def __init__(self, key, default_value, d_type=None, params=None):
        self.key = key
        self.default_value = default_value
        self.d_type = d_type
        self.params = params
        self._add_prop()
        self._check()

    def _add_prop(self):
        self._values[self.key] = self.default_value

    @property
    def value(self):
        """Get values."""
        value = self._values.get(self.key)
        if self.d_type == ClassType.NETWORK:
            if isinstance(value, str):
                cls = ClassFactory.get_cls(ClassType.NETWORK, value)
                value = cls() if self.params is None else cls(**self.params)
            else:
                if self.params:
                    value = ClassFactory.get_instance(ClassType.NETWORK, value, **self.params)
                else:
                    value = ClassFactory.get_instance(ClassType.NETWORK, value)
        return value

    def _check(self):
        if self.d_type is None:
            return

    @classmethod
    def _flatten(cls, value, name=None):
        dict_values = {}
        if isinstance(value, dict):
            for k, v in value.items():
                dict_values.update(cls._flatten(v, "{}.{}".format(name, k) if name else k))
        else:
            dict_values[name] = value
        return dict_values

    @classmethod
    def update(cls, value):
        """Update value."""
        value = cls._flatten(value)
        cls._values.update(value)

    @classmethod
    def clear(self):
        """Reset arch desc."""
        self._values = {}
