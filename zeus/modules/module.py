# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Fine Grained SearchSpace Define."""

from copy import deepcopy
from collections import OrderedDict
from zeus.common import ClassFactory, ClassType
from zeus.modules.operators import ops
from zeus.modules.operators.functions.serializable import ModuleSerializable


@ClassFactory.register(ClassType.NETWORK)
class Module(ModuleSerializable, ops.Module):
    """FineGrainedSpace."""

    def __init__(self, *args, **kwargs):
        super(Module, self).__init__()
        self._losses = OrderedDict()
        self._add_pretrained_hook()

    def set_module(self, names, layer):
        """Set Models by name."""
        parent_model = self
        if not isinstance(names, list):
            names_path = names.split('.')
        else:
            names_path = deepcopy(names)
        next_names = names_path.pop(0)
        if not names_path:
            self.add_module(names[0], layer)
        else:
            next_model = getattr(parent_model, next_names)
            next_model.set_module(names_path, layer)

    def add_loss(self, loss):
        """Add a loss function into module."""
        self._losses[loss.__class__.__name__] = loss

    @property
    def pretrained_hook(self):
        """Define pretrained hook function or pertrained file path."""
        return None

    def _add_pretrained_hook(self):
        if self.pretrained_hook is None:
            return
        if isinstance(self.pretrained_hook, str):
            hook = ClassFactory.get_cls(ClassType.PRETRAINED_HOOK, self.pretrained_hook)
        else:
            hook = self.pretrained_hook
        self._register_load_state_dict_pre_hook(hook)

    @property
    def loss(self):
        """Define loss name or class."""
        return None

    @property
    def out_channels(self):
        """Output Channel for Module."""
        return [conv.out_channels for name, conv in self.named_modules() if
                isinstance(conv, ops.Conv2d) or conv.__class__.__name__ == 'Conv2d'][-1]

    def _create_loss(self):
        """Create loss class."""
        if self.loss is None:
            return
        loss_cls = ClassFactory.get_cls(ClassType.LOSS, self.loss)
        desc = self.desc.get('loss')
        loss_obj = loss_cls(**desc) if desc is not None else loss_cls()
        self.add_loss(loss_obj)

    def overall_loss(self):
        """Call loss function, default sum all losses."""
        self._create_loss()
        from zeus.modules.loss.multiloss import MultiLoss
        return MultiLoss(*list(self._losses.values()))
