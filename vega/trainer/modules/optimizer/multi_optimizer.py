# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Manage LrScheduler class."""
from collections import OrderedDict
from vega.common import ClassFactory, ClassType
from vega.trainer.modules.lr_schedulers import LrScheduler
from vega.common.config import Config
from .optim import Optimizer
from ..conf.optim import OptimConfig


@ClassFactory.register(ClassType.OPTIMIZER)
class MultiOptimizers(object):
    """Register and call multi-optimizer class."""

    config = OptimConfig()

    def __init__(self, config=None):
        """Initialize."""
        self.is_multi_opt = True
        if config is not None:
            self.config = Config(config)
        self._opts = OrderedDict()

    def __call__(self, model=None, distributed=False):
        """Call Optimizer class."""
        for config in self.config:
            name = config.get('model')
            sub_model = getattr(model, config.get('model'))
            sub_opt = Optimizer(config)(sub_model, distributed)
            sub_lr = None
            sub_loss = None
            if config.get('lr_scheduler'):
                sub_lr = LrScheduler(config=config.get('lr_scheduler'))(sub_opt)
            if config.get('loss'):
                sub_loss = ClassFactory.get_instance(ClassType.LOSS, config.get('loss'))
            self._opts[name] = dict(opt=sub_opt, lr=sub_lr, loss=sub_loss, model=sub_model)
        return self

    def get_opts(self):
        """Get opts."""
        return self._opts.items()
