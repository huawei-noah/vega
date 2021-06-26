# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Manage LrScheduler class."""

from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class SchedulerDict(object):
    """Register and call VAEGANoptimizer class."""

    def __init__(self, optimizer, cfg):
        """Initialize."""
        self.cfg = cfg
        self.optimizer = optimizer
        for item in cfg['modules']:
            sub_optimizer_name = cfg[item]['optimizer']
            tem_lr_scheduler = ClassFactory.get_cls(
                ClassType.LR_SCHEDULER, cfg[item].type)
            sub_optimizer = getattr(optimizer, sub_optimizer_name)
            params = cfg[item].get("params", {})
            setattr(self, item, tem_lr_scheduler(sub_optimizer, **params))
