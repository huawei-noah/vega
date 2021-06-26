# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Basic Warm up lr scheduler.

Example:
    >>> # in yml file `trainer` part
    >>> # use WarmupScheduler and MultiStepLR as after_scheduler
    >>> lr_scheduler:
    >>>     type: WarmupScheduler
    >>>     by_epoch: False
    >>>     params:
    >>>         warmup_type: linear | constant | exp
    >>>         warmup_iters: 20
    >>>         warmup_ratio: 0.1
    >>>         after_scheduler_config:
    >>>             by_epoch: False
    >>>             type: MultiStepLR
    >>>             params:
    >>>                 milestones: [60, 120]
    >>>                 gamma: 0.5

"""

from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class WarmupScheduler(object):
    """Multiple Step learning rate with warm up.

    :param milestones: list of decay epochs
    :type milestones: list of int
    :param decay_rates: list of decay rates
    :type decay_rates: list of float
    :param warmup: whether to warm up
    :type warmup: bool
    :param epoch_steps: steps in one epoch
    :type epoch_steps: int
    """

    def __init__(self,
                 optimizer,
                 warmup_type='linear',
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 after_scheduler_config=None,
                 **kwargs):
        """Init WarmupScheduler."""
        _type = after_scheduler_config["type"]
        if _type == "MultiStepLR":
            from .multistep import MultiStepLR
            self.lr = MultiStepLR(
                optimizer,
                milestones=after_scheduler_config["params"]["milestones"],
                gamma=after_scheduler_config["params"]["gamma"],
                warmup=True,
                warmup_epochs=warmup_iters
            )
        elif _type == "CosineAnnealingLR":
            from .cosine_annealing import CosineAnnealingLR
            self.lr = CosineAnnealingLR(
                optimizer,
                T_max=after_scheduler_config["params"]["T_max"],
                eta_min=after_scheduler_config["params"].get("eta_min", 0),
                last_epoch=after_scheduler_config["params"].get("last_epoch", -1),
                warmup=True,
                warmup_epochs=warmup_iters
            )
        else:
            raise("Not supported lr scheduler: {}".format(_type))

    def get_lr(self):
        """Get lr."""
        return self.lr.get_lr()

    def step(self, epoch=None):
        """Step forward for current scheduler."""
        self.lr.step(epoch)
