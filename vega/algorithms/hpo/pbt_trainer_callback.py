# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer callback for pbt."""
import logging
from vega.common.class_factory import ClassFactory, ClassType
from vega.trainer.callbacks import Callback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class PbtTrainerCallback(Callback):
    """Construct the trainer of Adelaide-EA."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.epochs = self.trainer.epochs
        self.params_list = self.trainer.hps.trainer.all_configs
        self.load_para_interval = self.epochs // len(self.params_list.keys())

    def before_epoch(self, epoch, logs=None):
        """Be called before epoch."""
        config_id = str(epoch // self.load_para_interval)
        cur_config = self.params_list[config_id]
        for key, value in cur_config.items():
            para_name = key.split(".")[-1]
            if "optimizer" in key:
                self.trainer.optimizer.param_groups[0][para_name] = value
