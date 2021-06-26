# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer program for pba."""
import logging
from vega.common.class_factory import ClassFactory, ClassType
from vega.trainer.callbacks import Callback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class PbaTrainerCallback(Callback):
    """Construct the trainer of Adelaide-EA."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.epochs = self.trainer.epochs
        self.transforms = self.trainer.hps.dataset.transforms
        self.transform_interval = self.epochs // len(self.transforms[0]['all_para'].keys())
        self.hps = self.trainer.hps

    def before_epoch(self, epoch, logs=None):
        """Be called before epoch."""
        config_id = str(epoch // self.transform_interval)
        transform_list = self.transforms[0]['all_para'][config_id]
        self.hps.dataset.transforms[0]['para_array'] = transform_list
        self.trainer.train_loader = self.trainer._init_dataloader(mode='train')
        self.trainer.valid_loader = self.trainer._init_dataloader(mode='val')
