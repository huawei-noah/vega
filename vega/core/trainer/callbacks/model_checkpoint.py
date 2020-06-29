# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ModelCheckpoint callback defination."""
from .callbacks import Callback
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class ModelCheckpoint(Callback):
    """Callback that saves the evaluated Performance."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.is_chief = self.params['is_chief']
        self.do_validation = self.params['do_validation']

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        if self.is_chief:
            self.trainer._save_checkpoint(epoch)
            if not self.trainer.cfg.get('save_best_model', False):
                return
            self.performance = logs.get('summary_perfs', None)
            best_changed = self.performance['best_valid_perfs_changed']
            if best_changed:
                self.trainer.output_model()
