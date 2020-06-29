# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""DARTS FUll trainer."""
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.datasets.pytorch import Dataset
from vega.core.common import FileOps, Config, DefaultConfig
from vega.core.trainer.callbacks import Callback


@ClassFactory.register(ClassType.CALLBACK)
class DartsFullTrainerCallback(Callback):
    """A special callback for CARSFullTrainer."""

    def __init__(self):
        super(DartsFullTrainerCallback, self).__init__()

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        self.trainer.model.drop_path_prob = self.trainer.cfg.drop_path_prob * epoch / self.trainer.cfg.epochs
