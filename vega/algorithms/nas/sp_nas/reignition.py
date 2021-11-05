# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""This is SPNAS Reignition Callback."""
import logging
import copy
from vega.common import ClassFactory, ClassType
from vega.trainer.callbacks.callback import Callback
from vega.report import ReportClient


@ClassFactory.register(ClassType.CALLBACK)
class ReignitionCallback(Callback):
    """Reignition callback."""

    def __init__(self):
        """Initialize callback."""
        super(ReignitionCallback, self).__init__()
        self.priority = 100
        self.desc_copy = None

    def init_trainer(self, logs=None):
        """Be called before train."""
        logging.info("Start SPNas Reigniting.")
        self.desc_copy = copy.deepcopy(self.trainer.model_desc)
        backbone = self.desc_copy.get('backbone')
        code = backbone.get('code')
        self.trainer.model_desc = dict(type='SerialClassificationNet', code=code)

    def after_epoch(self, epoch, logs=None):
        """Save desc into FasterRCNN."""
        self.desc_copy['backbone']['weight_file'] = self.trainer.weights_file
        self.trainer.model_desc = self.desc_copy
        ReportClient().update(self.trainer.step_name, self.trainer.worker_id, desc=self.desc_copy)
