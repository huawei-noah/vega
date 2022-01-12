# -*- coding: utf-8 -*-

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

"""DARTS FUll trainer."""
from vega.common import ClassFactory, ClassType
from vega.trainer.callbacks import Callback


@ClassFactory.register(ClassType.CALLBACK)
class DartsFullTrainerCallback(Callback):
    """A special callback for CARSFullTrainer."""

    def __init__(self):
        super(DartsFullTrainerCallback, self).__init__()

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        self.trainer.config.report_on_epoch = True
        self.trainer.model.drop_path_prob = self.trainer.config.drop_path_prob * epoch / self.trainer.config.epochs
