# -*- coding:utf-8 -*-

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

"""The trainer program for ESR_EA."""
from vega.common import ClassFactory, ClassType
from vega.trainer.callbacks import Callback


@ClassFactory.register(ClassType.CALLBACK)
class ESRTrainerCallback(Callback):
    """Construct the trainer of ESR-EA."""

    def before_train(self, epoch, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        # Use own save checkpoint and save performance function
        self.trainer.auto_save_ckpt = False
        self.trainer.auto_save_perf = False
