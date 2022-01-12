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
        self.transforms = self.trainer.hps.dataset.transforms
        self.transform_interval = self.trainer.epochs // len(self.transforms[0]['all_para'].keys())
        self.hps = self.trainer.hps

    def before_epoch(self, epoch, logs=None):
        """Be called before epoch."""
        config_id = str(epoch // self.transform_interval)
        transform_list = self.transforms[0]['all_para'][config_id]
        self.hps.dataset.transforms[0]['para_array'] = transform_list
        self.trainer.train_loader = self.trainer._init_dataloader(mode='train')
        self.trainer.valid_loader = self.trainer._init_dataloader(mode='val')
