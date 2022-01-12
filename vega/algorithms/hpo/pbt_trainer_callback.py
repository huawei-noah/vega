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
        self.params_list = self.trainer.hps.trainer.all_configs
        self.load_para_interval = self.trainer.epochs // len(self.params_list.keys())

    def before_epoch(self, epoch, logs=None):
        """Be called before epoch."""
        config_id = str(epoch // self.load_para_interval)
        cur_config = self.params_list[config_id]
        for key, value in cur_config.items():
            para_name = key.split(".")[-1]
            if "optimizer" in key:
                self.trainer.optimizer.param_groups[0][para_name] = value
