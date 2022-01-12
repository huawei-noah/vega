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

"""Running Horovod Train."""

import os
import argparse
import logging
import horovod.torch as hvd
from vega.common import ClassFactory
from vega.common.general import General
from vega.core.pipeline.conf import PipeStepConfig
from vega.common import FileOps


parser = argparse.ArgumentParser(description='Horovod Fully Train')
parser.add_argument('--cf_file', type=str, help='ClassFactory pickle file')
args = parser.parse_args()

logging.info('start horovod setting')
hvd.init()
hvd.join()
cf_content = FileOps.load_pickle(args.cf_file)
model_desc = cf_content.get('model_desc')
worker_id = cf_content.get('worker_id')
ClassFactory.__registry__ = cf_content.get('registry')
General.from_dict(cf_content.get('general_config'))
PipeStepConfig.from_dict(cf_content.get('pipe_step_config'))
cls_trainer = ClassFactory.get_cls('trainer', "Trainer")

device_id = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[hvd.local_rank()]
os.environ["CUDA_VISIBLE_DEVICES"] = device_id

trainer = cls_trainer(model_desc=model_desc, id=worker_id, horovod=True)
trainer.train_process()
