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

"""The example of training model."""

import logging
import vega


logging.info("load trial")
trial = vega.TrialAgent()

logging.info("create model")
vega.set_backend("pytorch", "GPU")
model = vega.get_network("ResNet", depth=18).cuda()

logging.info("load dataset")
train_loader = vega.get_dataset("Cifar10", data_path="/cache/datasets/cifar10", mode="train", batch_size=256).loader
test_loader = vega.get_dataset("Cifar10", data_path="/cache/datasets/cifar10", mode="test", batch_size=256).loader

logging.info("create trainer")
trainer = vega.get_trainer(model=model, id=trial.worker_id, hps=trial.hps)
trainer.config.mixup = True
trainer.train_loader = train_loader
trainer.valid_loader = test_loader

logging.info("start training")
trainer.train_process()
logging.info("training is complete, output folder: {}".format(trainer.get_local_worker_path()))
