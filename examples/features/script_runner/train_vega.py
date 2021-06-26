# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The example of training model."""

import logging
import vega


logging.info("load trial")
trial = vega.TrialAgent()

logging.info("create model")
vega.set_backend("pytorch", "GPU")
model = vega.network("ResNet", depth=18).cuda()

logging.info("load dataset")
train_loader = vega.dataset("Cifar10", data_path="/cache/datasets/cifar10", mode="train", batch_size=256).loader
test_loader = vega.dataset("Cifar10", data_path="/cache/datasets/cifar10", mode="test", batch_size=256).loader

logging.info("create trainer")
trainer = vega.trainer(model=model, id=trial.worker_id, hps=trial.hps)
trainer.config.mixup = True
trainer.train_loader = train_loader
trainer.valid_loader = test_loader

logging.info("start training")
trainer.train_process()
logging.info("training is complete, output folder: {}".format(trainer.get_local_worker_path()))
