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
from zeus.trainer.trainer_api import Trainer
from zeus.common.class_factory import ClassFactory, ClassType
from zeus.trainer.trial_agent import TrialAgent
from zeus.datasets import Adapter


logging.info("load trial")
trial = TrialAgent()

logging.info("create model")
vega.set_backend("pytorch", "GPU")
resnet = ClassFactory.get_cls(ClassType.NETWORK, "ResNet")
model = resnet(depth=18).cuda()

logging.info("load dataset")
dataset_cls = ClassFactory.get_cls(ClassType.DATASET, "Cifar10")
train_dataset = dataset_cls(data_path="/cache/datasets/cifar10", mode="train", batch_size=256)
test_dataset = dataset_cls(data_path="/cache/datasets/cifar10", mode="test", batch_size=256)
train_loader = Adapter(train_dataset).loader
test_loader = Adapter(test_dataset).loader

logging.info("create trainer")
trainer = Trainer(model=model, id=trial.worker_id, hps=trial.hps)
trainer.config.mixup = True
trainer.train_loader = train_loader
trainer.valid_loader = test_loader

logging.info("start training")
trainer.train_process()
logging.info("training is complete, output folder: {}".format(trainer.get_local_worker_path()))
