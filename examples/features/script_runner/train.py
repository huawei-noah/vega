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
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn import functional as F
from vega.trainer.trial_agent import TrialAgent

logging.basicConfig(level=logging.INFO)
logging.info("load trial")
trial = TrialAgent()

logging.info("create model")
resnet18 = models.resnet18(pretrained=False).cuda()

logging.info(f"hps: {trial.hps}")
hps = trial.hps
epochs = hps["trainer"].get("epochs", trial.epochs)
otpimizer_name = hps["trainer"]["optimizer"]["type"]
otpimizer_params = hps["trainer"]["optimizer"]["params"]
batch_size = hps["dataset"]["batch_size"]

logging.info("load dataset")
train_dataset = torchvision.datasets.CIFAR10(
    root="/cache/datasets/cifar10", train=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(
    root="/cache/datasets/cifar10", train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=256, shuffle=False)

if otpimizer_name == "SGD":
    optimizer = torch.optim.SGD(resnet18.parameters(), **otpimizer_params)
else:
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=otpimizer_params["lr"])
loss_fn = torch.nn.CrossEntropyLoss().cuda()

logging.info("training ...")
for epoch in range(epochs):
    logging.info(f"epoch: {epoch}")
    optimizer.zero_grad()
    for data, target in train_loader:
        out = resnet18(data.cuda())
        out = F.log_softmax(out, dim=1)
        loss = loss_fn(out, target.cuda())
        loss.backward()
        optimizer.step()

logging.info("evaluate ...")
resnet18.eval()
num_correct = 0
with torch.no_grad():
    for data, target in test_loader:
        out = resnet18(data.cuda())
        out = out.argmax(dim=1)
        num_correct += torch.eq(out, target.cuda()).sum().float().item()
accuracy = float(num_correct / len(train_loader.dataset))

logging.info(f"accuracy: {accuracy}")
result = trial.update(performance={"accuracy": accuracy})
logging.info(f"have sent to server, record: {result}")
