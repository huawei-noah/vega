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

"""Train simclr model."""
import logging
import torch
from torch.optim import Adam
from .model import SimclrModel
from .loss import NT_Xent


def simclr_train(init_model, train_loader, epochs=1):
    """Train the simclr model and save the pretrain weight to apply to downstream task.

    param init_model: the model for downstream task, which contains backbone and head
    type init_model: torch.nn.Module
    param train_loader: the train dataset
    type train_loader: torch.DataLoader
    param epochs: the epochs to train the simclr model
    type epochs: int
    """
    model = SimclrModel(init_model).cuda()
    optimizer = Adam(model.parameters(), lr=0.001)
    for step, ((pos1, pos2), _) in enumerate(train_loader):
        batch_size = pos1.shape[0]
        break
    loss_fn = NT_Xent(batch_size=batch_size)

    for epoch in range(epochs):
        for step, ((pos1, pos2), _) in enumerate(train_loader):
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()
            feature1, out1 = model(pos1)
            feature2, out2 = model(pos2)

            loss = loss_fn(out1, out2)
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                logging.info("Simclr train, epoch [{}/{}], step [{}/{}], current loss: [{}]"
                             .format(epoch, epochs, step, len(train_loader), loss.item()))
    save_name = 'simclr_model_epoch_{}.pth'.format("epoch")
    torch.save(model.state_dict(), save_name)
    return update_init_model(init_model, save_name)


def update_init_model(init_model, weight_file):
    """Update the downstream task model according to the pretrain simclr model.

    param init model: the model for downstream task, which contains backbone and head
    type init_model: torch.nn.Module
    param weight_file: the pretrained weight by simclr
    type weight_file: str
    """
    pretrain_dict = torch.load(weight_file)
    logging.info("Loading the pretrain model of simclr.")
    model_dict = init_model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}

    model_dict.update(pretrain_dict)
    init_model.load_state_dict(model_dict)
    return init_model
