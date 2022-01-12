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

"""Image classification Trainer."""
import torch
import torch.nn as nn
from modnas import backend
from modnas.registry.trainer import register
from ..base import TrainerBase


def accuracy(output, target, topk=(1, )):
    """Compute the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


@register
class ImageClsTrainer(TrainerBase):
    """Image classification Trainer class."""

    def __init__(self,
                 writer=None,
                 expman=None,
                 data_provider=None,
                 optimizer=None,
                 lr_scheduler=None,
                 criterion='CrossEntropyLoss',
                 w_grad_clip=0):
        super().__init__(writer)
        self.w_grad_clip = w_grad_clip
        self.expman = expman
        self.optimizer = None
        self.lr_scheduler = None
        self.data_provider = None
        self.criterion = None
        config = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'data_provider': data_provider,
            'criterion': criterion,
        }
        self.config = config

    def init(self, model, config=None):
        """Initialize trainer states."""
        self.config.update(config or {})
        if self.config['optimizer']:
            self.optimizer = backend.get_optimizer(model.parameters(), self.config['optimizer'], config)
        if self.config['lr_scheduler']:
            self.lr_scheduler = backend.get_lr_scheduler(self.optimizer, self.config['lr_scheduler'], config)
        if self.config['data_provider']:
            self.data_provider = backend.get_data_provider(self.config['data_provider'])
        if self.config['criterion']:
            self.criterion = backend.get_criterion(self.config['criterion'], getattr(model, 'device_ids', None))
        self.device = self.config.get('device', backend.get_device())

    def get_num_train_batch(self, epoch):
        """Return number of train batches."""
        return 0 if self.data_provider is None else self.data_provider.get_num_train_batch(epoch=epoch)

    def get_num_valid_batch(self, epoch):
        """Return number of valid batches."""
        return 0 if self.data_provider is None else self.data_provider.get_num_valid_batch(epoch=epoch)

    def get_next_train_batch(self):
        """Return next train batch."""
        return self.proc_batch(self.data_provider.get_next_train_batch())

    def get_next_valid_batch(self):
        """Return next valid batch."""
        return self.proc_batch(self.data_provider.get_next_valid_batch())

    def proc_batch(self, batch):
        """Return processed data batch."""
        return tuple(v.to(device=self.device, non_blocking=True) for v in batch)

    def state_dict(self):
        """Return current states."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }

    def load_state_dict(self, sd):
        """Resume states."""
        if self.optimizer is not None:
            self.optimizer.load_state_dict(sd['optimizer'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(sd['lr_scheduler'])

    def get_lr(self):
        """Return current learning rate."""
        if self.lr_scheduler:
            if hasattr(self.lr_scheduler, 'get_last_lr'):
                return self.lr_scheduler.get_last_lr()[0]
            return self.lr_scheduler.get_lr()[0]
        return self.optimizer.param_groups[0]['lr']

    def get_optimizer(self):
        """Return optimizer."""
        return self.optimizer

    def loss(self, output=None, data=None, model=None):
        """Return loss."""
        return None if self.criterion is None else self.criterion(None, None, output, *data)

    def train_epoch(self, estim, model, tot_steps, epoch, tot_epochs):
        """Train for one epoch."""
        for step in range(tot_steps):
            self.train_step(estim, model, epoch, tot_epochs, step, tot_steps)

    def train_step(self, estim, model, epoch, tot_epochs, step, tot_steps):
        """Train for one step."""
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        lr = self.get_lr()
        if step == 0:
            self.data_provider.reset_train_iter()
        model.train()
        batch = self.get_next_train_batch()
        trn_X, trn_y = batch
        optimizer.zero_grad()
        loss, logits = estim.loss_output(batch, model=model, mode='train')
        loss.backward()
        # gradient clipping
        if self.w_grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.w_grad_clip)
        optimizer.step()
        prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
        if step == tot_steps - 1:
            lr_scheduler.step()
        return {
            'acc_top1': prec1.item(),
            'acc_top5': prec5.item(),
            'loss': loss.item(),
            'LR': lr,
            'N': len(trn_y),
        }

    def valid_epoch(self, estim, model, tot_steps, epoch=0, tot_epochs=1):
        """Validate for one epoch."""
        if not tot_steps:
            return None
        for step in range(tot_steps):
            self.valid_step(estim, model, epoch, tot_epochs, step, tot_steps)

    def valid_step(self, estim, model, epoch, tot_epochs, step, tot_steps):
        """Validate for one step."""
        if step == 0:
            self.data_provider.reset_valid_iter()
        model.eval()
        with torch.no_grad():
            batch = self.get_next_valid_batch()
            val_X, val_y = batch
            loss, logits = estim.loss_output(batch, model=model, mode='eval')
        prec1, prec5 = accuracy(logits, val_y, topk=(1, 5))
        return {
            'acc_top1': prec1.item(),
            'acc_top5': prec5.item(),
            'loss': loss.item(),
            'N': len(val_y),
        }
