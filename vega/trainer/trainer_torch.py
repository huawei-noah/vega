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

"""Torch Trainer."""

import torch
import numpy as np
import vega
from vega.common.general import General
from vega.trainer.trainer_base import TrainerBase
from vega.modules.loss import Loss
from vega.trainer.modules.lr_schedulers import LrScheduler
from vega.trainer.modules.optimizer import Optimizer
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRAINER)
class TrainerTorch(TrainerBase):
    """Trainer torch class."""

    def build(self):
        """Build the trainer by assembling the necessary components."""
        super().build()
        if self.optimizer is None:
            self.optimizer = Optimizer()(model=self.model, distributed=self.horovod)
        if hasattr(self.model, 'add_loss'):
            loss_cls = Loss()()
            self.model.add_loss(loss_cls)
            self.loss = self.model.overall_loss()
        else:
            self.loss = Loss()()
        if self.config.adaptive_muti_loss and hasattr(self.loss, "adaptive_muti_loss"):
            self.loss.adaptive_muti_loss(save_path=self.get_local_worker_path(self.step_name, self.worker_id),
                                         weight=self.config.loss_weight)
        if self.lr_scheduler is None:
            self.lr_scheduler = LrScheduler()(self.optimizer,
                                              steps=len(self.train_loader) if self.train_loader is not None else None,
                                              epochs=self.config.epochs)
        if self.actions_list is not None:
            self.total_optimizer = self.optimizer
            self.total_loss = self.loss
            self.total_lr_scheduler = self.lr_scheduler
        # Some trainer has different train batch size from valid batch
        self.train_metrics = self._init_metrics()
        self.valid_metrics = self._init_metrics()
        if self.use_amp:
            from apex import amp
            if not vega.is_npu_device():
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=self.config.opt_level,
                    loss_scale=self.config.apex_loss_scale)
            else:
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=self.config.opt_level,
                    loss_scale=self.config.apex_loss_scale,
                    combine_grad=self.config.apex_combine_grad)
        # mode ddp should after amp.initialize
        if self.hccl:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device_id],
                broadcast_buffers=General.cluster.enable_broadcast_buffers
            )

    def set_training_settings(self):
        """Set trainer training setting."""
        self.make_batch = self._default_make_batch
        if isinstance(self.config.optimizer, list):
            self.train_step = self._multi_train_step
        else:
            self.train_step = self._default_train_step
        self.valid_step = self._default_valid_step

    def init_env(self):
        """Init trainer environment."""
        super().init_env()
        torch.manual_seed(self.config.seed)
        self._init_setting()

    def _init_setting(self):
        """Init CUDA setting."""
        if vega.is_gpu_device():
            import torch.cuda as torch_cuda
            self.config.device = vega.is_gpu_device() if vega.is_gpu_device() is not True else 0
            torch_cuda.manual_seed(self.config.seed)
        elif vega.is_npu_device():
            import torch.npu as torch_npu
            torch_npu.set_device(vega.get_devices())
            torch_npu.manual_seed(self.config.seed)
        elif vega.is_cpu_device():
            self.config.device = -1
            return
        else:
            raise ValueError('Set a correct device: cuda or npu.')

    def _train_epoch(self):
        self.model.train()
        for batch_index, batch in enumerate(self.train_loader):
            if self.config.max_train_steps and batch_index >= self.config.max_train_steps:
                return
            batch = self.make_batch(batch)
            batch_logs = {'train_batch': batch}
            self.callbacks.before_train_step(batch_index, batch_logs)
            train_batch_output = self.train_step(batch)
            batch_logs.update(train_batch_output)
            if self.config.is_detection_trainer:
                batch_logs.update({'is_detection_trainer': True})
            self.callbacks.after_train_step(batch_index, batch_logs)

    def _valid_epoch(self):
        self.callbacks.before_valid()
        valid_logs = None
        self.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(self.valid_loader):
                batch = self.make_batch(batch)
                batch_logs = {'valid_batch': batch}
                self.callbacks.before_valid_step(batch_index, batch_logs)
                valid_batch_output = self.valid_step(batch)
                self.callbacks.after_valid_step(batch_index, valid_batch_output)

        self.callbacks.after_valid(valid_logs)

    def _default_make_batch(self, batch):
        """Unpack batch to get input and target."""
        if not vega.is_cpu_device():
            batch = self._set_device(batch)
        return batch

    def _set_device(self, data):
        if torch.is_tensor(data):
            if vega.is_gpu_device():
                return data.cuda()
            else:
                return data.to(vega.get_devices())
        if isinstance(data, dict):
            return {k: self._set_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._set_device(v) for v in data]
        elif isinstance(data, tuple):
            return tuple([self._set_device(v) for v in data])
        return data

    def _default_train_step(self, batch):
        self.optimizer.zero_grad()
        input, target = None, None
        if isinstance(batch, dict):
            output = self.model(**batch)
        elif isinstance(batch, list) and isinstance(batch[0], dict):
            output = self.model(batch)
        elif isinstance(batch, list) and isinstance(batch[0], list):
            output = self.model(*batch)
        else:
            # classification
            input, target = batch
            if self.config.mixup:
                mixup_ratio = np.random.beta(0.1, 0.1)
                mixed_x, y_a, y_b = self._mixup_batch(input, target, mixup_ratio)
                output = self.model(mixed_x)
            else:
                output = self.model(input) if not isinstance(input, dict) else self.model(**input)
        # loss
        if self.config.mixup:
            loss = self._mixup_loss(self.loss, output, y_a, y_b, mixup_ratio)
        else:
            loss = self.loss(output, target)
        if self.use_amp:
            self._set_amp_loss(loss)
        else:
            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        return {'loss': loss.item(),
                'train_batch_output': output,
                'lr': self.lr_scheduler.get_lr()}

    def _set_amp_loss(self, loss):
        from apex import amp
        if vega.is_npu_device():
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()
        else:
            if self.horovod:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    self.optimizer.synchronize()
                with self.optimizer.skip_synchronize():
                    self.optimizer.step()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

    def _multi_train_step(self, batch):
        train_batch_output = None
        for opt_name, sub_opt in self.optimizer.get_opts():
            self.optimizer = sub_opt.get('opt')
            self.loss = sub_opt.get('loss')
            self.lr_scheduler = sub_opt.get('lr')
            train_batch_output = self._default_train_step(batch)
        return train_batch_output

    def _default_valid_step(self, batch):
        if isinstance(batch, dict):
            output = self.model(**batch)
        elif isinstance(batch, list) and isinstance(batch[0], dict):
            output = self.model(batch)
        else:
            input, target = batch
            output = self.model(input) if not isinstance(input, dict) else self.model(**input)
        return {'valid_batch_output': output}

    def _mixup_batch(self, x, y, ratio):
        indices = torch.randperm(x.shape[0])
        mixed_x = ratio * x + (1 - ratio) * x[indices]
        y_a, y_b = y, y[indices]
        return mixed_x, y_a, y_b

    def _mixup_loss(self, loss, pred, y_a, y_b, ratio):
        return ratio * loss(pred, y_a) + (1 - ratio) * loss(pred, y_b)
