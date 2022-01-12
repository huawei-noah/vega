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

"""TIMM method trainer."""

import logging
import os
import importlib
import torch
from timm import create_model
from timm.optim.optim_factory import create_optimizer, add_weight_decay
from timm.scheduler import create_scheduler
from timm.data import Dataset, create_transform
from timm.utils import ModelEma
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.loader import fast_collate, PrefetchLoader
from timm.data.distributed_sampler import OrderedDistributedSampler
try:
    import apex
    from apex import amp
except Exception:
    logging.debug('apex is no installed.')
import horovod.torch as hvd
import vega
from vega.common import Config
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.trainer.callbacks import Callback


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        rand_erase_prob=0.,
        rand_erase_mode='const',
        rand_erase_count=1,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        fp16=False,
        tf_preprocessing=False,
        world_size=None,
        rank=None
):
    """Create data loader for timm."""
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
    )

    sampler = None
    if distributed:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank)
        else:
            sampler = OrderedDistributedSampler(dataset, num_replicas=world_size, rank=rank)

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=is_training,
    )
    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            re_prob=rand_erase_prob if is_training else 0.,
            re_mode=rand_erase_mode,
            re_count=rand_erase_count,
            mean=mean,
            std=std,
            fp16=fp16)

    return loader


@ClassFactory.register(ClassType.CALLBACK)
class TimmTrainerCallback(Callback):
    """A special callback for TimmTrainer."""

    disable_callbacks = ["LearningRateScheduler", "ModelStatistics", "ModelBuilder"]

    def before_train(self, logs=None):
        """Be called before the training process."""
        self._init_all_settings()

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoch."""
        if self.distributed:
            self.trainer.train_loader.sampler.set_epoch(epoch)
        self.num_updates = epoch * len(self.trainer.train_loader)
        self.epoch = epoch
        self.trainer.model.train()

    def make_batch(self, batch):
        """Prepare batch data for train_step."""
        input, target = batch
        if not self.config.prefetcher:
            if vega.is_gpu_device():
                input, target = input.cuda(), target.cuda()
            elif vega.is_npu_device():
                input, target = input.to(vega.get_devices()), target.to(vega.get_devices())
        return input, target

    def train_step(self, batch):
        """Train one step of model."""
        input, target = batch
        self.trainer.optimizer.zero_grad()
        logits = self.trainer.model(input)
        loss = self.trainer.loss(logits, target)
        if self.use_amp:
            with amp.scale_loss(loss, self.trainer.optimizer) as scaled_loss:
                scaled_loss.backward()
                self.trainer.optimizer.synchronize()
            with self.trainer.optimizer.skip_synchronize():
                self.trainer.optimizer.step()
        else:
            loss.backward()
            self.trainer.optimizer.step()
        if self.use_ema:
            self.model_ema.update(self.trainer.model)
        self.num_updates += 1
        self.trainer.lr_scheduler.step_update(num_updates=self.num_updates)
        return {'loss': loss.item(),
                'train_batch_output': logits,
                'lr': self.trainer.lr_scheduler.get_epoch_values(self.epoch)}

    def before_valid(self, epoch, logs=None):
        """Be called before valid loop."""
        if self.use_ema:
            self.trainer.model = self.model_ema.ema
        self.trainer.model.eval()

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        if self.use_ema:
            self.trainer.model = self.model
        self.trainer.lr_scheduler.step(epoch=epoch + 1)
        if self.trainer.is_chief:
            self.trainer._backup()

    def _init_all_settings(self):
        """Init all settings from config."""
        self.config = self.trainer.config
        if self.trainer.hps and self.trainer.hps.get('trainer'):
            self.config.from_dict(self.trainer.hps.get('trainer'))
        if not vega.is_cpu_device():
            self.trainer._init_setting()
        self.distributed = self.trainer.horovod
        self.trainer.model = self._init_model()
        self.model = self.trainer.model
        self.use_syncbn = self.config.syncbn
        self.trainer.use_syncbn = self.use_syncbn
        if self.use_syncbn:
            self.trainer.model = apex.parallel.convert_syncbn_model(self.trainer.model)
        self.trainer.optimizer = self._init_optimizer()
        self.use_ema = hasattr(self.config, 'model_ema')
        if self.use_ema:
            self.model_ema = self._init_model_ema()
        self.trainer.lr_scheduler = self._init_lr_scheduler()
        self.trainer.loss = self._init_loss()
        self.use_amp = self.config.use_amp
        if self.use_amp:
            self.trainer.model, self.trainer.optimizer = amp.initialize(self.trainer.model,
                                                                        self.trainer.optimizer,
                                                                        opt_level='O1')
        self._init_dataloader()
        self.trainer.valid_metrics = self.trainer._init_metrics(None)
        self.trainer.callbacks._set_params(self.trainer)

    def _init_model_ema(self):
        """Init Model Ema."""
        args = self.config.model_ema
        model_ema = ModelEma(self.trainer.model,
                             decay=args.model_ema_decay,
                             device='cpu' if args.model_ema_force_cpu else '',
                             resume=None)
        return model_ema

    def _init_model(self):
        """Init network model from timm according to model type in config."""
        args = self.config.model_desc
        model = create_model(args.model_name,
                             pretrained=args.pretrained,
                             num_classes=args.num_classes,
                             drop_rate=args.drop,
                             drop_path_rate=args.drop_path,
                             global_pool=args.gp,
                             bn_tf=args.bn_tf,
                             bn_momentum=args.bn_momentum,
                             bn_eps=args.bn_eps,
                             checkpoint_path=args.initial_checkpoint)
        if vega.is_gpu_device():
            model = model.cuda()
        elif vega.is_npu_device():
            model = model.to(vega.get_devices())
        return model

    def _init_optimizer(self):
        """Init optimizer from timm according to optim type in config."""
        optimizer = create_optimizer(self.config.optimizer().to_dict()["params"], self.trainer.model)
        if self.distributed:
            optimizer = hvd.DistributedOptimizer(optimizer,
                                                 named_parameters=self.trainer.model.named_parameters(),
                                                 compression=hvd.Compression.none)
        return optimizer

    def _init_lr_scheduler(self):
        """Init lr scheduler from timm according to type in config."""
        args = self.config.lr_scheduler().to_dict()["params"]
        args['epochs'] = self.config.epochs
        lr_scheduler, self.config.epochs = create_scheduler(Config(args), self.trainer.optimizer)
        start_epoch = args.get('start_epoch', 0)
        lr_scheduler.step(start_epoch)
        return lr_scheduler

    def _init_loss(self):
        """Init loss function from timm according to type in config."""
        loss_name = self.config.loss.type
        loss_config = self.config.loss().to_dict()["params"]
        loss_class = getattr(importlib.import_module('timm.loss'), loss_name)
        loss_fn = loss_class(**loss_config)
        if vega.is_gpu_device():
            loss_fn = loss_fn.cuda()
        elif vega.is_npu_device():
            loss_fn = loss_fn.to(vega.get_devices())
        return loss_fn

    def _reset_sync_opt(self):
        """Rest sysnc opt."""
        params = add_weight_decay(self.model, self.config.optimizer.weight_decay)
        self.optimizer.param_groups = []
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.optimizer.add_param_group(param_group)

    def _init_dataloader(self):
        """Init dataloader from timm."""
        if self.distributed and hvd.local_rank() == 0 and 'remote_data_dir' in self.config.dataset:
            FileOps.copy_folder(self.config.dataset.remote_data_dir, self.config.dataset.data_dir)
        if self.distributed:
            hvd.join()
        args = self.config.dataset
        train_dir = os.path.join(self.config.dataset.data_dir, 'train')
        dataset_train = Dataset(train_dir)
        world_size, rank = None, None
        if self.distributed:
            world_size, rank = hvd.size(), hvd.rank()
        self.trainer.train_loader = create_loader(
            dataset_train,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=self.config.prefetcher,
            rand_erase_prob=args.reprob,
            rand_erase_mode=args.remode,
            rand_erase_count=args.recount,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='random',
            mean=tuple(args.mean),
            std=tuple(args.std),
            num_workers=args.workers,
            distributed=self.distributed,
            world_size=world_size,
            rank=rank
        )
        valid_dir = os.path.join(self.config.dataset.data_dir, 'val')
        dataset_eval = Dataset(valid_dir)
        self.trainer.valid_loader = create_loader(
            dataset_eval,
            input_size=tuple(args.input_size),
            batch_size=4 * args.batch_size,
            is_training=False,
            use_prefetcher=self.config.prefetcher,
            interpolation=args.interpolation,
            mean=tuple(args.mean),
            std=tuple(args.std),
            num_workers=args.workers,
            distributed=self.distributed,
            world_size=world_size,
            rank=rank
        )
        self.trainer.batch_num_train = len(self.trainer.train_loader)
        self.trainer.batch_num_valid = len(self.trainer.valid_loader)
