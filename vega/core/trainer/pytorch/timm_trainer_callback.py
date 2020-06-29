# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""TIMM method trainer."""
import os
import importlib
import torch
from vega.core.common import Config
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps
from vega.core.trainer.callbacks import Callback
from timm import create_model
from timm.optim.optim_factory import create_optimizer, add_weight_decay
from timm.scheduler import create_scheduler
from timm.data import Dataset, create_transform
from timm.utils import ModelEma
# additional dependencies
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.loader import fast_collate, PrefetchLoader
from timm.data.distributed_sampler import OrderedDistributedSampler

try:
    import horovod.torch as hvd
except Exception:
    # logging.warning("horovod not been installed, {}".format(str(e)))
    pass


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
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
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

    def before_train(self, logs=None):
        """Be called before the training process."""
        self._init_all_settings()

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        if self.horovod:
            self.train_loader.sampler.set_epoch(epoch)
        self.num_updates = epoch * len(self.train_loader)

    def make_batch(self, batch):
        """Prepare batch data for train_step."""
        input, target = batch
        if self.cfg.cuda and not self.cfg.prefetcher:
            input, target = input.cuda(), target.cuda()
        return input, target

    def train_step(self, batch):
        """Train one step of model."""
        self.model.train()
        input, target = batch
        self.optimizer.zero_grad()
        logits = self.model(input)
        loss = self.loss(logits, target)
        if self.use_amp:
            import apex.amp as amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
                self.optimizer.synchronize()
            with self.optimizer.skip_synchronize():
                self.optimizer.step()
        else:
            loss.backward()
            self.optimizer.step()
        if self.use_ema:
            self.model_ema.update(self.model)
        self.num_updates += 1
        self.lr_scheduler.step_update(num_updates=self.num_updates)
        return {'loss': loss.item(),
                'train_batch_output': logits}

    def before_valid(self, epoch, logs=None):
        """Be called before valid loop."""
        if self.use_ema:
            self.trainer.model = self.model_ema.ema

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.lr_scheduler.step(epoch=epoch + 1)

    def _init_all_settings(self):  # noqa: C901
        """Init all settings from config."""
        self.cfg = self.trainer.cfg
        if self.cfg.cuda:
            self.trainer._init_cuda_setting()
        if self.trainer.hps is not None:
            self.trainer._init_hps(self.trainer.hps)
        self.epochs = self.trainer.epochs
        self.horovod = self.trainer.horovod
        self.model = self._init_model()
        self.trainer.model = self.model
        self.use_syncbn = self.cfg.get('syncbn', False)
        self.trainer.use_syncbn = self.use_syncbn
        if self.use_syncbn:
            import apex
            self.model = apex.parallel.convert_syncbn_model(self.model)
        self.optimizer = self._init_optimizer()
        self.trainer.optimizer = self.optimizer
        self.use_ema = 'model_ema' in self.cfg
        if self.use_ema:
            self.model_ema = self._init_model_ema()
        self.lr_scheduler = self._init_lr_scheduler()
        self.trainer.lr_scheduler = self.lr_scheduler
        self.loss = self._init_loss()
        self.trainer.loss = self.loss
        if self.horovod:
            self.trainer._init_horovod_setting()
        self.use_amp = self.cfg.get('amp', False)
        if self.use_amp:
            import apex.amp as amp
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
        self.trainer.use_amp = self.use_amp
        self._init_dataloader()
        self.trainer.train_loader = self.train_loader
        self.trainer.valid_loader = self.valid_loader
        self.trainer.model = self.model
        self.trainer.valid_metrics = self.trainer._init_metrics(None)
        self.trainer._init_step_functions(self.make_batch, self.train_step, None)

        # re-assign callbacks params, due to the order issue
        tmp_params = self.trainer.callbacks.params
        tmp_params['is_chief'] = self.trainer.is_chief
        self.trainer.callbacks.set_params(tmp_params)

        self.trainer.has_built = True

    def _init_model_ema(self):
        """Init Model Ema."""
        args = self.cfg.model_ema
        model_ema = ModelEma(self.model,
                             decay=args.model_ema_decay,
                             device='cpu' if args.model_ema_force_cpu else '',
                             resume=self.cfg.get('load_checkpoint', None))
        return model_ema

    def _init_model(self):
        """Init network model from timm according to model type in config."""
        args = self.cfg.model_desc
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
        if self.cfg.cuda:
            model = model.cuda()
        return model

    def _init_optimizer(self):
        """Init optimizer from timm according to optim type in config."""
        optimizer = create_optimizer(self.cfg.optim, self.model)
        if self.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer,
                                                 named_parameters=self.model.named_parameters(),
                                                 compression=hvd.Compression.none)
        return optimizer

    def _init_lr_scheduler(self):
        """Init lr scheduler from timm according to type in config."""
        args = self.cfg.lr_scheduler.copy()
        args['epochs'] = self.cfg.epochs
        lr_scheduler, self.epochs = create_scheduler(Config(args), self.optimizer)
        start_epoch = args.get('start_epoch', 0)
        lr_scheduler.step(start_epoch)
        return lr_scheduler

    def _init_loss(self):
        """Init loss function from timm according to type in config."""
        loss_config = self.cfg.loss.copy()
        loss_name = loss_config.pop('type')
        loss_class = getattr(importlib.import_module('timm.loss'), loss_name)
        loss_fn = loss_class(**loss_config)
        if self.cfg.cuda:
            loss_fn = loss_fn.cuda()
        return loss_fn

    def _reset_sync_opt(self):
        """Rest sysnc opt."""
        params = add_weight_decay(self.model, self.cfg.optim.weight_decay)
        self.optimizer.param_groups = []
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.optimizer.add_param_group(param_group)

    def _init_dataloader(self):
        """Init dataloader from timm."""
        if self.horovod and hvd.local_rank() == 0 and 'remote_data_dir' in self.cfg.dataset:
            FileOps.copy_folder(self.cfg.dataset.remote_data_dir, self.cfg.dataset.data_dir)
        if self.horovod:
            hvd.join()
        args = self.cfg.dataset
        train_dir = os.path.join(self.cfg.dataset.data_dir, 'train')
        dataset_train = Dataset(train_dir)
        world_size, rank = None, None
        if self.horovod:
            world_size, rank = hvd.size(), hvd.rank()
        self.train_loader = create_loader(
            dataset_train,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=self.cfg.prefetcher,
            rand_erase_prob=args.reprob,
            rand_erase_mode=args.remode,
            rand_erase_count=args.recount,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='random',
            mean=tuple(args.mean),
            std=tuple(args.std),
            num_workers=args.workers,
            distributed=self.horovod,
            world_size=world_size,
            rank=rank
        )
        valid_dir = os.path.join(self.cfg.dataset.data_dir, 'val')
        dataset_eval = Dataset(valid_dir)
        self.valid_loader = create_loader(
            dataset_eval,
            input_size=tuple(args.input_size),
            batch_size=4 * args.batch_size,
            is_training=False,
            use_prefetcher=self.cfg.prefetcher,
            interpolation=args.interpolation,
            mean=tuple(args.mean),
            std=tuple(args.std),
            num_workers=args.workers,
            distributed=self.horovod,
            world_size=world_size,
            rank=rank
        )
