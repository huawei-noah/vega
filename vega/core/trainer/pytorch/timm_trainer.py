# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""TIMM method trainer."""
import logging
import os
import importlib
import torch
from vega.core.common import Config
from vega.core.trainer.pytorch.trainer import Trainer
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps
from vega.core.metrics.pytorch.metrics import Metrics
try:
    from timm import create_model
    from timm.optim import create_optimizer, add_weight_decay
    from timm.scheduler import create_scheduler
    from timm.data import Dataset, create_loader
    from timm.utils import ModelEma
except Exception:
    # logging.warning("timm not been installed, {}".format(str(e)))
    pass
try:
    import apex
    from apex import amp
except Exception:
    # logging.warning("apex not been installed, {}".format(str(e)))
    pass
try:
    import horovod.torch as hvd
except Exception:
    # logging.warning("horovod not been installed, {}".format(str(e)))
    pass


@ClassFactory.register(ClassType.TRAINER)
class TimmTrainer(Trainer):
    """Class of Timm Fully Trainer.

    :param model: network model
    :type model: torch.nn.Module
    :param id: trainer id
    :type id: int
    """

    def __init__(self, model, id):
        """Init TimmTrainer."""
        super(TimmTrainer, self).__init__(model, id)
        logging.info('Init Timm FullyTrainer')

    def _init_all_settings(self):  # noqa: C901
        """Init all settings from config."""
        if self.cfg.cuda:
            self._init_cuda_setting()
        if self.hps is not None:
            self._init_hps(self.hps)
        if self.model is None:
            if self.cfg.use_timm_model:
                self.model = self._init_model()
            else:
                self.model = super()._init_model()
        if self.cfg.load_checkpoint:
            self.model = self._load_checkpoint()
        self.use_syncbn = self.cfg.get('syncbn', False)
        if self.use_syncbn:
            self.model = apex.parallel.convert_syncbn_model(self.model)
        self.epochs = self.cfg.epochs
        if self.cfg.use_timm_optim:
            self.optimizer = self._init_optimizer()
        else:
            self.optimizer = super()._init_optimizer()
        self.use_ema = 'model_ema' in self.cfg
        if self.use_ema:
            self.model_ema = self._init_model_ema()
        if self.cfg.use_timm_lr_sched:
            self.lr_scheduler = self._init_lr_scheduler()
        else:
            self.lr_scheduler = super()._init_lr_scheduler()
        if self.cfg.use_timm_loss:
            self.loss = self._init_loss()
        else:
            self.loss = super()._init_loss()
        if self.horovod:
            self._init_horovod_setting()
        self.use_amp = self.cfg.get('amp', False)
        if self.use_amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
        if self.cfg.use_timm_dataset:
            self._init_dataloader()
        else:
            super()._init_dataloader()

    def _init_model_ema(self):
        args = self.cfg.model_ema
        model_ema = ModelEma(self.model,
                             decay=args.model_ema_decay,
                             device='cpu' if args.model_ema_force_cpu else '',
                             resume=self.cfg.load_checkpoint)
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
        params = add_weight_decay(self.model, self.cfg.optim.weight_decay)
        self.optimizer.param_groups = []
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.optimizer.add_param_group(param_group)

    def _init_dataloader(self):
        """Init dataloader from timm."""
        if self.horovod and hvd.local_rank() == 0:
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

    def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        logging.info("start train process")
        self._init_all_settings()
        best_pfm = [0., 0.]
        for i in range(self.epochs):
            if self.horovod:
                self.train_loader.sampler.set_epoch(i)
            self.train(self.train_loader, i)
            if self.cfg.with_valid:
                pfm = self.valid(self.valid_loader, i)
            self.lr_scheduler.step(epoch=i + 1)
            if self._first_rank:
                if self.cfg.with_valid:
                    if pfm[0] > best_pfm[0]:
                        best_pfm = pfm
                        self._save_performance(pfm)
                        self._save_checkpoint(i)
                        self._backup()
                    logging.info("Epoch [%d/%d], current top1: [%f] top5: [%f], best top1: [%f] top5: [%f]",
                                 i, self.epochs, pfm[0], pfm[1], best_pfm[0], best_pfm[1])
        logging.info("finished training")

    def train(self, loader, epoch):
        """Train one step of model.

        :param loader: train data loader
        :type loader: DataLoader
        :param epoch: current epoch.
        :type epoch: int
        """
        metrics = Metrics(self.cfg.metric)
        self.model.train()
        loss_sum = 0.
        data_num = 0
        num_updates = epoch * len(loader)
        for step, (input, target) in enumerate(loader):
            if self.cfg.cuda and not self.cfg.prefetcher:
                input, target = input.cuda(), target.cuda()
            self.optimizer.zero_grad()
            logits = self.model(input)
            loss = self.loss(logits, target)
            if self.use_amp:
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
            metrics(logits, target)
            n = input.size(0)
            data_num += n
            loss_sum += loss.item() * n
            loss_avg = loss_sum / data_num
            lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            prec = metrics.results
            if self._first_rank and step % self.cfg.report_freq == 0:
                logging.info("step [%d/%d], top1 [%f], top5 [%f], loss avg [%f], lr [%f]",
                             step, len(loader), prec[0], prec[1], loss_avg, lr)
            num_updates += 1
            if self.cfg.use_timm_lr_sched:
                self.lr_scheduler.step_update(num_updates=num_updates)

    def valid(self, loader, epoch):
        """Validate one step of model.

        :param loader: valid data loader
        :type loader: DataLoader
        :param epoch: current epoch.
        :type epoch: int
        :return: performance.
        :rtype: type
        """
        metrics = Metrics(self.cfg.metric)
        model = self.model_ema.ema if self.use_ema else self.model
        model.eval()
        with torch.no_grad():
            for _, (input, target) in enumerate(loader):
                if self.cfg.cuda and not self.cfg.prefetcher:
                    input, target = input.cuda(), target.cuda()
                logits = model(input)
                metrics(logits, target)
        prec = metrics.results
        if self.horovod:
            prec = [self._metric_average(acc, self.cfg.metric.type) for acc in prec]
        return prec
