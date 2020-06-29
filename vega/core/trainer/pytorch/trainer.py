# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Trainer."""
import importlib
import json
import logging
import os
import pickle
import pandas as pd
from functools import partial
from inspect import isfunction
from copy import deepcopy
import torch
from collections import OrderedDict
from vega.core.common import Config, FileOps, init_log
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.utils import update_dict
from vega.core.trainer.distributed_worker import DistributedWorker
from vega.core.common.task_ops import TaskOps
from vega.core.trainer.utils import WorkerTypes
from vega.datasets.pytorch.common.dataset import Dataset
from vega.search_space.networks import NetworkDesc, NetworkFactory, NetTypes

from vega.core.metrics.pytorch.metrics import Metrics
from vega.core.trainer.callbacks import CallbackList, ProgressLogger,\
    MetricsEvaluator, PerformanceSaver, LearningRateScheduler,\
    ModelStatistics, ModelCheckpoint

try:
    import horovod.torch as hvd
except Exception:
    # logging.warning("horovod not been installed, {}".format(str(e)))
    pass
try:
    import apex
    from apex import amp
except Exception:
    # logging.warning("apex not been installed, {}".format(str(e)))
    pass


logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.TRAINER)
class Trainer(DistributedWorker):
    """Trainer class.

    :param id: id of the model, defaults to None
    :type id: int, optional
    """

    __worker_id__ = 0

    def __init__(self, model=None, id=None, hps=None, load_ckpt_flag=False, **kwargs):
        """Init Trainer."""
        super(Trainer, self).__init__(self.cfg)

        self.worker_type = WorkerTypes.TRAINER
        Trainer.__worker_id__ += 1
        if id is not None:
            self._worker_id = id
        else:
            self._worker_id = Trainer.__worker_id__

        # Data Memeber list of Trainer
        self.is_chief = True
        self.use_cuda = True
        self.epochs = self.cfg.epochs
        self.do_validation = True
        self.auto_save_ckpt = True
        self.auto_save_perf = True
        self.skip_train = False
        self.valid_freq = self.cfg.get('valid_freq', 1)
        self.hps = hps
        self.model = model
        self.optimizer = None
        self.lr_scheduler = None
        self.loss = None
        self.use_syncbn = self.cfg.get('syncbn', False)
        self.use_amp = self.cfg.get('amp', False)
        self.train_metrics = None
        self.valid_metrics = None
        self.train_loader = None
        self.valid_loader = None
        self.train_step = None
        self.valid_step = None
        self.make_batch = None
        self.callbacks = None
        self.model_desc = {}
        self.visual_data = {}
        self.load_ckpt_flag = load_ckpt_flag
        self.checkpoint_file_name = 'weights.pth'
        self.model_pickle_file_name = 'model.pkl'
        self.performance_file_name = 'performance.txt'
        self.horovod = self.cfg.get('horovod', False)
        # Used by TimmTrainerCallbacks since it builds its trainer in
        # the before_train callback
        self.lazy_built = self.cfg.get('lazy_built', False)
        # Indicate whether the necessary components of a trainer
        # has been built for running
        self.has_built = False
        self._callbacks_mapping()

    def _callbacks_mapping(self):
        """Convert config to callback setting."""
        mapping = {}
        callback_config = self.cfg.get('callbacks')
        if callback_config:
            mapping[callback_config] = {'type': callback_config}
        if self.cfg.get('model_statistics'):
            mapping['model_statistics'] = {'type': 'ModelStatistics'}
        # default callbacks
        if 'call_point' in self.cfg.lr_scheduler:
            lr_scheduler_point = self.cfg.lr_scheduler.pop('call_point')
            mapping['lr_scheduler_point'] = {'type': 'LearningRateScheduler', 'call_point': lr_scheduler_point}
        else:
            mapping['lr_scheduler_point'] = {'type': 'LearningRateScheduler'}
        mapping['progress_logger'] = {'type': 'ProgressLogger', 'train_verbose': self.cfg.get('report_verbose', 2),
                                      'train_report_steps': self.cfg.report_freq}
        self.cfg.callbacks = Config(mapping)

    def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        init_log(log_file="worker_{}.txt".format(self.worker_id))
        logging.debug("Use the unified Trainer")
        if not self.lazy_built:
            self.build(model=self.model, hps=self.hps,
                       load_ckpt_flag=self.load_ckpt_flag)
        self.train()

    def build(self, model=None, optimizer=None, loss=None,
              lr_scheduler=None, metrics=None, hps=None,
              callbacks=None, train_loader=None, valid_loader=None,
              make_batch=None, train_step=None, valid_step=None,
              load_ckpt_flag=False,
              checkpoint_file_name="weights.pth",
              model_pickle_file_name="model.pkl",
              performance_file_name="performance.txt"):
        """Build the trainer by assembling the necessary components."""
        # Intitialize hyperparameters by parameters or configurations
        self.checkpoint_file_name = checkpoint_file_name
        self.model_pickle_file_name = model_pickle_file_name
        self.performance_file_name = performance_file_name

        self._init_cuda_setting()
        self._init_hps(hps)

        self.do_validation = self.cfg.with_valid
        self.model = self._init_model(model)
        self.load_ckpt_flag = load_ckpt_flag
        if self.load_ckpt_flag:
            self.load_checkpoint()
        else:
            self._load_pretrained_model()
        if self.model is not None and self.use_cuda:
            self.model = self.model.cuda()

        self.use_syncbn = self.cfg.get('syncbn', False)
        if self.use_syncbn:
            self.model = apex.parallel.convert_syncbn_model(self.model)
        self.optimizer = self._init_optimizer(optimizer)
        self.loss = self._init_loss(loss)
        self.lr_scheduler = self._init_lr_scheduler(lr_scheduler)
        # Some trainer has different train batch size from valid batch
        self.train_metrics = self._init_metrics(metrics)
        self.valid_metrics = self._init_metrics(metrics)
        self.train_loader = self._init_dataloader(
            mode='train', loader=train_loader)
        self.valid_loader = self._init_dataloader(
            mode='test', loader=valid_loader)
        self._init_horovod_setting()
        self.use_amp = self.cfg.get('amp', False)
        if self.use_amp:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level='O1')
        if self.callbacks is None:
            self.callbacks = callbacks
        self._init_step_functions(make_batch, train_step, valid_step)
        # self.output_model_desc()
        cur_working_dir = FileOps.join_path(self.local_output_path, self.step_name)
        FileOps.make_dir(cur_working_dir)
        # Make sure Trainer has been built for training
        self.has_built = True

    def train(self):
        """Do the training with data, callbacks and step functions etc."""
        self._init_callbacks(self.callbacks)
        self._train_loop()

    def _train_loop(self):
        """Do the training with data, callbacks and step functions etc."""
        # Allow user to build trainer in before_train() callback, but they
        # should set lazy_built in configuration file to True
        self.callbacks.before_train()
        if self.skip_train:
            return
        for epoch in range(self.epochs):
            epoch_logs = {'train_num_batches': len(self.train_loader)}
            if self.do_validation:
                epoch_logs.update({'valid_num_batches': len(self.valid_loader)})
            self.callbacks.before_epoch(epoch, epoch_logs)
            for batch_index, batch in enumerate(self.train_loader):
                batch = self.make_batch(batch)
                batch_logs = {'train_batch': batch}
                self.callbacks.before_train_step(batch_index, batch_logs)
                train_batch_output = self.train_step(batch)
                batch_logs.update(train_batch_output)
                if self.cfg.is_detection_trainer:
                    batch_logs.update({'is_detection_trainer': True})
                self.callbacks.after_train_step(batch_index, batch_logs)
            if self.do_validation and self._should_run_validation(epoch):
                self._valid_loop()
            self.callbacks.after_epoch(epoch)
        self.callbacks.after_train()

    def _valid_loop(self):
        self.callbacks.before_valid()
        self.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(self.valid_loader):
                batch = self.make_batch(batch)
                batch_logs = {'valid_batch': batch}
                self.callbacks.before_valid_step(batch_index, batch_logs)
                valid_batch_output = self.valid_step(batch)
                self.callbacks.after_valid_step(batch_index, valid_batch_output)
            # TODO: will be removed to callback
            pfm = self.valid_metrics.results
            if self.horovod:
                pfm = pfm[0][0] if isinstance(pfm[0], list) else pfm[0]
                pfm = self._metric_average(pfm, list(self.valid_metrics.results_dict.keys())[0])
            if self.is_chief and self.auto_save_perf:
                self._save_performance(pfm)
        self.callbacks.after_valid()

    def _default_train_step(self, batch):
        self.model.train()
        input, target = batch
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.loss(output, target)
        if self.use_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
                self.optimizer.synchronize()
            with self.optimizer.skip_synchronize():
                self.optimizer.step()
        else:
            loss.backward()
            if 'grad_clip' in self.cfg:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
        return {'loss': loss.item(),
                'train_batch_output': output}

    def _default_valid_step(self, batch):
        input, target = batch
        if self.cfg.is_detection_trainer:
            output = self.model(input, forward_train=False)
        else:
            output = self.model(input)
        return {'valid_batch_output': output}

    def _should_run_validation(self, epoch):
        # Zero valid_freq means doesn't run _valid_loop of the trainer
        # and user may provide _valid_loop in other callbacks
        if self.valid_freq == 0:
            return False
        else:
            return (epoch + 1) % self.valid_freq == 0

    def _default_make_batch(self, batch):
        """Unpack batch to get input and target."""
        input, target = batch
        if self.use_cuda and not self.cfg.is_detection_trainer:
            input, target = input.cuda(), target.cuda()
        return (input, target)

    def _predefined_callbacks(self):
        predefined_callbacks = (MetricsEvaluator, LearningRateScheduler,
                                ProgressLogger, PerformanceSaver,
                                ModelStatistics, ModelCheckpoint)
        return predefined_callbacks

    def _init_callbacks(self, callbacks):
        # Initialize callbacks by configuration or parameters
        if callbacks is None:
            _callbacks = []
            callbacks_config = self.cfg.callbacks.copy()
            for callback_config in callbacks_config.values():
                callback_name = callback_config.pop('type')
                if ClassFactory.is_exists(ClassType.CALLBACK, callback_name):
                    callback_class = ClassFactory.get_cls(
                        ClassType.CALLBACK, callback_name)
                    callback = callback_class(**callback_config)
                    _callbacks.append(callback)
                else:
                    raise ValueError(
                        "Undefined callback {}".format(callback_name))
        else:
            _callbacks = callbacks
        # Sort the callbacks
        metrics_evaluator = None
        model_checkpoint = None
        model_statistics = None
        predefined_callbacks = []
        customized_callbacks = []
        for callback in _callbacks:
            if isinstance(callback, self._predefined_callbacks()):
                if isinstance(callback, MetricsEvaluator):
                    metrics_evaluator = callback
                if isinstance(callback, ModelStatistics):
                    model_statistics = callback
                if isinstance(callback, ModelCheckpoint):
                    model_checkpoint = callback
                else:
                    predefined_callbacks.append(callback)
            else:
                customized_callbacks.append(callback)
        if metrics_evaluator is None:
            metrics_evaluator = MetricsEvaluator()
        if model_checkpoint is None:
            model_checkpoint = ModelCheckpoint()
        _callbacks = [metrics_evaluator, model_checkpoint] + \
            customized_callbacks + predefined_callbacks
        if 'model_statistic' in self.cfg and self.cfg.model_statistic:
            if model_statistics is None:
                model_statistics = ModelStatistics()
            _callbacks = [model_statistics] + _callbacks
        # Creat Callbacklist and set its trainer and pramameters
        self.callbacks = CallbackList(_callbacks)
        _callbacks_params = {
            'epochs': self.epochs,
            'is_chief': self.is_chief,
            'use_cuda': self.use_cuda,
            'do_validation': self.do_validation,
            'is_detection_trainer': self.cfg.is_detection_trainer
        }
        self.callbacks.set_params(_callbacks_params)
        self.callbacks.set_trainer(self)

    def _init_step_functions(self, make_batch=None,
                             train_step=None, valid_step=None):
        # Init make_batch function by user or using the default one
        if self.make_batch is None:
            if make_batch is not None:
                self.make_batch = make_batch
            else:
                self.make_batch = self._default_make_batch

        # Init train_step function by user or using the default one
        if self.train_step is None:
            if train_step is not None:
                self.train_step = train_step
            else:
                self.train_step = self._default_train_step

        # Init valid_step function by user or using the default one
        if self.valid_step is None:
            if valid_step is not None:
                self.valid_step = valid_step
            else:
                self.valid_step = self._default_valid_step

    def _init_all_settings(self):
        """Init all settings from config."""
        if self.cfg.cuda:
            self._init_cuda_setting()
        self._init_hps(self.hps)
        if self.model is None:
            self.model = self._init_model()
        if self.model is not None and self.cfg.cuda:
            self.model = self.model.cuda()
        if self._flag_load_checkpoint:
            self.load_checkpoint()
        else:
            self._load_pretrained_model()
        self.use_syncbn = self.cfg.get('syncbn', False)
        if self.use_syncbn:
            self.model = apex.parallel.convert_syncbn_model(self.model)
        self.epochs = self.cfg.epochs
        self.optimizer = self._init_optimizer()
        self.lr_scheduler = self._init_lr_scheduler()
        self.loss = self._init_loss()
        if self.horovod:
            self._init_horovod_setting()
        self.use_amp = self.cfg.get('amp', False)
        if self.use_amp:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level='O1')
        self.train_loader = self._init_dataloader(mode='train')
        self.valid_loader = self._init_dataloader(mode='test')

    def _init_cuda_setting(self):
        """Init CUDA setting."""
        if not self.cfg.cuda:
            self.cfg.device = -1
            return
        self.cfg.device = self.cfg.cuda if self.cfg.cuda is not True else 0
        self.use_cuda = True
        if self.horovod:
            torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(self.cfg.seed)

    def _init_horovod_setting(self):
        """Init horovod setting."""
        self.is_chief = True
        if self.horovod:
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            if hvd.rank() != 0:
                self.is_chief = False
            else:
                self.is_chief = True

    def _init_model(self, model=None):
        """Load model desc from save path and parse to model."""
        if model is not None:
            return model
        model_cfg = ClassFactory.__configs__.get('model')
        if 'model_desc_file' in model_cfg and model_cfg.model_desc_file is not None:
            desc_file = model_cfg.model_desc_file.replace("{model_zoo}", self.model_zoo_path)
            desc_file = desc_file.replace("{local_base_path}", self.local_base_path)
            if ":" not in desc_file:
                desc_file = os.path.abspath(desc_file)
            if ":" in desc_file:
                local_desc_file = FileOps.join_path(
                    self.local_output_path, os.path.basename(desc_file))
                FileOps.copy_file(desc_file, local_desc_file)
                desc_file = local_desc_file
            if self.horovod:
                hvd.join()
            model_desc = Config(desc_file)
            logging.info("net_desc:{}".format(model_desc))
        elif 'model_desc' in model_cfg and model_cfg.model_desc is not None:
            model_desc = model_cfg.model_desc
        else:
            return None
        if model_desc is not None:
            self.model_desc = model_desc
            net_desc = NetworkDesc(model_desc)
            model = net_desc.to_model()
            return model
        else:
            return None

    def _init_hps(self, hps):
        """Convert trainer values in hps to cfg.

        :param hps: hyperparameters
        :type hps: dict
        """
        if "hps_file" in self.cfg and self.cfg.hps_file is not None:
            hps_file = self.cfg.hps_file.replace("{local_base_path}", self.local_base_path)
            hps = Config(hps_file)
        if hps is not None:
            self.cfg = Config(update_dict(hps.get('trainer'), self.cfg))
            self.hps = hps
        # if 'hps_file' in self.cfg and self.cfg.hps_file is not None:
        #     hps_dict = {}
        #     local_hp_file = FileOps.join_path(self.local_output_path,
        #                                       os.path.basename(self.cfg.hps_file))
        #     FileOps.copy_file(self.cfg.hps_file, local_hp_file)
        #     with open(local_hp_file) as json_file:
        #         hps_dict = json.load(json_file)
        #         hps_dict = Config(hps_dict)
        #     self.cfg = Config(update_dict(hps_dict.get('trainer'), self.cfg))
        #     if self.hps is None:
        #         self.hps = hps_dict
        #     else:
        #         update_dict(hps_dict, self.hps)
        #     logging.info("load hps file:{}".format(self.hps))

    def _init_optimizer(self, optimizer=None):
        """Init optimizer from torch.optim according to optim type in config."""
        if optimizer is not None:
            return optimizer
        optim_config = self.cfg.optim.copy()
        optim_name = optim_config.pop('type')
        if ClassFactory.is_exists(ClassType.OPTIM, optim_name):
            optim_class = ClassFactory.get_cls(ClassType.OPTIM, optim_name)
        else:
            optim_class = getattr(importlib.import_module('torch.optim'),
                                  optim_name)
        learnable_params = [
            param for param in self.model.parameters() if param.requires_grad
        ]
        optimizer = optim_class(learnable_params, **optim_config)
        if self.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer,
                                                 named_parameters=self.model.named_parameters(),
                                                 compression=hvd.Compression.none)
        return optimizer

    def _init_loss(self, loss_fn=None):
        """Init loss function from torch according to type in config."""
        if loss_fn is not None:
            return loss_fn
        loss_config = self.cfg.loss.copy()
        loss_name = loss_config.pop('type')
        if NetworkFactory.is_exists(NetTypes.LOSS, loss_name):
            loss_class = NetworkFactory.get_network(NetTypes.LOSS, loss_name)
        elif ClassFactory.is_exists('trainer.loss', loss_name):
            loss_class = ClassFactory.get_cls('trainer.loss', loss_name)
        else:
            loss_class = getattr(importlib.import_module('torch.nn'), loss_name)
        loss_fn = loss_class(**loss_config)
        if self.cfg.cuda:
            loss_fn = loss_fn.cuda()
        return loss_fn

    def _init_lr_scheduler(self, scheduler=None):
        """Init lr scheduler from torch.optim.lr_scheduler according to type in config."""
        if scheduler is not None:
            return scheduler
        scheduler_config = self.cfg.lr_scheduler.copy()
        scheduler_name = scheduler_config.pop('type')
        if ClassFactory.is_exists(ClassType.LR_SCHEDULER, scheduler_name):
            scheduler_class = ClassFactory.get_cls(
                ClassType.LR_SCHEDULER, scheduler_name)
        else:
            scheduler_class = getattr(importlib.import_module(
                'torch.optim.lr_scheduler'), scheduler_name)
        return scheduler_class(self.optimizer, **scheduler_config)

    def _init_metrics(self, metrics=None):
        """Init metrics."""
        if metrics is not None:
            return metrics
        else:
            return Metrics(self.cfg.metric)

    def _init_dataloader(self, mode, loader=None):
        """Init dataloader."""
        if loader is not None:
            return loader
        if self.horovod:
            if hvd.local_rank() == 0:
                Dataset()
            hvd.join()
        if mode == "train" and self.hps is not None and self.hps.get("dataset") is not None:
            dataset = Dataset(mode=mode, hp=self.hps)
        else:
            dataset = Dataset(mode=mode)
        if self.horovod:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank())
            dataset.sampler = sampler
        return dataset.dataloader

    def _load_pretrained_model(self):
        if self.model is None:
            return
        if "pretrained_model_file" in self.cfg and self.cfg.pretrained_model_file is not None:
            model_file = self.cfg.pretrained_model_file.replace(
                "{model_zoo}", self.model_zoo_path)
            model_file = os.path.abspath(model_file)
            ckpt = torch.load(model_file)
            self.model.load_state_dict(ckpt)
            return

    def load_checkpoint(self, worker_id=None, step_name=None, saved_folder=None):
        """Load checkpoint."""
        if saved_folder is None:
            if worker_id is None:
                worker_id = self.worker_id
            if step_name is None:
                step_name = self.step_name
            saved_folder = self.get_local_worker_path(step_name, worker_id)
        checkpoint_file = FileOps.join_path(
            saved_folder, self.checkpoint_file_name)
        model_pickle_file = FileOps.join_path(
            saved_folder, self.model_pickle_file_name)
        try:
            with open(model_pickle_file, 'rb') as f:
                model = pickle.load(f)
                ckpt = torch.load(
                    checkpoint_file, map_location=torch.device('cpu'))
                model.load_state_dict(ckpt['weight'])
                if self.cfg.cuda:
                    model = model.cuda()
                self.model = model
        except Exception:
            logging.info(
                'Checkpoint file is not existed, use default model now.')
            return

    def _save_checkpoint(self, epoch):
        """Save checkpoint."""
        checkpoint_file = FileOps.join_path(
            self.get_local_worker_path(), self.checkpoint_file_name)
        model_pickle_file = FileOps.join_path(
            self.get_local_worker_path(), self.model_pickle_file_name)
        # pickle model
        with open(model_pickle_file, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'weight': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        torch.save(ckpt, checkpoint_file)

    def _save_performance(self, performance):
        """Save performance into performance.txt.

        :param performance: performance value
        """
        logging.debug("performance=%s", str(performance))
        self.performance_file = FileOps.join_path(
            self.get_local_worker_path(), self.performance_file_name)
        with open(self.performance_file, 'w') as f:
            if isinstance(performance, list):
                for p in performance:
                    f.write("{}\n".format(p))
            elif isinstance(performance, dict):
                for p in performance.values():
                    f.write("{}\n".format(p))
            else:
                f.write("{}".format(performance))

    def get_performance(self, worker_id=None, step_name=None, saved_folder=None):
        """Read Performance values from perform.txt.

        :param step_name: step name in the pipeline.
        :type step_name: str.
        :param worker_id: the worker's worker id.
        :type worker_id: str.
        :return: performance value
        :rtype: int/float/list

        """
        if saved_folder is None:
            if worker_id is None:
                worker_id = self.worker_id
            if step_name is None:
                step_name = self.step_name
            saved_folder = self.get_local_worker_path(step_name, worker_id)
        performance_file = FileOps.join_path(
            saved_folder, self.performance_file_name)
        if not os.path.isfile(performance_file):
            logging.info(
                "Performance file is not exited, file={}".format(performance_file))
            return []
        with open(performance_file, 'r') as f:
            performance = []
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                data = json.loads(line)
                if isinstance(data, list):
                    data = data[0]
                performance.append(data)
            logging.info("performance={}".format(performance))
        return performance

    def _metric_average(self, val, name):
        """Do metric average.

        :param val: input value
        :param name: metric name
        :return:
        """
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    @property
    def _first_rank(self):
        """Check if the first rank."""
        if self.horovod and hvd.rank() != 0:
            return False
        else:
            return True

    def output_model_desc(self, id=None, model_desc=None, performance=None):
        """Save model desc and performance.

        :param id: model desc id, usally worker id instead.
        :type id: int or str.
        :param model_desc: model description.
        :type model_desc: json.
        :param performance: performance value, eg. {"accuracy": 98.23}.
        :type performance: json.

        """
        if id is None:
            id = self.worker_id
        if model_desc is None:
            if not hasattr(self, "model_desc"):
                logger.error(
                    "Failed to save model desc, param 'model_desc' is not assigned.")
                return
            model_desc = self.model_desc
        _file = FileOps.join_path(
            self.local_output_path, self.step_name, "model_desc_{}.json".format(str(id)))
        FileOps.make_base_dir(_file)
        try:
            with open(_file, "w") as f:
                json.dump(model_desc, f)
        except Exception as ex:
            logger.error("Failed to save model desc, file={}, desc={}, msg={}".format(
                _file, model_desc, str(ex)))
            return
        if performance is not None:
            self.output_evaluate_result(id, performance)

    def _backup(self):
        """Backup result worker folder."""
        if self.need_backup is True and self.backup_base_path is not None:
            backup_worker_path = FileOps.join_path(
                self.backup_base_path, self.get_worker_subpath())
            FileOps.copy_folder(
                self.get_local_worker_path(), backup_worker_path)

    def _save_visual_data(self, is_train=True, pfms=None, loss=None, lr=None):
        # TODO Will move to metric base class later.
        for _name, value in pfms.items():
            if is_train:
                _name = "{}_{}".format("t", _name)
            else:
                _name = "{}_{}".format("v", _name)
            if isinstance(value, list):
                for i, _item in enumerate(value):
                    _name = "{}_{}".format(_name, i)
                    self.visual_data[_name] = _item.data.item()
            elif isinstance(value, dict):
                for k, v in value.keys():
                    _name = "{}_{}".format(k, i)
                    self.visual_data[_name] = v
            elif value is not None:
                self.visual_data[_name] = value.data.item()
        if loss is not None:
            self.visual_data["loss"] = loss
        if lr is not None:
            self.visual_data["lr"] = lr

    def output_evaluate_result(self, id=None, performance=None, evaluate_type="gpu"):
        """Save model performance.

        :param id: model desc id, usally worker id instead.
        :type id: int or str.
        :param performance: performance value, eg. {"accuracy": 98.23}.
        :type performance: json.
        :param evaluate_type: evaluate type, eg. "gpu", "davinci", "arm".
        :type evaluate_type: str.
        """
        if performance is None:
            return
        if id is None:
            id = self.worker_id
        _file = FileOps.join_path(
            self.local_output_path, self.step_name, "performance_{}_{}.txt".format(evaluate_type, str(id)))
        FileOps.make_base_dir(_file)
        try:
            performance = str(performance)
            with open(_file, "w") as f:
                f.write(performance)
        except Exception as ex:
            logger.error("Failed to save performance, file={}, pfm={}, msg={}".format(
                _file, performance, str(ex)))
            return

    def output_hps(self, id=None, hps=None):
        """Save model desc and performance.

        :param id: model desc id, usually worker id.
        :type id: int or str.
        :param hps: hyper parameters.
        :type hps: json.

        """
        if id is None:
            id = self.worker_id
        if hps is None:
            if not hasattr(self, "hps"):
                logger.error(
                    "Failed to save hyperparameters, param 'hps' is not assigned.")
                return
            hps = self.hps
        _file = FileOps.join_path(
            self.local_output_path, self.step_name, "hyperparameters.json")
        FileOps.make_base_dir(_file)
        try:
            with open(_file, "w") as f:
                json.dump({str(id): hps}, f)
        except Exception as ex:
            logger.error("Failed to save hyperparameters, file={}, hps={}, msg={}".format(
                _file, hps, str(ex)))
            return

    def output_model(self, id=None, model=None, model_desc=None, performance=None):
        """Save model, model description, performance.

        :param id: model desc id, usually worker id.
        :type id: int or str.
        :param model: hyper parameters.
        :type hps: json.

        """
        if id is None:
            id = self.worker_id
        if model is None:
            if not hasattr(self, "model"):
                logger.error(
                    "Failed to save model, param 'model' is not assigned.")
                return
            model = self.model
        if model_desc is None:
            if not hasattr(self, "model_desc"):
                logger.error(
                    "Failed to save model, param 'model_desc' is not assigned.")
                return
            model_desc = self.model_desc
        _pth_file = FileOps.join_path(
            self.local_output_path, self.step_name, "model_{}.pth".format(id))
        FileOps.make_base_dir(_pth_file)
        try:
            torch.save(model.state_dict(), _pth_file)
        except Exception as ex:
            logger.error("Failed to save model pth, file={}, msg={}".format(
                _pth_file, str(ex)))
        self.output_model_desc(id, model_desc, performance)
