# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Base Trainer."""
import glob
import logging

import zeus

if zeus.is_torch_backend():
    import torch

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
if zeus.is_npu_device():
    from hccl.manage.api import get_rank_size
    from hccl.manage.api import get_rank_id

from zeus.common import FileOps, init_log
from zeus.common.class_factory import ClassFactory, ClassType
from zeus.common.config import Config
from zeus.trainer.callbacks import CallbackList
from zeus.trainer.conf import TrainerConfig
from zeus.trainer.distributed_worker import DistributedWorker
from zeus.trainer.modules.losses import Loss
from zeus.trainer.modules.lr_schedulers import LrScheduler
from zeus.trainer.modules.optimizer import Optimizer
from zeus.trainer.utils import WorkerTypes
from zeus.datasets import Adapter
from zeus.common.general import General


class TrainerBase(DistributedWorker):
    """Trainer base class."""

    config = TrainerConfig()

    def __init__(self, model=None, id=None, hps=None, load_ckpt_flag=False,
                 model_desc=None, lazy_build=True, **kwargs):
        super().__init__()

        self.worker_type = WorkerTypes.TRAINER
        TrainerBase.__worker_id__ += 1
        if id is not None:
            self._worker_id = id
        else:
            self._worker_id = TrainerBase.__worker_id__

        # Data Memeber list of Trainer
        self.is_chief = True
        self.use_cuda = self.config.cuda
        self.epochs = self.config.epochs
        self.do_validation = True
        self.auto_save_ckpt = True
        self.auto_save_perf = True
        self.skip_train = False
        self.valid_interval = self.config.valid_interval
        self.hps = hps
        self.model = model
        self.model_desc = model_desc
        self.optimizer = None
        self.lr_scheduler = None
        self.loss = None
        self.use_syncbn = self.config.syncbn
        self.use_amp = self.config.amp
        self.train_metrics = None
        self.valid_metrics = None
        self.call_metrics_on_train = self.config.call_metrics_on_train
        self.train_verbose = self.config.train_verbose
        self.valid_verbose = self.config.valid_verbose
        self.train_report_steps = self.config.train_report_steps
        self.valid_report_steps = self.config.valid_report_steps
        self.train_loader = None
        self.valid_loader = None
        self.train_step = None
        self.valid_step = None
        self.make_batch = None
        self.model_fn = None
        self.train_input_fn = None
        self.valid_input_fn = None
        self.callbacks = None
        self.performance = None
        self.runtime = None
        self.load_checkpoint = False
        self._resume_training = False
        self._start_epoch = 0
        self.visual_data = {}
        self.load_ckpt_flag = load_ckpt_flag
        self.distributed = self.config.distributed
        # Used by TimmTrainerCallbacks since it builds its trainer in
        # the before_train callback
        self.lazy_built = self.config.lazy_built
        # Indicate whether the necessary components of a trainer
        # has been built for running
        self._world_size = 1
        self._rank_id = 0
        self._local_rank_id = 0
        self.config.kwargs = kwargs
        self.checkpoint_file_name = 'checkpoint.pth'
        self.model_pickle_file_name = 'model.pkl'
        worker_path = self.get_local_worker_path()
        self.model_path = FileOps.join_path(worker_path, self.model_pickle_file_name)
        self.checkpoint_file = FileOps.join_path(worker_path, self.checkpoint_file_name)
        self.weights_file = FileOps.join_path(worker_path, "model_{}.pth".format(self.worker_id))
        self.loss_input = kwargs.get('loss_input', None)
        self.gpu_nums = kwargs.get('gpu_nums', 1)
        if TrainerConfig.model_desc is None:
            TrainerConfig.model_desc = model_desc
        if not lazy_build:
            self.init_trainer()

    def init_trainer(self):
        """Init Trainer."""
        init_log(level=General.logger.level,
                 log_file="log_worker_{}.txt".format(self.worker_id),
                 log_path=self.get_local_worker_path())
        self._set_default_funcs()
        self._set_condition()
        self._init_callbacks()
        self.callbacks.init_trainer()

        self.init_train_op()

    def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        init_log(level=General.logger.level,
                 log_file="log_worker_{}.txt".format(self.worker_id),
                 log_path=self.local_log_path)
        self._set_default_funcs()
        self._set_condition()
        self._init_callbacks()
        self.callbacks.init_trainer()
        if not self.lazy_built:
            self.build()
        self._train_loop()

    def build(self):
        """Build the trainer by assembling the necessary components."""
        logging.debug("Trainer Config: {}".format(self.config))
        self._init_hps()
        self.do_validation = self.config.with_valid
        self.use_syncbn = self.config.syncbn
        if self.use_syncbn and zeus.is_torch_backend():
            self.model = apex.parallel.convert_syncbn_model(self.model)
        self.train_loader = self._init_dataloader(mode='train')
        self.valid_loader = self._init_dataloader(mode='val')
        self.batch_num_train = self.train_loader.get_dataset_size() if zeus.is_ms_backend() else len(self.train_loader)
        self.batch_num_valid = self.valid_loader.get_dataset_size() if zeus.is_ms_backend() else len(self.valid_loader)

    def train(self, inputs, labels):
        """Train model."""
        pass

    def predict(self, input):
        """Inference model."""
        pass

    def save(self, file_name):
        """Save model."""
        pass

    def load(self, model_name, by_name):
        """Load model."""
        pass

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        pass

    def get_weights(self):
        """Get the weights."""
        pass

    def init_trainer_op(self):
        """Init Train Op."""
        pass

    def _train_epoch(self):
        pass

    def _valid_epoch(self):
        pass

    def _set_default_funcs(self):
        pass

    def _set_condition(self):
        pass

    def _init_tf_estimator(self):
        pass

    def _init_horovod_setting(self):
        """Init horovod setting."""
        self.is_chief = True

    def _init_hps(self, hps=None):
        """Load hps from file."""
        if hps is not None:
            self.hps = hps
        elif self.config.hps_file is not None:
            desc_file = self.config.hps_file.replace("{local_base_path}", self.local_base_path)
            self.hps = Config(desc_file)
        elif self.config.hps_folder is not None:
            folder = self.config.hps_folder.replace("{local_base_path}", self.local_base_path)
            pattern = FileOps.join_path(folder, "desc_*.json")
            desc_file = glob.glob(pattern)[0]
            self.hps = Config(desc_file)
        if self.hps and self.hps.get('trainer'):
            self.config.from_json(self.hps.get('trainer'))
            self.load_checkpoint = self.config.load_checkpoint
        self.epochs = self.config.epochs

    def _init_minimize_op(self, loss, global_step, var_list=None):
        """Init loss minimize operation, include loss scale method."""
        loss_scale = self.config.loss_scale if self.use_amp else 1.
        if loss_scale != 1:
            scaled_grad_vars = self.optimizer.compute_gradients(loss * loss_scale, var_list=var_list)
            unscaled_grad_vars = []
            for grad, var in scaled_grad_vars:
                unscaled_grad_vars.append((grad, var) if grad is None else (grad / loss_scale, var))
            minimize_op = self.optimizer.apply_gradients(unscaled_grad_vars, global_step)
        else:
            grad_vars = self.optimizer.compute_gradients(loss, var_list=var_list)
            minimize_op = self.optimizer.apply_gradients(grad_vars, global_step)
        return minimize_op

    def _init_metrics(self, metrics=None):
        """Init metrics."""
        if metrics is not None:
            return metrics
        else:
            if zeus.is_torch_backend():
                from zeus.metrics.pytorch.metrics import Metrics
            elif zeus.is_tf_backend():
                from zeus.metrics.tensorflow.metrics import Metrics
            elif zeus.is_ms_backend():
                from zeus.metrics.mindspore.metrics import Metrics
            return Metrics()

    def _init_dataloader(self, mode, loader=None):
        """Init dataloader."""
        if loader is not None:
            return loader
        if mode == "train" and self.hps is not None and self.hps.get("dataset") is not None:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            dataset = dataset_cls(mode=mode, hps=self.hps.get("dataset"))
        else:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            dataset = dataset_cls(mode=mode)
        if self.distributed and mode == "train":
            dataset.set_distributed(self._world_size, self._rank_id)
        # adapt the dataset to specific backend
        dataloader = Adapter(dataset).loader
        return dataloader

    def _train_loop(self):
        """Do the training with data, callbacks and step functions etc."""
        # Allow user to build trainer in before_train() callback, but they
        # should set lazy_built in configuration file to True
        self.callbacks.before_train()
        if self.skip_train:
            return
        repeat_time = 1 if zeus.is_ms_backend() else self.epochs
        for epoch in range(self._start_epoch, repeat_time):
            epoch_logs = {'train_num_batches': self.batch_num_train}
            if self.do_validation:
                epoch_logs.update({'valid_num_batches': self.batch_num_valid})
            self.callbacks.before_epoch(epoch, epoch_logs)
            self._train_epoch()
            if self.do_validation and self._should_run_validation(epoch):
                self._valid_epoch()
            self.callbacks.after_epoch(epoch)
        self.callbacks.after_train()
        if self.distributed:
            self._shutdown_distributed()

    def _should_run_validation(self, epoch):
        # Zero valid_interval means doesn't run _valid_loop of the trainer
        # and user may provide _valid_loop in other callbacks
        if self.valid_interval == 0:
            return False
        else:
            return epoch % self.valid_interval == 0 or (epoch + 1) == self.epochs

    def _init_callbacks(self):
        disables = []
        customs = self.config.callbacks or []
        if customs and not isinstance(customs, list):
            customs = [customs]
        if not self.config.model_statistics:
            disables.append('ModelStatistics')
        self.callbacks = CallbackList(customs, disables)
        self.callbacks.set_trainer(self)

    def _metric_average(self, val, name):
        """Do metric average.

        :param val: input value
        :param name: metric name
        :return:
        """
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def _backup(self):
        """Backup result worker folder."""
        if self.need_backup is True and self.backup_base_path is not None:
            backup_worker_path = FileOps.join_path(
                self.backup_base_path, self.get_worker_subpath())
            FileOps.copy_folder(
                self.get_local_worker_path(self.step_name, self.worker_id), backup_worker_path)

    def _shutdown_distributed(self):
        if zeus.is_npu_device() and self.distributed:
            self.sess.run(self.npu_shutdown)
            self.sess.close()
