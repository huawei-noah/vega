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

"""Base Trainer."""

import os
import glob
import logging
import vega
from vega.common import FileOps
from vega.common.class_factory import ClassFactory, ClassType
from vega.common.config import Config
from vega.trainer.callbacks import CallbackList
from vega.trainer.conf import TrainerConfig
from vega.trainer.distributed_worker import DistributedWorker
from vega.trainer.utils import WorkerTypes
from vega.datasets import Adapter
from vega.common.general import General
from vega.common.utils import update_dict
from vega.common.wrappers import train_process_wrapper


class TrainerBase(DistributedWorker):
    """Trainer base class."""

    def __init__(self, model=None, id=None, hps=None, load_ckpt_flag=False,
                 model_desc=None, multi_task=None, horovod=False, hccl=False,
                 **kwargs):
        super().__init__()

        self.config = TrainerConfig()
        self.worker_type = WorkerTypes.TRAINER
        if id is not None:
            self._worker_id = id

        self.actions_list = self.config.actions_list
        # Data Memeber list of Trainer
        self.is_chief = True
        self.epochs = self.config.epochs
        self.do_validation = True
        self.auto_save_ckpt = True
        self.auto_save_perf = True
        self.save_ext_model = self.config.save_ext_model
        self.skip_train = False
        self.valid_interval = self.config.valid_interval
        self.hps = hps
        self.model = model
        self.model_desc = model_desc
        self.optimizer = None
        self.lr_scheduler = None
        self.loss = None
        self.use_syncbn = self.config.syncbn
        self.use_amp = self.config.use_amp
        self.train_metrics = None
        self.valid_metrics = None
        self.call_metrics_on_train = self.config.call_metrics_on_train
        self.train_verbose = self.config.train_verbose
        self.valid_verbose = self.config.valid_verbose
        self.train_report_steps = self.config.train_report_steps
        self.valid_report_steps = self.config.valid_report_steps
        self.multi_task = multi_task
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
        self.best_performance = None
        self.runtime = None
        self.load_checkpoint = False
        self.load_weights_file = self.config.load_weights_file
        self._resume_training = False
        self.ext_model = None
        self._start_epoch = 0
        self.visual_data = {}
        self.load_ckpt_flag = load_ckpt_flag
        self.ddp = General._parallel and General.devices_per_trainer > 1 and vega.is_gpu_device()
        self.horovod = horovod
        self.hccl = hccl
        self.num_workers = General.cluster.num_workers
        self.sampler = None
        # Used by TimmTrainerCallbacks since it builds its trainer in
        # the before_train callback
        self.lazy_built = self.config.lazy_built
        # Indicate whether the necessary components of a trainer
        # has been built for running
        self._next_rung = False
        self.config.kwargs = kwargs
        self.checkpoint_file_name = 'checkpoint.pth'
        self.model_pickle_file_name = 'model.pkl'
        worker_path = self.get_local_worker_path()
        self.model_path = FileOps.join_path(worker_path, self.model_pickle_file_name)
        self.checkpoint_file = FileOps.join_path(worker_path, self.checkpoint_file_name)
        if self.multi_task:
            self.weights_file = FileOps.join_path(worker_path, "model_{}.pth".format(self.multi_task))
        else:
            self.weights_file = FileOps.join_path(worker_path, "model_{}.pth".format(self.worker_id))
        self.loss_input = kwargs.get('loss_input', None)
        self.gpu_nums = kwargs.get('gpu_nums', 1)
        self.use_unsupervised_pretrain = self.config.use_unsupervised_pretrain
        if TrainerConfig.model_desc is None:
            TrainerConfig.model_desc = model_desc
        self.standalone = General.cluster.master_ip is None or General.message_port is None

    @train_process_wrapper
    def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        if self.standalone:
            logging.info("Standalone mode. The result data will not be sent to server through report.")
        self.init_env()
        self._init_callbacks()
        self.callbacks.init_trainer()
        self.set_training_settings()
        if not self.lazy_built:
            self.build()
        self._train_loop()
        self.closeout()
        return self.model

    def build(self):
        """Build the trainer by assembling the necessary components."""
        logging.debug("Trainer Config: {}".format(self.config))
        self._init_hps()
        self.do_validation = self.config.with_valid
        self.use_syncbn = self.config.syncbn
        if self.use_syncbn and vega.is_torch_backend():
            import apex
            self.model = apex.parallel.convert_syncbn_model(self.model)
        if not self.train_loader:
            self.train_loader = self._init_dataloader(mode='train')
        if not self.valid_loader:
            self.valid_loader = self._init_dataloader(mode='val')
        self.batch_num_train = len(self.train_loader)
        self.batch_num_valid = len(self.valid_loader)

    def set_training_settings(self):
        """Set training settings."""
        pass

    def init_env(self):
        """Init trainer environment."""
        self.num_workers = General.cluster.num_workers
        if self.hccl:
            self.rank_id = int(os.environ.get('RANK_ID', "0"))
            self.device_id = int(os.environ.get('DEVICE_ID', "0"))
            if not General.cluster.show_all_ranks:
                self.is_chief = "-" not in str(self.worker_id)
        elif self.horovod:
            if vega.is_torch_backend():
                import horovod.torch as hvd
            elif vega.is_tf_backend():
                import horovod.tensorflow as hvd
                hvd.init()
            self.rank_id = hvd.rank()
            if not General.cluster.show_all_ranks:
                self.is_chief = self.rank_id == 0

    def _init_hps(self, hps=None):
        """Load hps from file."""
        # load config
        if hps is not None:
            pass
        elif self.config.hps_file is not None:
            hps_file = self.config.hps_file.replace("{local_base_path}", self.local_base_path)
            if os.path.isdir(hps_file):
                pattern = os.path.join(hps_file, "hps_*.json")
                hps_file = glob.glob(pattern)[0]
            hps = Config(hps_file)
            if "trainer" in hps:
                if "epochs" in hps["trainer"]:
                    hps["trainer"].pop("epochs")
                if "checkpoint_path" in hps["trainer"]:
                    hps["trainer"].pop("checkpoint_path")
        # merge config
        if not self.hps:
            self.hps = hps
        elif hps:
            self.hps = update_dict(self.hps, hps)
        # set config
        if self.hps and self.hps.get('trainer'):
            self.config.from_dict(self.hps.get('trainer'))
            self.load_checkpoint = self.config.load_checkpoint
        self.epochs = self.config.epochs

    def _init_metrics(self, metrics=None):
        """Init metrics."""
        if metrics is not None:
            return metrics
        else:
            if vega.is_torch_backend():
                from vega.metrics.pytorch.metrics import Metrics
            elif vega.is_tf_backend():
                from vega.metrics.tensorflow.metrics import Metrics
            elif vega.is_ms_backend():
                from vega.metrics.mindspore.metrics import Metrics
            return Metrics()

    def _init_dataloader(self, mode, loader=None, transforms=None):
        """Init dataloader."""
        if loader is not None:
            return loader
        dataset = self._init_dataset(mode)
        if transforms is not None:
            dataset.transforms = transforms
        if (self.hccl or self.horovod) and mode == "train":
            dataset.set_distributed(self.num_workers, self.rank_id)
        # adapt the dataset to specific backend
        adapter = Adapter(dataset)
        if (self.hccl or self.horovod) and mode == "train" and hasattr(adapter, "sampler"):
            self.sampler = adapter.sampler
        dataloader = adapter.loader
        return dataloader

    def _init_dataset(self, mode):
        """Init dataset."""
        if mode == "train" and self.hps is not None and self.hps.get("dataset") is not None:
            if self.hps.get("dataset") and self.hps.get("dataset").get('type'):
                dataset_cls = ClassFactory.get_cls(ClassType.DATASET, self.hps.get("dataset").get('type'))
            else:
                dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            dataset = dataset_cls(mode=mode, hps=self.hps.get("dataset"))
        elif self.hps:
            if self.hps.get("dataset") and self.hps.get("dataset").get('type'):
                dataset_cls = ClassFactory.get_cls(ClassType.DATASET, self.hps.get("dataset").get('type'))
                dataset = dataset_cls(mode=mode, hps=self.hps.get("dataset"))
            else:
                dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
                dataset = dataset_cls(mode=mode)
        else:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            dataset = dataset_cls(mode=mode)
        return dataset

    def _train_loop(self):
        """Do the training with data, callbacks and step functions etc."""
        # Allow user to build trainer in before_train() callback, but they
        # should set lazy_built in configuration file to True
        self.callbacks.before_train()
        if self.skip_train:
            return

        if self.use_unsupervised_pretrain and vega.is_torch_backend():
            from .simclr.transforms import TransformsSimCLR
            from .simclr.train import simclr_train
            train_loader = self._init_dataloader(mode="train", transforms=TransformsSimCLR())
            self.model = simclr_train(self.model, train_loader)

        while True:
            repeat_time = 1 if vega.is_ms_backend() else self.epochs
            repeat_time = 1 if vega.is_tf_backend() and self.config.train_in_once else repeat_time
            for epoch in range(self._start_epoch, repeat_time):
                epoch_logs = {'train_num_batches': self.batch_num_train}
                if self.do_validation:
                    epoch_logs.update({'valid_num_batches': self.batch_num_valid})
                self.callbacks.before_epoch(epoch, epoch_logs)
                if self.config.with_train and hasattr(self, "_train_epoch"):
                    self._train_epoch()
                if self.do_validation and hasattr(self, "_valid_epoch") and self._should_run_validation(epoch):
                    self._valid_epoch()
                self.callbacks.after_epoch(epoch)
            self.callbacks.after_train()
            if not self._next_rung:
                break

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

    def _backup(self):
        """Backup result worker folder."""
        if self.need_backup is True and self.backup_base_path is not None:
            backup_worker_path = FileOps.join_path(
                self.backup_base_path, self.get_worker_subpath())
            FileOps.copy_folder(
                self.get_local_worker_path(self.step_name, self.worker_id), backup_worker_path)

    def closeout(self):
        """Closeout."""
        pass
