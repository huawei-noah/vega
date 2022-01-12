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

"""Mindspore Trainer."""

import os
import logging
import vega
from mindspore import context
from mindspore.train import Model as MsModel
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore import save_checkpoint
from vega.trainer.callbacks.ms_callbacks import EvalCallBack
from vega.trainer.trainer_base import TrainerBase
from vega.trainer.modules.optimizer import Optimizer
from vega.trainer.modules.lr_schedulers import LrScheduler
from vega.modules.loss import Loss
from vega.common import ClassFactory, ClassType
from vega.common.general import General


@ClassFactory.register(ClassType.TRAINER)
class TrainerMs(TrainerBase):
    """Trainer mindspore class."""

    def build(self):
        """Build the trainer by assembling the necessary components."""
        super().build()
        no_decay_params = self.config.optimizer.params.get("no_decay_params", [])
        if self.config.lr_scheduler.params:
            self.lr_scheduler = LrScheduler()
            dynamic_lr = self.lr_scheduler()(base_lr=self.config.optimizer.params["lr"],
                                             global_step=self.config.epochs * len(self.train_loader),
                                             total_epoch=self.config.epochs)

            self.optimizer = Optimizer()(model=self.model, dynamic_lr=dynamic_lr, no_decay_params=no_decay_params)
        else:
            self.optimizer = Optimizer()(model=self.model, no_decay_params=no_decay_params)
        logging.debug(f"The optimizer is {self.optimizer}.")
        if hasattr(self.model, 'add_loss'):
            loss_cls = Loss()()
            self.model.add_loss(loss_cls)
            self.loss = self.model.overall_loss()
        else:
            self.loss = Loss()()
        self.metric_name = self.config.metric.type

        # Some trainer has different train batch size from valid batch
        self.train_metrics = None
        self.valid_metrics = self._init_metrics()
        self.ms_metrics = self.valid_metrics() if isinstance(self.valid_metrics(), dict) else {
            self.metric_name: self.valid_metrics()}

        if self.use_amp:
            loss_scale = FixedLossScaleManager(self.config.loss_scale, drop_overflow_update=False)
            logging.info(f"Use auto mix precision, and loss scale is {self.config.loss_scale},"
                         f"loss_scale_manager is {loss_scale}.")
            self.ms_model = MsModel(network=self.model,
                                    loss_fn=self.loss,
                                    optimizer=self.optimizer,
                                    metrics=self.ms_metrics,
                                    loss_scale_manager=loss_scale,
                                    amp_level=self.config.opt_level,
                                    keep_batchnorm_fp32=self.config.keep_batchnorm_fp32)
        else:
            self.ms_model = MsModel(network=self.model,
                                    loss_fn=self.loss,
                                    optimizer=self.optimizer,
                                    metrics=self.ms_metrics)

        if not self.config.with_train:
            save_path = self.get_local_worker_path(self.step_name, self.worker_id)
            ckpt_file_name = os.path.join(save_path, "model_" + str(self.worker_id) + ".ckpt")
            save_checkpoint(self.model, ckpt_file_name)
            logging.info("Save checkpoint file without training.")

    def init_env(self):
        """Init mindspore trainer environment."""
        super().init_env()
        self._init_ms_context()

    def _train_epoch(self):
        config_ck = CheckpointConfig(save_checkpoint_steps=self.config.save_steps, keep_checkpoint_max=1)
        # save the network model and parameters for subsequence fine-tuning
        save_path = self.get_local_worker_path(self.step_name, self.worker_id)
        ckpoint_cb = ModelCheckpoint(config=config_ck, directory=save_path)
        loss_cb = LossMonitor(per_print_times=1)
        time_cb = TimeMonitor(data_size=self.train_loader.get_dataset_size())
        callback_list = [ckpoint_cb, loss_cb, time_cb]
        if self.config.eval_per_epoch and not self.config.mixup:
            eval_cb = EvalCallBack(self.ms_model, self.valid_loader, self.dataset_sink_mode, self)
            callback_list.append(eval_cb)

        try:
            self.ms_model.train(epoch=self.epochs,
                                train_dataset=self.train_loader,
                                callbacks=callback_list,
                                dataset_sink_mode=self.dataset_sink_mode)
        except RuntimeError as e:
            logging.warning(f"failed to train the model, skip it, message: {str(e)}")

    def _valid_epoch(self):
        if self.config.mixup and self.config.loss.type == 'CrossEntropyLoss':
            from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
            loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True)
            self.ms_model = MsModel(network=self.model,
                                    loss_fn=loss_fn,
                                    optimizer=self.optimizer,
                                    metrics=self.ms_metrics)
        self.callbacks.before_valid()

        try:
            eval_metrics = self.ms_model.eval(valid_dataset=self.valid_loader,
                                              dataset_sink_mode=self.dataset_sink_mode)
            self.valid_metrics.update(eval_metrics)
            valid_logs = dict()
            valid_logs['cur_valid_perfs'] = self.valid_metrics.results

            self.callbacks.after_valid(valid_logs)
        except RuntimeError as exc:
            logging.warning("RuntimeError occurred when eval the model. Skip eval this model.")
            logging.warning("The RuntimeError message is : {}.".format(exc))

    def _init_ms_context(self):
        mode = General.ms_execute_mode
        logging.info(f"Run train/val in mode: {mode}.")
        if vega.is_npu_device():
            logging.info(f"minspore context, mode: {context.get_context('mode')}, "
                         f"target: {context.get_context('device_target')}, "
                         f"device_id: {context.get_context('device_id')}")
            logging.info(f"DEVICE_ID: {os.environ['DEVICE_ID']}")
            context.set_context(mode=mode, device_target="Ascend")
        else:
            context.set_context(mode=mode, device_target="CPU")

        self.dataset_sink_mode = General.dataset_sink_mode
        logging.info(f"Dataset_sink_mode:{self.dataset_sink_mode}.")
