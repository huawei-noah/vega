# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Mindspore Trainer."""

import os
from mindspore import context
from mindspore.train import Model as MsModel
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from vega.trainer.callbacks.ms_callbacks import EvalCallBack
import vega
from vega.trainer.trainer_base import TrainerBase
from vega.trainer.modules.optimizer import Optimizer
from vega.trainer.modules.lr_schedulers import LrScheduler
from vega.modules.loss import Loss
from vega.common import ClassFactory, ClassType
import logging
from mindspore.communication.management import init as hccl_init
from mindspore.context import ParallelMode


@ClassFactory.register(ClassType.TRAINER)
class TrainerMs(TrainerBase):
    """Trainer mindspore class."""

    def build(self):
        """Build the trainer by assembling the necessary components."""
        super().build()
        if self.config.lr_scheduler.params:
            self.lr_scheduler = LrScheduler()
            dynamic_lr = self.lr_scheduler()(base_lr=self.config.optimizer.params["lr"],
                                             global_step=self.config.epochs * len(self.train_loader),
                                             total_epoch=self.config.epochs)
            self.optimizer = Optimizer()(model=self.model, dynamic_lr=dynamic_lr)
        else:
            self.optimizer = Optimizer()(model=self.model)
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

        self.ms_model = MsModel(network=self.model,
                                loss_fn=self.loss,
                                optimizer=self.optimizer,
                                metrics=self.ms_metrics)

    def _set_condition(self):
        self._init_ms_context()
        self._init_distributed_setting()

    def _train_epoch(self):
        config_ck = CheckpointConfig(save_checkpoint_steps=self.config.save_steps, keep_checkpoint_max=1)
        # save the network model and parameters for subsequence fine-tuning
        save_path = self.get_local_worker_path(self.step_name, self.worker_id)
        ckpoint_cb = ModelCheckpoint(config=config_ck, directory=save_path)
        loss_cb = LossMonitor(per_print_times=1)
        eval_cb = EvalCallBack(self.ms_model, self.valid_loader, self.dataset_sink_mode, self)
        callback_list = [ckpoint_cb, loss_cb] if self.config.mixup else [ckpoint_cb, loss_cb, eval_cb]
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

    def _init_distributed_setting(self):
        if not self.distributed:
            return
        else:
            logging.info("init hccl ...")
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            hccl_init()

    def _init_ms_context(self):
        if hasattr(self.config, "execute_mode"):
            mode = context.PYNATIVE_MODE if self.config.execute_mode == "PYNATIVE_MODE" else context.GRAPH_MODE
        else:
            mode = context.GRAPH_MODE
        if vega.is_npu_device():
            context.set_context(mode=mode, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
        else:
            context.set_context(mode=mode, device_target="CPU")

        self.dataset_sink_mode = True if vega.is_npu_device() else False
