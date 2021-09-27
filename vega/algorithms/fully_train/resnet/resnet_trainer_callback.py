# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Resnet Trainer."""

import os
from mindspore import context
from mindspore import Tensor
from mindspore.train import Model as MsModel
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.parallel import set_algo_parameters
import vega
from vega.trainer.trainer_base import TrainerBase
from vega.common import ClassFactory, ClassType
import logging
from mindspore.communication.management import init as hccl_init
from mindspore.context import ParallelMode
from .src.resnet import resnet50 as resnet
from .src.dataset import create_dataset2 as create_dataset
from .src.CrossEntropySmooth import CrossEntropySmooth
from .src.lr_generator import get_lr
from mindspore.nn.optim import Momentum
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from vega.datasets.conf.dataset import DatasetConfig
from vega.trainer.callbacks.ms_callbacks import EvalCallBack
from vega.common.general import General


def init_weight(net):
    """Initialize weight."""
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))


def init_group_prams(net):
    """Initialize group_prams."""
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': 0.0001},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


@ClassFactory.register(ClassType.TRAINER)
class ResnetTrainer(TrainerBase):
    """Trainer mindspore class."""

    def build(self):
        """Build the trainer by assembling the necessary components."""
        logging.debug("Trainer Config: {}".format(self.config))
        self._init_hps()
        self.do_validation = False
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
        logging.debug("Trainer Config: {}".format(self.config))
        config = DatasetConfig().to_dict()
        self.train_config = config['_class_data'].train
        self.valid_config = config['_class_data'].val
        self.loss = CrossEntropySmooth(sparse=self.config.loss.params.sparse,
                                       reduction=self.config.loss.params.reduction,
                                       smooth_factor=self.config.loss.params.smooth_factor,
                                       num_classes=self.train_config.n_class)
        self.metric_name = self.config.metric.type

        self.train_metrics = None
        self.valid_metrics = self._init_metrics()
        self.ms_metrics = self.valid_metrics() if isinstance(self.valid_metrics(), dict) else {
            self.metric_name: self.valid_metrics()}

        self.net = resnet(class_num=self.train_config.n_class)
        init_weight(net=self.net)
        from mindspore.train.loss_scale_manager import FixedLossScaleManager
        self.loss_scale = FixedLossScaleManager(self.config.loss_scale, drop_overflow_update=False)

    def init_env(self):
        """Construct the trainer of Resnet."""
        super().init_env()
        self._init_ms_context()
        self._init_distributed_setting()

    def _train_epoch(self):
        """Construct the trainer of Resnet."""
        try:
            dataset = create_dataset(dataset_path=self.train_config.data_path + '/train', do_train=True,
                                     repeat_num=1, batch_size=self.train_config.batch_size, target='Ascend',
                                     distribute=True)
            step_size = dataset.get_dataset_size()

            lr = Tensor(
                get_lr(lr_init=self.config.lr_scheduler.params.lr_init, lr_end=self.config.lr_scheduler.params.lr_end,
                       lr_max=self.config.lr_scheduler.params.lr_max,
                       warmup_epochs=0, total_epochs=self.config.epochs, steps_per_epoch=step_size,
                       lr_decay_mode=self.config.lr_scheduler.params.lr_decay_mode))
            group_params = init_group_prams(self.net)
            opt = Momentum(group_params, lr, self.config.optimizer.params.momentum, loss_scale=self.config.loss_scale)

            self.ms_model = MsModel(network=self.net,
                                    loss_fn=self.loss,
                                    optimizer=opt,
                                    loss_scale_manager=self.loss_scale,
                                    amp_level="O2", keep_batchnorm_fp32=False,
                                    acc_level="O0",
                                    metrics=self.ms_metrics)
            config_ck = CheckpointConfig(save_checkpoint_steps=self.config.save_steps, keep_checkpoint_max=1)
            save_path = self.get_local_worker_path(self.step_name, self.worker_id)
            ckpoint_cb = ModelCheckpoint(config=config_ck, directory=save_path)
            loss_cb = LossMonitor()
            self.valid_loader = create_dataset(dataset_path=self.valid_config.data_path + '/val', do_train=False,
                                               batch_size=self.valid_config.batch_size,
                                               target='Ascend')
            eval_cb = EvalCallBack(self.ms_model, self.valid_loader, self.dataset_sink_mode, self)
            callback_list = [ckpoint_cb, loss_cb, eval_cb]

            self.ms_model.train(epoch=self.epochs,
                                train_dataset=dataset,
                                callbacks=callback_list,
                                dataset_sink_mode=False)
        except RuntimeError as e:
            logging.warning(f"failed to train the model, skip it, message: {str(e)}")

    def _init_distributed_setting(self):
        """Construct the trainer of Resnet."""
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        context.set_auto_parallel_context(all_reduce_fusion_config=self.config.all_reduce_fusion_config)

    def _init_ms_context(self):
        mode = General.ms_execute_mode
        logging.info(f"Run train/val in mode: {mode}.")
        if vega.is_npu_device():
            context.set_context(mode=mode, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
        else:
            context.set_context(mode=mode, device_target="CPU")

        self.dataset_sink_mode = General.dataset_sink_mode
        logging.info(f"Dataset_sink_mode:{self.dataset_sink_mode}.")
