# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Manage LrScheduler class."""
import logging
import vega
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.config import obj2config
from ...conf import OptimConfig

if vega.is_gpu_device():
    try:
        if vega.is_torch_backend():
            import horovod.torch as hvd
        elif vega.is_tf_backend():
            import horovod.tensorflow as hvd
    except Exception:
        pass
elif vega.is_npu_device():
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer


class Optimizer(object):
    """Register and call Optimizer class."""

    config = OptimConfig()

    def __init__(self):
        """Initialize."""
        # register pytorch/tensorflow optim as default
        optim_name = self.config.type
        self.optim_cls = ClassFactory.get_cls(ClassType.OPTIM, optim_name)

    def __call__(self, model=None, lr_scheduler=None, epoch=None, distributed=False):
        """Call Optimizer class.

        :param model: model, used in torch case
        :param lr_scheduler: learning rate scheduler, used in tf case
        :param epoch: epoch of training, used in tf case
        :param distributed: use distributed
        :return: optimizer
        """
        params = obj2config(self.config).get("params", {})
        logging.debug("Call Optimizer. name={}, params={}".format(self.optim_cls.__name__, params))
        optimizer = None
        try:
            if vega.is_torch_backend():
                learnable_params = [param for param in model.parameters() if param.requires_grad]
                optimizer = self.optim_cls(learnable_params, **params)
                if distributed:
                    optimizer = hvd.DistributedOptimizer(optimizer,
                                                         named_parameters=model.named_parameters(),
                                                         compression=hvd.Compression.none)
            elif vega.is_tf_backend():
                lr_scheduler.step(epoch)
                params['learning_rate'] = lr_scheduler.get_lr()[0]
                optimizer = self.optim_cls(**params)
                if distributed:
                    optimizer = hvd.DistributedOptimizer(optimizer) if vega.is_gpu_device() else \
                        NPUDistributedOptimizer(optimizer)
            return optimizer
        except Exception as ex:
            logging.error("Failed to call Optimizer name={}, params={}".format(self.optim_cls.__name__, params))
            raise ex


if vega.is_torch_backend():
    import torch.optim as torch_opt

    ClassFactory.register_from_package(torch_opt, ClassType.OPTIM)
elif vega.is_tf_backend():
    import tensorflow.train as tf_train

    ClassFactory.register_from_package(tf_train, ClassType.OPTIM)
