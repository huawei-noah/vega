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
import zeus
from zeus.common import ClassFactory, ClassType
from ..config_bakcend_map import ConfigBackendMapping
from ..conf.optim import OptimConfig, OptimMappingDict


if zeus.is_gpu_device():
    try:
        if zeus.is_torch_backend():
            import horovod.torch as hvd
        elif zeus.is_tf_backend():
            import horovod.tensorflow as hvd
    except Exception:
        pass
elif zeus.is_npu_device() and zeus.is_tf_backend():
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

if zeus.is_tf_backend():
    from zeus.trainer.modules.optimizer.optimizer import dynamic_optimizer, dynamic_distributed_optimizer


class Optimizer(object):
    """Register and call Optimizer class."""

    config = OptimConfig()

    def __init__(self):
        """Initialize."""
        # register pytorch/tensorflow optim as default
        raw_config = self.config.to_json()
        raw_config.type = self.config.type
        map_dict = OptimMappingDict
        self.map_config = ConfigBackendMapping(
            map_dict.type_mapping_dict, map_dict.params_mapping_dict).backend_mapping(raw_config)
        self.optim_cls = ClassFactory.get_cls(ClassType.OPTIMIZER, self.map_config.type)

    def __call__(self, model=None, distributed=False):
        """Call Optimizer class.

        :param model: model, used in torch case
        :param distributed: use distributed
        :return: optimizer
        """
        params = self.map_config.get("params", {})
        logging.debug("Call Optimizer. name={}, params={}".format(self.optim_cls.__name__, params))
        optimizer = None
        try:
            if zeus.is_torch_backend():
                learnable_params = [param for param in model.parameters() if param.requires_grad]
                optimizer = self.optim_cls(learnable_params, **params)
                if distributed:
                    optimizer = self.set_distributed(optimizer, model)
            elif zeus.is_tf_backend():
                optimizer = dynamic_optimizer(self.optim_cls, **params)
            elif zeus.is_ms_backend():
                learnable_params = [param for param in model.trainable_params() if param.requires_grad]
                optimizer = self.optim_cls(learnable_params, **params)
            return optimizer
        except Exception as ex:
            logging.error("Failed to call Optimizer name={}, params={}".format(self.optim_cls.__name__, params))
            raise ex

    @classmethod
    def set_distributed(cls, optimizer, model=None):
        """Set distributed optimizer."""
        if zeus.is_torch_backend():
            optimizer = hvd.DistributedOptimizer(optimizer,
                                                 named_parameters=model.named_parameters(),
                                                 compression=hvd.Compression.none)
        elif zeus.is_tf_backend():
            optim_class = hvd.DistributedOptimizer if zeus.is_gpu_device() else NPUDistributedOptimizer
            optimizer = dynamic_distributed_optimizer(optim_class, optimizer)
        return optimizer


if zeus.is_torch_backend():
    import torch.optim as torch_opt

    ClassFactory.register_from_package(torch_opt, ClassType.OPTIMIZER)
elif zeus.is_tf_backend():
    import tensorflow.compat.v1.train as tf_train

    ClassFactory.register_from_package(tf_train, ClassType.OPTIMIZER)

elif zeus.is_ms_backend():
    import mindspore.nn.optim as ms_opt

    ClassFactory.register_from_package(ms_opt, ClassType.OPTIMIZER)
