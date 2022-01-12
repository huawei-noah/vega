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

"""Manage LrScheduler class."""

from types import MethodType
import logging
import vega
from vega.common import ClassFactory, ClassType
from vega.common.config import Config
from vega.common.general import General
from ..config_bakcend_map import ConfigBackendMapping
from ..conf.optim import OptimConfig, OptimMappingDict


class Optimizer(object):
    """Register and call Optimizer class."""

    config = OptimConfig()

    def __new__(cls, *args, **kwargs):
        """Create optimizer or multi-optimizer class."""
        if isinstance(cls.config.to_dict, list):
            t_cls = ClassFactory.get_cls(ClassType.OPTIMIZER, 'MultiOptimizers')
            return super().__new__(t_cls)
        return super().__new__(cls)

    def __init__(self, config=None):
        """Initialize."""
        self.is_multi_opt = False
        if config is not None:
            self.config = Config(config)
        raw_config = self.config.to_dict()
        raw_config.type = self.config.type
        map_dict = OptimMappingDict
        self.map_config = ConfigBackendMapping(
            map_dict.type_mapping_dict, map_dict.params_mapping_dict).backend_mapping(raw_config)
        self.optim_cls = ClassFactory.get_cls(ClassType.OPTIMIZER, self.map_config.type)

    def __call__(self, model=None, distributed=False, **kwargs):
        """Call Optimizer class.

        :param model: model, used in torch case
        :param distributed: use distributed
        :return: optimizer
        """
        params = self.map_config.get("params", {})
        logging.debug("Call Optimizer. name={}, params={}".format(self.optim_cls.__name__, params))
        optimizer = None
        try:
            if vega.is_torch_backend():
                learnable_params = [param for param in model.parameters() if param.requires_grad]
                optimizer = self.optim_cls(learnable_params, **params)
                if distributed:
                    optimizer = self.set_distributed(optimizer, model)
            elif vega.is_tf_backend():
                from vega.trainer.modules.optimizer.optimizer import dynamic_optimizer
                optimizer = dynamic_optimizer(self.optim_cls, **params)
            elif vega.is_ms_backend():
                if "dynamic_lr" in kwargs:
                    params.update({"learning_rate": kwargs["dynamic_lr"]})
                learnable_params = [param for param in model.trainable_params() if param.requires_grad]
                if 'no_decay_params' in kwargs and len(kwargs['no_decay_params']) > 0:
                    logging.info(f"no_decay_params is {kwargs['no_decay_params']}.")
                    decayed_params = []
                    no_decayed_params = []
                    for param in learnable_params:
                        decay_flag = True
                        for no_decay in kwargs['no_decay_params']:
                            if no_decay in param.name:
                                no_decayed_params.append(param)
                                decay_flag = False
                                break
                        if decay_flag:
                            decayed_params.append(param)

                    learnable_params = [{'params': decayed_params, 'weight_decay': params['weight_decay']},
                                        {'params': no_decayed_params},
                                        {'order_params': model.trainable_params()}]
                if 'no_decay_params' in params:
                    params.pop('no_decay_params')
                optimizer = self.optim_cls(learnable_params, **params)
            return optimizer
        except Exception as ex:
            logging.error("Failed to call Optimizer name={}, params={}".format(self.optim_cls.__name__, params))
            raise ex

    @classmethod
    def set_distributed(cls, optimizer, model=None):
        """Set distributed optimizer."""
        if General.cluster.horovod and vega.is_torch_backend():
            import horovod.torch as hvd
            optimizer = hvd.DistributedOptimizer(optimizer,
                                                 named_parameters=model.named_parameters(),
                                                 compression=hvd.Compression.none)
        elif General.cluster.horovod and vega.is_tf_backend():
            import horovod.tensorflow as hvd
            from vega.trainer.modules.optimizer.optimizer import OptimizerStep
            base_lr = optimizer.base_lr
            weight_decay = optimizer.weight_decay
            optimizer = hvd.DistributedOptimizer(optimizer)
            setattr(optimizer, "base_lr", base_lr)
            setattr(optimizer, "weight_decay", weight_decay)
            optimizer.step = MethodType(OptimizerStep.step, optimizer)
            optimizer.set_lr = MethodType(OptimizerStep.set_lr, optimizer)
            optimizer.regularize_loss = MethodType(OptimizerStep.regularize_loss, optimizer)
        elif General.cluster.hccl and vega.is_tf_backend():
            from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
            from vega.trainer.modules.optimizer.optimizer import dynamic_distributed_optimizer
            optimizer = dynamic_distributed_optimizer(NPUDistributedOptimizer, optimizer)
        return optimizer


if vega.is_torch_backend():
    import torch.optim as torch_opt

    ClassFactory.register_from_package(torch_opt, ClassType.OPTIMIZER)
    if vega.is_npu_device():
        try:
            from apex.optimizers import NpuFusedSGD

            ClassFactory.register_cls(NpuFusedSGD, ClassType.OPTIMIZER)
        except Exception:
            logging.debug('apex of NPU is not installed.')
elif vega.is_tf_backend():
    import tensorflow.compat.v1.train as tf_train

    ClassFactory.register_from_package(tf_train, ClassType.OPTIMIZER)

elif vega.is_ms_backend():
    import mindspore.nn.optim as ms_opt

    ClassFactory.register_from_package(ms_opt, ClassType.OPTIMIZER)
