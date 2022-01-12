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

"""TrainWorker for searching quantization model."""
import logging
import copy
import vega
from vega.common import ClassFactory, ClassType, General
from vega.trainer.callbacks import Callback
from vega.metrics import calc_model_flops_params, calc_forward_latency
from vega.networks.quant import Quantizer
from vega.trainer.modules.lr_schedulers import LrScheduler
from vega.trainer.modules.optimizer import Optimizer

if vega.is_torch_backend():
    import torch
elif vega.is_tf_backend():
    import tensorflow as tf


@ClassFactory.register(ClassType.CALLBACK)
class QuantTrainerCallback(Callback):
    """Callback class for Quant Trainer."""

    disable_callbacks = ["ModelStatistics"]

    def __init__(self):
        super(Callback, self).__init__()
        self.flops_count = None
        self.params_count = None
        self.latency_count = None

    def before_train(self, logs=None):
        """Be called before the train process."""
        self.config = self.trainer.config
        model_code = copy.deepcopy(self.trainer.model.desc)
        model = self.trainer.model
        logging.info('current code: %s, %s', model_code.nbit_w_list, model_code.nbit_a_list)
        quantizer = Quantizer(model, model_code.nbit_w_list, model_code.nbit_a_list)
        model = quantizer()
        self.trainer.model = model
        count_input = [1, 3, 32, 32]
        if General.data_format == 'channels_last':
            count_input = [1, 32, 32, 3]
        sess_config = None
        if vega.is_torch_backend():
            if vega.is_gpu_device():
                model = model.cuda()
                count_input = torch.FloatTensor(*count_input).cuda()
            elif vega.is_npu_device():
                model = model.to(vega.get_devices())
                count_input = torch.FloatTensor(*count_input).to(vega.get_devices())
            self.trainer.optimizer = Optimizer()(model=self.trainer.model, distributed=self.trainer.horovod)
            self.trainer.lr_scheduler = LrScheduler()(self.trainer.optimizer)
        elif vega.is_tf_backend():
            tf.compat.v1.reset_default_graph()
            count_input = tf.random.uniform(count_input, dtype=tf.float32)
            sess_config = self.trainer._init_session_config()
        self.flops_count, self.params_count = calc_model_flops_params(model, count_input,
                                                                      custom_hooks=quantizer.custom_hooks())
        self.latency_count = calc_forward_latency(model, count_input, sess_config)
        logging.info("after quant model glops=%sM, params=%sK, latency=%sms",
                     self.flops_count * 1e-6, self.params_count * 1e-3, self.latency_count * 1000)
        self.validate()

    def after_epoch(self, epoch, logs=None):
        """Update flops and params."""
        summary_perfs = logs.get('summary_perfs', {})
        summary_perfs.update({'flops': self.flops_count * 1e-6, 'params': self.params_count * 1e-3,
                              'latency': self.latency_count * 1000})
        logs.update({'summary_perfs': summary_perfs})

    def validate(self):
        """Check whether the model fits in the #flops range or #parameter range specified in config.

        :return: true or false, which specifies whether the model fits in the range
        :rtype: bool
        """
        limits_config = self.config.limits or dict()
        if "flop_range" in limits_config:
            flop_range = limits_config["flop_range"]
            if self.flops_count < flop_range[0] or self.flops_count > flop_range[1]:
                raise ValueError("flops count exceed limits range.")
        if "param_range" in limits_config:
            param_range = limits_config["param_range"]
            if self.params_count < param_range[0] or self.params_count > param_range[1]:
                raise ValueError("params count exceed limits range.")
