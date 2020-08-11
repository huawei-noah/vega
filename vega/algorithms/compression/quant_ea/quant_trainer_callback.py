# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""TrainWorker for searching quantization model."""
import logging
import copy
import vega
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.callbacks import Callback
from vega.core.metrics import calc_model_flops_params

if vega.is_torch_backend():
    import torch
    from .utils.pytorch.quant_model import Quantizer
    from .utils.pytorch.quant_conv import quant_custom_ops
elif vega.is_tf_backend():
    import tensorflow as tf
    from .utils.tensorflow.quant_model import Quantizer


@ClassFactory.register(ClassType.CALLBACK)
class QuantTrainerCallback(Callback):
    """Callback class for Quant Trainer."""

    disable_callbacks = ["ModelStatistics"]

    def __init__(self):
        super(Callback, self).__init__()
        self.flops_count = None
        self.params_count = None

    def before_train(self, logs=None):
        """Be called before the train process."""
        self.config = self.trainer.config
        self.device = self.trainer.config.device
        self.model_code = copy.deepcopy(self.trainer.config.codec)
        if vega.is_torch_backend():
            self.trainer.model._init_weights()
            self.trainer.model = self._quantize_model(self.trainer.model).to(self.device)
            input = torch.randn(1, 3, 32, 32).to(self.device)
            self.flops_count, self.params_count = calc_model_flops_params(
                self.trainer.model, input, quant_custom_ops())
        elif vega.is_tf_backend():
            tf.reset_default_graph()
            self.trainer.model = self._quantize_model(self.trainer.model)
            quant_info = copy.deepcopy(self.trainer.model.quant_info)
            input = tf.random_uniform([1, 32, 32, 3], dtype=tf.float32)
            self.flops_count, self.params_count = calc_model_flops_params(self.trainer.model, input)
            self.flops_count += quant_info['extra_flops']
            self.params_count += quant_info['extra_params']
        self.validate()

    def after_epoch(self, epoch, logs=None):
        """Update gflops and kparams."""
        summary_perfs = logs.get('summary_perfs', {})
        summary_perfs.update({'gflops': self.flops_count, 'kparams': self.params_count})
        logs.update({'summary_perfs': summary_perfs})

    def _quantize_model(self, model):
        """Quantize the model.

        :param model: pytorch model
        :type model: nn.Module
        :return: quantized pytorch model
        :rtype: nn.Module
        """
        q = Quantizer()
        if self.model_code is not None:
            nbit_w_list = self.model_code.nbit_w_list
            nbit_a_list = self.model_code.nbit_a_list
        else:
            nbit_w_list = model.nbit_w_list
            nbit_a_list = model.nbit_a_list
        logging.info('current code: %s, %s', nbit_w_list, nbit_a_list)
        model = q.quant_model(model, nbit_w_list, nbit_a_list)
        return model

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
