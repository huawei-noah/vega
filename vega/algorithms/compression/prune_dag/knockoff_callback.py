# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""This is KnockoffFeaturesCallback."""
import logging
import torch
from torch.nn.parameter import Parameter

import vega
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module
from vega.trainer.callbacks.callback import Callback
from .kncokoff_generator import KFGenerator
from vega.common.general import General


@ClassFactory.register(ClassType.CALLBACK)
class KnockoffFeaturesCallback(Callback):
    """Knock Off features call back."""

    def __init__(self):
        """Initialize ModelCheckpoint callback."""
        super(KnockoffFeaturesCallback, self).__init__()
        self.priority = 210

    def init_trainer(self, logs=None):
        """Be called before train."""
        if not vega.is_torch_backend():
            return
        logging.info("Start Kf scale training.")
        first_conv = True
        for name, module in self.trainer.model.named_modules():
            if not isinstance(module, torch.nn.Conv2d):
                continue
            if first_conv and module.kernel_size == (7, 7):
                first_conv = False
            else:
                change_module(self.trainer.model, name, KfConv2d(module))

        setattr(self.trainer.model, "generator", KFGenerator(self.trainer.config.generator_model_file))
        for name, params in self.trainer.model.named_parameters():
            if "kf_scale" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        self.trainer.model.cuda()

    def before_train(self, logs=None):
        """Run before train."""
        if isinstance(self.trainer.model, torch.nn.DataParallel):
            self.trainer.model.module.generator.eval()
        else:
            self.trainer.model.generator.eval()

        def generate_input_hook(module, inputs):
            """Define hook function to generate dataset."""
            data = inputs[0]
            if isinstance(module, torch.nn.DataParallel):
                input_list = []
                kf_input = module.module.generator(data)
                ngpu = General.devices_per_trainer
                num_pgpu = data.shape[0] // ngpu
                for igpu in range(ngpu):
                    input_list.append(torch.cat([data[igpu * num_pgpu: (igpu + 1) * num_pgpu],
                                                 kf_input[igpu * num_pgpu:(igpu + 1) * num_pgpu]], dim=0))
                return torch.cat(input_list, dim=0)
            return torch.cat((data, module.generator(data)), dim=0)

        def split_kf_output_hook(module, inputs, result):
            """Define hook function to split output."""
            if isinstance(module, torch.nn.DataParallel):
                output_list = []
                ngpu = General.devices_per_trainer
                num_pgpu = result.shape[0] // ngpu
                for igpu in range(ngpu):
                    output_list.append(result[igpu * num_pgpu * 2: igpu * num_pgpu * 2 + num_pgpu])
                return torch.cat(output_list, dim=0)
            return result[: result.size(0) // 2, :]

        self.trainer.model.register_forward_pre_hook(generate_input_hook)
        self.trainer.model.register_forward_hook(split_kf_output_hook)

    def after_train_step(self, batch_index, logs=None):
        """Clamp kf scale."""
        for module in self.trainer.model.modules():
            if isinstance(module, KfConv2d):
                module.kf_scale.data.clamp_(min=0, max=1)

    def after_train(self, logs=None):
        """Save kf scale model."""
        self._save_model()

    def _save_model(self):
        if vega.is_torch_backend():
            state_dict = {k: v for k, v in self.trainer.model.state_dict().items() if "kf_scale" in k}
            torch.save(state_dict, self.trainer.weights_file)


def change_module(model, name, entity):
    """Chane modules."""
    if not entity:
        return
    tokens = name.split('.')
    attr_name = tokens[-1]
    parent_names = tokens[:-1]
    for s in parent_names:
        model = getattr(model, s)
    setattr(model, attr_name, entity)


@ClassFactory.register(ClassType.NETWORK)
class KfConv2d(Module):
    """Knock off Conv2d."""

    def __init__(self, org_cong):
        super(KfConv2d, self).__init__()
        self.conv = org_cong
        self.kf_scale = Parameter(torch.ones(1, org_cong.out_channels, 1, 1).cuda())
        self.kf_scale.data.fill_(0.5)

    def forward(self, x):
        """Call forward functions."""
        x = self.conv(x)
        if self.training:
            num_ori = int(x.shape[0] // 2)
            x = torch.cat([self.kf_scale * x[:num_ori] + (1 - self.kf_scale) * x[num_ori:], x[num_ori:]], dim=0)
        return x
