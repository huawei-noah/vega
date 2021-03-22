# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNetVariant for Detection."""
import logging
import copy
import torch
from zeus.common import ClassType, ClassFactory
from zeus.modules.module import Module
from zeus.model_zoo import ModelZoo
from zeus.modules.operators import PruneConv2DFilter, PruneBatchNormFilter, PruneLinearFilter


@ClassFactory.register(ClassType.NETWORK)
class PruneGetter(Module):
    """Get output layer by layer names and connect into a OrderDict."""

    def __init__(self, model_name=None, pretrained_model_file=None, downsamples=[4, 14, 27, 46],
                 residauls=[7, 10, 17, 20, 23, 30, 33, 36, 39, 42, 49, 52], model=None, in_channels=None,
                 out_channels=None, num_classes=None, **kwargs):
        super(PruneGetter, self).__init__()
        self.model = model
        if model_name is not None:
            model_desc = dict(type=model_name)
            model_desc.update(kwargs)
            self.model = ModelZoo().get_model(model_desc, pretrained_model_file)
        self.num_classes = num_classes
        convs = [module for name, module in self.model.named_modules() if isinstance(module, torch.nn.Conv2d)]
        self.in_channels_size = sum([conv.in_channels for conv in convs])
        self.out_channels_size = sum([conv.out_channels for conv in convs])
        if in_channels and len(in_channels) < self.in_channels_size:
            raise ValueError("in channels mask length should be getter than {}".format(self.in_channels_size))
        if out_channels and len(out_channels) < self.out_channels_size:
            raise ValueError("out channels mask length should be getter than {}".format(self.out_channels_size))
        in_channels_code = self.define_props('in_channels', in_channels)
        out_channels_code = self.define_props('out_channels', out_channels)
        if not in_channels_code and not out_channels_code:
            logging.info("channels_code is null. use 1 as default, this model will not pruned.")
            in_channels_code = [1] * self.in_channels_size
            out_channels_code = [1] * self.out_channels_size
        # TODO check downsample auto
        self.downsamples = downsamples
        self.residauls = residauls
        self.model = self._prune(self.model, in_channels_code, out_channels_code)

    def load_state_dict(self, state_dict, strict=True):
        """Call subclass load_state_dict function."""
        return self.model.load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Call subclass state_dict function."""
        return self.model.state_dict(destination, prefix, keep_vars)

    def _prune(self, model, in_channels_code, out_channels_code):
        logging.info("Start to Prune Model.")
        convs = [module for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
        org_convs = copy.deepcopy(convs)
        batch_norms = [module for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
        pre_end_mask_code = None
        block_start_mask = None
        for idx, conv in enumerate(convs):
            end_mask = None
            if idx == 0:
                start_mask = None
            else:
                pre_out_channels = org_convs[idx - 1].out_channels
                if conv.in_channels == pre_out_channels and pre_end_mask_code:
                    start_mask = pre_end_mask_code
                else:
                    start_mask = in_channels_code[:conv.in_channels]
                # cache downsaple conv mask code
                if not block_start_mask:
                    block_start_mask = start_mask
                if idx in self.downsamples:
                    # downsaple conv
                    start_mask = block_start_mask
                    block_start_mask = None
                    end_mask = pre_end_mask_code
                elif idx in self.residauls:
                    # Identify jump node
                    end_mask = block_start_mask
                    block_start_mask = None
                else:
                    in_channels_code = in_channels_code[conv.in_channels:]
            end_mask = end_mask or out_channels_code[:conv.out_channels]
            out_channels_code = out_channels_code[conv.out_channels:]
            # all code is 0, default 1
            PruneConv2DFilter(conv).filter(end_mask, start_mask)
            batch_norm = batch_norms[idx]
            PruneBatchNormFilter(batch_norm).filter(end_mask)
            pre_end_mask_code = end_mask
        linear = [module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)][-1]
        if self.num_classes:
            linear.out_features = self.num_classes
        PruneLinearFilter(linear).filter(pre_end_mask_code)
        # remove the redundant code and calculate the length
        if in_channels_code:
            self.in_channels_size = self.in_channels_size - len(in_channels_code)
        if out_channels_code:
            self.out_channels_size = self.out_channels_size - len(out_channels_code)
        return model
