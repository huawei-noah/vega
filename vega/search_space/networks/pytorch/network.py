# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Network."""
from abc import ABCMeta
import torch.nn as nn
import torch
import torch.nn.functional as F


class Network(nn.Module, metaclass=ABCMeta):
    """Base class for networks.

    A network refers to a family of models (e.g. ResNet)
    """

    def __init__(self, network_seqs=None, is_freeze=False, condition=None,
                 out_list=None, **kwargs):
        """Init Network."""
        super(Network, self).__init__()
        self.is_freeze = is_freeze
        self.condition = condition
        self.out_list = out_list
        if network_seqs is None:
            return
        for index, seq in enumerate(network_seqs):
            if isinstance(seq, list):
                model = self.add_model(str(index), seq)
            else:
                model = seq
            self.add_module(str(index), model)
        if self.condition == 'quant':
            from vega.search_space.networks.pytorch.operator.quant import QuantizerModel
            for _, m in self.named_children():
                child_model = m
            model = QuantizerModel().quant_model(model=child_model, nbit_w_list=kwargs['nbit_w_list'],
                                                 nbit_a_list=kwargs['nbit_a_list'])
            self.add_module(str(0), model)
        elif self.condition == 'prune':
            from vega.search_space.networks.pytorch.operator.prune import generate_init_model
            for _, m in self.named_children():
                child_model = m
            model = generate_init_model(local_pth=kwargs['path'], model=child_model,
                                        chn_mask=kwargs['chn_mask'], chn_node_mask=kwargs['chn_node_mask'])
            self.add_module(str(0), model)

    def add_model(self, name, seq):
        """Add model into torch modules.

        :param name: model name
        :param seq: model seq
        :return: model
        """
        model = nn.Sequential()
        if not isinstance(seq, list):
            model.add_module(name, seq)
        else:
            for index, item in enumerate(seq):
                model.add_module(name + str(index), item)
        return model

    @property
    def input_shape(self):
        """Get the model input tensor shape."""
        raise NotImplementedError

    @property
    def output_shape(self):
        """Get the model output tensor shape."""
        raise NotImplementedError

    @property
    def model_layers(self):
        """Get the model layers."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        """Build a model from config."""
        raise NotImplementedError

    def train(self, mode=True):
        """Train setting.

        :param mode: if train
        :type mode: bool
        """
        super().train(mode)
        if self.is_freeze and mode:
            # TODO: just freeze BatchNorm?
            for m in self.modules():
                if isinstance(m, nn.batchnorm._BatchNorm):
                    self._freeze(m)

    def _freeze(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, *args, forward_train=True, **kwargs):
        """Call default forward function."""
        if len(args) == 1:
            if forward_train:
                return self.forward_train(*args, **kwargs)
            else:
                return self.forward_valid(*args, **kwargs)
        else:
            if forward_train:
                return self.multi_forward_train(*args, **kwargs)
            else:
                return self.multi_forward_valid(*args, **kwargs)

    def forward_train(self, input, **kwargs):
        """Call train forward function."""
        models = self.children()
        if self.condition == 'add':
            # for add
            output = None
            for model in models:
                if output is None:
                    output = model(input)
                else:
                    output = output + model(input)
        elif self.condition == 'concat':
            # for merge
            outputs = []
            for model in models:
                outputs.append(model(input))
            output = torch.cat(outputs, 1)
        elif self.condition == 'interpolate':
            output = input
            for model in models:
                output = model(output)
            output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)
        elif self.condition == 'append':
            output = self.append(input, models)
        elif self.condition == 'Process_list':
            # adelaide microdecoder output
            output = self.Input_list_forward(input, models, **kwargs)
        elif self.condition == 'merge':
            output = self.merge(input, models)
        elif self.condition == 'map':
            # faster-RCNN outputs (map)
            model = next(models)
            map_results = [model(x) for x in input]
            output = tuple(map_results)
        else:
            # for seq
            output = self.seq_forward(input, models, **kwargs)
        return output

    def append(self, input, models):
        """Append all models."""
        # faster-RCNN outputs (append)
        output = input
        outputs = []
        for model in models:
            output = model(output)
            outputs.append(output)
        return tuple(outputs)

    def merge(self, input, models):
        """Merge all models."""
        output = input
        outputs = []
        for model in models:
            model_out = model(output)
            if isinstance(model_out, tuple):
                model_out = list(model_out)
                outputs.extend(model_out)
            else:
                outputs.append(model_out)
            output = tuple(outputs)
        return output

    def seq_forward(self, input, models, **kwargs):
        """Call sequential train forward function."""
        output = input
        if self.out_list is None:
            for model in models:
                output = model(output)
        else:
            outputs = []
            models = list(models)
            for idx, model in enumerate(models):
                output = model(output)
                if idx in self.out_list:
                    outputs.append(output)
            output = outputs
        return output

    def Input_list_forward(self, input, models, **kwargs):
        """Call list of input train forward function."""
        if self.out_list is None:
            if isinstance(input, list):
                outputs = []
                for model, idx in zip(models, [i for i in range(len(input))]):
                    output = model(input[idx])
                    outputs.append(output)
                output = outputs
            else:
                raise ValueError("Input must list!")
        else:
            input = list(input)
            for model, idx in zip(models, self.out_list):
                if isinstance(idx, list):
                    assert len(idx) == 2
                    output = model(input[idx[0]], input[idx[1]])
                    input.append(output)
                else:
                    input.append(model(input[idx]))
            output = input
        return output

    def forward_valid(self, input, **kwargs):
        """Call test forward function."""
        raise NotImplementedError

    def mutil_forward_train(self, *args, **kwargs):
        """Call mutil input train forward function."""
        models = list(self.children())
        output = []
        for idx in range(len(args)):
            output.append(models[idx](args[idx]))
        output = models[-1](*tuple(output))
        return output

    def multi_forward_valid(self, input, **kwargs):
        """Call test forward function."""
        raise NotImplementedError
