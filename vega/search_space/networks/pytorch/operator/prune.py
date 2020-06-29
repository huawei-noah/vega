# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Prune operators."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vega.search_space.networks.net_utils import NetTypes
from torch.autograd import Function
from vega.search_space.networks.network_factory import NetworkFactory


def load_model(local_pth, model):
    """Load model."""
    checkpoint = torch.load(local_pth)
    model.load_state_dict(checkpoint, False)
    return model


def handle_layer(model, chn_node_mask, chn_mask, start_mask, end_mask, chn_node_id, chn_id):
    """Handle every layer."""
    network_name = ['0.0.0', '0.0.1', '1.0.0', '2.0.0', '3.0.0', '3.0.1', '4.0.0',
                    '5.0.0', '6.0.0', '6.0.1', '7.0.0', '8.0.0']
    remove_shortcut = ['0.0.0', '1.0.0', '2.0.0', '3.0.0', '4.0.0', '5.0.0',
                       '6.0.0', '7.0.0', '8.0.0']
    for c_name, test in model.named_modules():
        if c_name in network_name:
            conv_id = 0
            for layer1 in test.modules():
                if isinstance(layer1, nn.Conv2d):
                    if conv_id == 0:
                        start_mask = chn_node_mask[chn_node_id - 1]
                        end_mask = chn_mask[chn_id]
                        chn_id += 1
                    if conv_id == 1:
                        start_mask = end_mask
                        end_mask = chn_node_mask[chn_node_id]
                    # shortcut
                    if conv_id == 2:
                        start_mask = chn_node_mask[chn_node_id - 1]
                        end_mask = chn_node_mask[chn_node_id]
                    start_mask = np.asarray(start_mask)
                    end_mask = np.asarray(end_mask)
                    idx0 = np.squeeze(np.argwhere(
                        np.asarray(np.ones(start_mask.shape) - start_mask)))
                    idx1 = np.squeeze(np.argwhere(
                        np.asarray(np.ones(end_mask.shape) - end_mask)))
                    mask = np.ones(layer1.weight.data.shape)
                    mask[:, idx0.tolist(), :, :] = 0
                    mask[idx1.tolist(), :, :, :] = 0
                    layer1.weight.data = layer1.weight.data * \
                        torch.FloatTensor(mask)
                    layer1.weight.data[:, idx0.tolist(), :, :].requires_grad = False
                    layer1.weight.data[idx1.tolist(), :, :, :].requires_grad = False
                    conv_id += 1
                    continue
                if isinstance(layer1, nn.BatchNorm2d):
                    idx1 = np.squeeze(np.argwhere(
                        np.asarray(np.ones(end_mask.shape) - end_mask)))
                    mask = np.ones(layer1.weight.data.shape)
                    mask[idx1.tolist()] = 0
                    layer1.weight.data = layer1.weight.data * \
                        torch.FloatTensor(mask)
                    layer1.bias.data = layer1.bias.data * \
                        torch.FloatTensor(mask)
                    layer1.running_mean = layer1.running_mean * \
                        torch.FloatTensor(mask)
                    layer1.running_var = layer1.running_var * \
                        torch.FloatTensor(mask)
                    layer1.weight.data[idx1.tolist()].requires_grad = False
                    layer1.bias.data[idx1.tolist()].requires_grad = False
                    layer1.running_mean[idx1.tolist()].requires_grad = False
                    layer1.running_var[idx1.tolist()].requires_grad = False
            # every BasicBlock +1
            if c_name in remove_shortcut:
                chn_node_id += 1


@NetworkFactory.register(NetTypes.Operator)
def generate_init_model(local_pth, model, chn_node_mask, chn_mask):
    """Generate init model by loading pretrained model.

    :param model_prune: searched pruned model
    :type model_prune: torch.nn.Module
    :return: initial model after loading pretrained model
    :rtype: torch.nn.Module
    """
    model_init = load_model(local_pth, model)
    chn_node_id = 0
    chn_id = 0
    start_mask = []
    end_mask = []
    for name, m1 in model_init.named_modules():
        if name == '0':
            for layer in m1.modules():
                if isinstance(layer, nn.Conv2d):
                    end_mask = chn_node_mask[chn_node_id]
                    end_mask = np.asarray(end_mask)
                    idx1 = np.squeeze(np.argwhere(np.asarray(np.ones(end_mask.shape) - end_mask)))
                    mask = np.ones(layer.weight.data.shape)
                    mask[idx1.tolist(), :, :, :] = 0
                    layer.weight.data = layer.weight.data * torch.FloatTensor(mask)
                    layer.weight.data[idx1.tolist(), :, :, :].requires_grad = False
                    chn_node_id += 1
                    continue
                if isinstance(layer, nn.BatchNorm2d):
                    idx1 = np.squeeze(np.argwhere(np.asarray(np.ones(end_mask.shape) - end_mask)))
                    mask = np.ones(layer.weight.data.shape)
                    mask[idx1.tolist()] = 0
                    layer.weight.data = layer.weight.data * torch.FloatTensor(mask)
                    layer.bias.data = layer.bias.data * torch.FloatTensor(mask)
                    layer.running_mean = layer.running_mean * torch.FloatTensor(mask)
                    layer.running_var = layer.running_var * torch.FloatTensor(mask)
                    layer.weight.data[idx1.tolist()].requires_grad = False
                    layer.bias.data[idx1.tolist()].requires_grad = False
                    layer.running_mean[idx1.tolist()].requires_grad = False
                    layer.running_var[idx1.tolist()].requires_grad = False
                    continue
        elif name == '1':
            handle_layer(m1, chn_node_mask, chn_mask, start_mask, end_mask, chn_node_id, chn_id)
        elif name == '4':
            if isinstance(m1, nn.Linear):
                idx1 = np.squeeze(np.argwhere(
                    np.asarray(np.ones(end_mask.shape) - end_mask)))
                mask = np.ones(m1.weight.data.shape)
                mask[:, idx1.tolist()] = 0
                m1.weight.data = m1.weight.data * torch.FloatTensor(mask)
                m1.weight.data[:, idx1.tolist()].requires_grad = False
    return model_init
