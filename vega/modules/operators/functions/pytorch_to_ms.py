# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Convert pytorch weight to mindspore checkpoint."""
import os
import torch
import logging
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor
import uuid


def pytorch2mindspore(pth_file):
    """Convert pytorch weight to mindspore checkpoint."""
    torch_para_dict = torch.load(pth_file)
    torch_weight_list = []
    torch_paras_name_list = []
    ms_params_list = []
    ms_para_name_list = []

    for index, name in enumerate(torch_para_dict):
        torch_paras_name_list.append(name)
        torch_weight = torch_para_dict[name]

        # if name == "fc.weight":
        # ms_name = "fc.linear.weight"
        # elif name == "fc.bias":
        # ms_name = "fc.linear.bias"
        if name.endswith("weight"):
            name = name[:name.rfind("weight")]
            ms_name = "backbone." + name + "conv2d.weight"
        elif name.endswith('bias'):
            name = name[:name.rfind('bias')]
            ms_name = "backbone." + name + 'batch_norm.beta'
        elif name.endswith('.running_mean'):
            # fix batch_norm name
            old_name_gamma = ms_para_name_list[index - 2]
            new_name_gamma = old_name_gamma[:old_name_gamma.rfind('conv2d.weight')] + "batch_norm.gamma"
            ms_para_name_list[index - 2] = new_name_gamma

            name = name[:name.rfind('.running_mean')]
            ms_name = "backbone." + name + '.batch_norm.moving_mean'

        elif name.endswith('.running_var'):
            name = name[:name.rfind('.running_var')]
            ms_name = "backbone." + name + '.batch_norm.moving_variance'

        elif name.endswith(".num_batches_tracked"):
            ms_name = name

        torch_weight_list.append(torch_weight)
        ms_para_name_list.append(ms_name)

    for index, name in enumerate(ms_para_name_list):
        logging.debug('========================py_name: {}'.format(torch_paras_name_list[index]))
        logging.debug('========================ms_name: {}'.format(name))
        param_dict = {}
        param_dict['name'] = name
        parameter = torch_weight_list[index]
        param_dict['data'] = Tensor(parameter.detach().numpy())
        ms_params_list.append(param_dict)

    save_path = os.path.dirname(pth_file)
    save_file_name = os.path.join(save_path, "torch2ms_" + uuid.uuid1().hex[:8] + ".ckpt")
    save_checkpoint(ms_params_list, save_file_name)
    return save_file_name


def pytorch2mindspore_extend(pth_file, model):
    """Convert torchvison  weight to vega weight of ms."""
    init_para_dict = torch.load(pth_file)
    init_names_list = []
    init_weights_list = []
    for index, name in enumerate(init_para_dict):
        init_names_list.append(name)
        init_weights_list.append(init_para_dict[name])

    vega_names_list = []
    vega_weights_list = []
    valid_names_list = []

    for name in model.parameters_dict():
        if not name.endswith("num_batches_tracked"):
            vega_names_list.append(name)

    for index, name in enumerate(vega_names_list):
        init_name = init_names_list[index]
        # if index < 1:
        #     continue
        if name.endswith("weight") and ("conv" or "downsample" in name or "down_sample" in name) and init_name.endswith(
                "weight") and ("conv" in init_name or "downsample" in init_name or "down_sample" in init_name):
            valid_names_list.append(name)
            vega_weights_list.append(init_weights_list[index])
        elif name.endswith("moving_mean") and init_name.endswith("running_mean"):
            valid_names_list.append(name)
            vega_weights_list.append(init_weights_list[index])
        elif name.endswith("moving_variance") and init_name.endswith(
                "running_var"):
            valid_names_list.append(name)
            vega_weights_list.append(init_weights_list[index])
        elif name.endswith("gamma") and init_name.endswith("weight") and (
                "bn" in init_name or "downsample" in init_name or "down_sample" in init_name):
            valid_names_list.append(name)
            vega_weights_list.append(init_weights_list[index])
        elif name.endswith("beta") and init_name.endswith("bias") and (
                "bn" in init_name or "downsample" in init_name or "down_sample" in init_name):
            valid_names_list.append(name)
            vega_weights_list.append(init_weights_list[index])
        else:
            continue

    ms_params_list = []

    for index, name in enumerate(valid_names_list):
        param_dict = {}
        param_dict['name'] = name
        parameter = vega_weights_list[index]
        param_dict['data'] = Tensor(parameter.detach().numpy())
        ms_params_list.append(param_dict)
    save_path = os.path.dirname(pth_file)
    save_file_name = os.path.join(save_path, "torch2ms_" + uuid.uuid1().hex[:8] + ".ckpt")
    save_checkpoint(ms_params_list, save_file_name)
    return save_file_name
