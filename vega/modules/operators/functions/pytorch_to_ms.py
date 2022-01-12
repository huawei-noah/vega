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

"""Convert pytorch weight to mindspore checkpoint."""

import os
import uuid
import logging
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor


def pytorch2mindspore(pth_file):
    """Convert pytorch weight to mindspore checkpoint."""
    torch_para_dict = torch.load(pth_file)
    (torch_paras_name_list, torch_weight_list, ms_para_name_list) = _convert_weight_name(torch_para_dict)
    ms_params_list = _convert_weight_format(torch_paras_name_list, torch_weight_list, ms_para_name_list)
    save_path = os.path.dirname(pth_file)
    save_file_name = os.path.join(save_path, "torch2ms_" + uuid.uuid1().hex[:8] + ".ckpt")
    save_checkpoint(ms_params_list, save_file_name)
    return save_file_name


def _convert_weight_name(torch_para_dict):
    torch_paras_name_list = []
    torch_weight_list = []
    ms_para_name_list = []
    for index, name in enumerate(torch_para_dict):
        torch_paras_name_list.append(name)
        torch_weight = torch_para_dict[name]
        if name.endswith("weight"):
            name = name[:name.rfind("weight")]
            ms_name = "backbone." + name + "conv2d.weight"
        elif name.endswith('bias'):
            name = name[:name.rfind('bias')]
            ms_name = "backbone." + name + 'batch_norm.beta'
        elif name.endswith('.running_mean'):
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
    return torch_paras_name_list, torch_weight_list, ms_para_name_list


def _convert_weight_format(torch_paras_name_list, torch_weight_list, ms_para_name_list):
    ms_params_list = []
    for index, name in enumerate(ms_para_name_list):
        logging.debug('==> py_name: {}'.format(torch_paras_name_list[index]))
        logging.debug('==> ms_name: {}'.format(name))
        param_dict = {}
        param_dict['name'] = name
        parameter = torch_weight_list[index]
        param_dict['data'] = Tensor(parameter.detach().numpy())
        ms_params_list.append(param_dict)
    return ms_params_list


def pytorch2mindspore_extend(pth_file, model):
    """Convert torchvison  weight to vega weight of ms."""
    init_para_dict = torch.load(pth_file)
    init_names_list = []
    init_weights_list = []
    for index, name in enumerate(init_para_dict):
        init_names_list.append(name)
        init_weights_list.append(init_para_dict[name])

    vega_names_list = []

    for name in model.parameters_dict():
        if not name.endswith("num_batches_tracked"):
            vega_names_list.append(name)
    valid_names_list, vega_weights_list = _get_name_weight(vega_names_list, init_names_list, init_weights_list)
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


def _get_name_weight(vega_names_list, init_names_list, init_weights_list):
    """Get name and weight from torch."""
    vega_weights_list = []
    valid_names_list = []
    for index, name in enumerate(vega_names_list):
        init_name = init_names_list[index]
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
    return valid_names_list, vega_weights_list
