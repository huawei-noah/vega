# -*- coding:utf-8 -*-

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

"""Some function need in spnet.py."""

import os.path as osp
from collections import OrderedDict
import torch.nn as nn
import torch


def dirac_init(module, bias=0):
    """Load state_dict to a module. This method is modified from :meth:`torch.nn.Module.load_state_dict`.

    :param module: Module that receives the state_dict
    :type module: OrderedDict
    :param bias: bias
    :type bias: int
    """
    nn.init.dirac_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def load_state_dict(module, state_dict, mapping, logger=None, mb_mapping=None):
    """Load state_dict to a module. This method is modified from :meth:`torch.nn.Module.load_state_dict`.

    :param module: Module that receives the state_dict
    :type module: OrderedDict
    :param state_dict: Weights
    :type state_dict: Module
    :param mapping: serail backbone with related pretrained weight name
    :type mapping: dict
    :param logger: Logger to log the error
    :type logger: logging.Logger
    :param mb_mapping: parallel backbone with related serial backbone
    :type mb_mapping: dict
    """
    unexpected_keys = []
    shape_mismatch_pairs = []
    mb_keys = []

    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name.find('mb') >= 0:
            continue
        if name not in own_state and name not in mapping.keys():
            unexpected_keys.append(name)
            continue
        if name in mapping.keys():
            if mapping[name] not in own_state:
                unexpected_keys.append(name)
                continue
            name = mapping[name]
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != own_state[name].size():
            shape_mismatch_pairs.append(
                [name, own_state[name].size(),
                 param.size()])
            continue
        own_state[name].copy_(param)
        if mb_mapping is not None and name in mb_mapping.keys():
            for mb_name in mb_mapping[name]:
                own_state[mb_name].copy_(param)
                mb_keys.append(mb_name)


def load_checkpoint(model,
                    filename,
                    pretrain_to_own,
                    map_location='cpu',
                    logger=None,
                    mb_mapping=None):
    """Load checkpoint from a pretrained checkpoint file.

    :param module: Module that receives the state_dict
    :type module: OrderedDict
    :param filename: A filepath.
    :type filename: str
    :param pretrain_to_own: serail backbone with related pretrained weight name
    :type pretrain_to_own: dict
    :param logger: Logger to log the error
    :type logger: logging.Logger
    :param mb_mapping: parallel backbone with related serial backbone
    :type mb_mapping: dict
    :param map_location: map location
    :type map_location: str
    :return: checkpoint
    :rtype: checkpoint
    """
    if not osp.isfile(filename):
        raise IOError('{} is not a checkpoint file'.format(filename))
    checkpoint = torch.load(filename, map_location=map_location)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'weight' in checkpoint:
        state_dict = checkpoint['weight']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict,
                        pretrain_to_own, logger, mb_mapping)
    else:
        load_state_dict(model, state_dict, pretrain_to_own, logger, mb_mapping)
    return checkpoint


def remove_layers(layer_arch, p_arch, len_dis):
    """Remove expand layers in serailnet than pretrained architecture.

    :param layer_arch: current serailnet encode
    :type layer_arch: str
    :param p_arch: pretrained serailnet encode
    :type p_arch: str
    :param len_dis: num of blocks more than pretrain
    :type len_dis: int
    :return: more expand block name
    :rtype: list
    """

    def map_index_to_name(index, layer_arch):
        name = []
        for i in index:
            for layer in range(1, len(layer_arch) + 1):
                if i < len(''.join(layer_arch[:layer])):
                    block_i = i - len(''.join(layer_arch[:layer - 1]))
                    n = 'layer' + str(layer) + '.' + str(block_i) + '.'
                    name.append(n)
                    break
        return name

    def find_diff_index(pos, new_arch, old_arch):
        agg_arch = zip(new_arch, old_arch)
        for i, tup in enumerate(agg_arch):
            if tup[0] != tup[1]:
                if len(pos) < 1:
                    pos.append(i)
                else:
                    pos.append(i + pos[-1] + 1)
                return find_diff_index(pos, new_arch[i + 1:], old_arch[i:])
                break
        return pos

    def get_bn_layers(remove_name, layer_arch, p_arch):
        bn_layers = [n + 'bn{}'.format(i)
                     for n in remove_name for i in range(1, 4)]
        p_len = len(p_arch.split('-'))
        for i in range(1, len(layer_arch) - p_len + 1):
            bn_layers.append('layer{}.0.downsample.1'.format(p_len + i))
        return bn_layers

    pretrained_arch = ''.join(p_arch.split('-'))
    remove_index = find_diff_index([], ''.join(layer_arch), pretrained_arch)
    if len(remove_index) < len_dis:
        if len(layer_arch) > len(p_arch.split('-')):
            len_dis = len_dis - (len(layer_arch) - len(p_arch.split('-')))
        remove_index = remove_index + list(range(len(''.join(layer_arch)) - len_dis + len(remove_index),
                                                 len(''.join(layer_arch))))
    remove_name = map_index_to_name(remove_index, layer_arch)
    bn_layers = get_bn_layers(remove_name, layer_arch, p_arch)
    return remove_name, bn_layers


def match_name(own_names, checkpoint_names):
    """Generate the mapping dict between current serialnet layer and pretrain.

    :param own_names: keys of current serialnet
    :type own_names: list
    :param checkpoint_names: keys of pretrained serailnet
    :type checkpoint_names: list
    :return: serail backbone with related pretrained weight name
    :rtype: dict
    """
    pretrain_to_own = dict()
    print("matching pretrained model with new distributed architecture....")
    bid_to_own = []
    for name in own_names:
        if name.find('layer') >= 0:
            block_name = '.'.join(name.split('.')[:2])
            if block_name not in bid_to_own:
                bid_to_own.append(block_name)
    flag = ''
    bid = -1
    for name in checkpoint_names:
        name_copy = name
        if 'backbone' in name_copy:
            name_copy = name_copy.replace('backbone.', '')
            pretrain_to_own[name] = name_copy
        if name_copy.find('layer') >= 0:
            temp = name_copy.split('.')
            if temp[0] != flag:
                s = bid + 1
            flag = temp[0]
            bid = s + int(temp[1])
            pretrain_to_own[name] = '.'.join([bid_to_own[bid]] + temp[2:])

    return pretrain_to_own
