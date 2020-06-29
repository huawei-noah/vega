# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Tools function for SPNAS."""

import torch
from collections import OrderedDict
import os


def extract_backbone_from_pth(local_output_path, pre_worker_id, pretrained):
    """Extract backbone weights from the whole trained checkpoint.

    :param local_output_path: local output path
    :type local_output_path: str
    :param pre_worker_id:pretrained worker id
    :type pre_worker_id: int
    :param pretrained: pretrained serialnet encode.
    :type pretrained: str
    :return: backbone weights pth.
    :rtype: str
    """
    assert isinstance(pre_worker_id, int)
    filename = os.path.join(local_output_path, str(pre_worker_id), pretrained + '.pth')
    assert os.path.exists(filename), "Error pretrained model path {}!".format(filename)
    print("Load backbone pretrained model from {}!".format(filename))
    checkpoint = torch.load(filename)

    f_bb = filename.replace('.pth', '_backbone.pth')
    if not os.path.exists(f_bb):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        state_dict_ = OrderedDict({key.replace('backbone.', ''): value
                                   for key, value in state_dict.items() if 'backbone' in key})
        if len(state_dict_) == 0:
            state_dict_ = state_dict
        torch.save(state_dict_, f_bb)

    return f_bb


def update_config(config, code):
    """Update config dict according to sample info.

    :param config: config dict
    :type config: dict
    :param code: sample info or model encode
    :type code: dict or str
    :return: updated config
    :rtype: dict
    """
    def contr_neck(arch, neck_i=256):

        neck = []
        for l in arch:
            if l == '2':
                neck_i = neck_i << 1
            elif l == '-':
                neck.append(neck_i)
        neck.append(neck_i)
        return neck

    if isinstance(code, dict):
        arch = code['arch'].split('_')[1]
        mb_arch = code['arch'].split('_')[2]
        pretrained_arch = code['pre_arch']
    elif isinstance(code, str):
        arch = code.split('_')[1]
        mb_arch = code.split('_')[2]
        pretrained_arch = arch
    else:
        raise ValueError("Unexpect code type!")

    num_stages = len(arch.split('-'))
    config['model']['backbone']['layers_arch'] = arch
    config['model']['backbone']['mb_arch'] = mb_arch
    config['model']['backbone']['num_stages'] = num_stages
    config['model']['backbone']['pretrained_arch'] = pretrained_arch
    config['model']['backbone']['strides'] = tuple([1] + [2] * (num_stages - 1))
    config['model']['backbone']['dilations'] = tuple([1] * num_stages)
    config['model']['backbone']['out_indices'] = tuple(range(num_stages))
    config['model']['backbone']['stage_with_dcn'] = tuple([False] * num_stages)
    config['model']['backbone']['stage_with_gcb'] = tuple([False] * num_stages)
    config['model']['backbone']['stage_with_gen_attention'] = tuple([()] * num_stages)

    config['model']['neck']['in_channels'] = contr_neck(arch)
    config['model']['neck']['num_outs'] = num_stages + 1
    config['model']['rpn_head']['anchor_strides'] = list(map(lambda x: pow(2, x), range(2, num_stages + 3)))
    config['model']['bbox_roi_extractor']['featmap_strides'] = list(map(lambda x: pow(2, x), range(2, num_stages + 2)))

    return config
