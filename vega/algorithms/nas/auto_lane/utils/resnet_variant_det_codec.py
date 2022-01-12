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

"""The ResNet_Variant for encode."""

import random
import re
import logging
from collections import OrderedDict
import numpy as np
from .listdict import ListDict
from .backbone_codec import Backbone


class ResNetVariantDetCodec(Backbone):
    """This is the class of ResNet Variant.

    :param arch: the string of arch
    :type arch: string
    :param base_depth: base depth
    :type base_depth: int
    :param base_channel: base channel num
    :type base_channel: int
    """

    type = 'ResNet_Variant'
    module_space = {'backbone': 'ResNet_Variant'}
    attr_space = dict(arch=dict(num_reduction=3,
                                num_stage=4,
                                num_block=[5, 35]),
                      base_channel=[48, 56, 64, 72])
    id_attrs = ['base_channel', 'arch', 'base_depth']
    _block_setting = {18: ('BasicBlock', 8),
                      34: ('BasicBlock', 16),
                      50: ('Bottleneck', 16),
                      101: ('Bottleneck', 33)}
    _base_strides = (1, 2, 2, 2)
    _base_dilations = (1, 1, 1, 1)
    _base_out_indices = (0, 1, 2, 3)

    def __init__(self,
                 arch,
                 base_depth,
                 base_channel,
                 *args,
                 **kwargs):
        """Contruct ResNet_Variant encoder."""
        super(ResNetVariantDetCodec, self).__init__(*args, **kwargs)
        self.arch = arch
        self.base_channel = int(base_channel)
        self.depth = self.base_depth = base_depth
        block = self._block_setting[base_depth][0]
        if block not in ['BasicBlock', 'Bottleneck']:
            raise Exception('Invalid block name. (should be BasicBlock or Bottleneck)')
        expansion = 1 if block == 'BasicBlock' else 4
        if self.train_from_scratch:
            self.zero_init_residual = False
            self.frozen_stages = -1
            self.norm_cfg = dict(type='BN', requires_grad=True)
        else:
            self.zero_init_residual = True
            self.frozen_stages = 1
        self.num_stages = len(arch.split('-'))
        self.strides = self._base_strides[:self.num_stages]
        self.dilations = self._base_dilations[:self.num_stages]
        self.out_indices = self._base_out_indices[:] if self.with_neck else (2,)
        self.out_strides = [2 ** (i + 2) for i in range(self.num_stages)] if self.with_neck else [16]
        num_scale = 0
        self.out_channels = []
        for stage in range(self.num_stages):
            n = self.arch.split('-')[stage].count('2')
            num_scale += n
            self.out_channels.append(self.base_channel * expansion * (2 ** num_scale))

    def __str__(self):
        """Return arch code.

        :return: arch code (str)
        :rtype: str
        """
        return self.arch_code

    @property
    def arch_code(self):
        """Return arch code.

        :return: arch code (str)
        :rtype: str
        """
        return 'r{}_{}_{}'.format(self.base_depth, self.base_channel, self.arch)

    @property
    def base_flops(self):
        """Return base flops.

        :return: base flops
        :rtype: float
        """
        pass

    @staticmethod
    def arch_decoder(arch_code: str):
        """Decode code for Resnet.

        :param arch_code: arch code
        :type arch_code: str
        :return: model dict (dict)
        :rtype: dict
        """
        base_arch_code = {18: 'r18_64_11-21-21-21',
                          34: 'r34_64_111-2111-211111-211',
                          50: 'r50_64_111-2111-211111-211',
                          101: 'r101_64_111-2111-21111111111111111111111-211'}
        if arch_code.startswith('ResNet'):
            base_depth = int(arch_code.split('ResNet')[-1])
            arch_code = base_arch_code[base_depth]
        try:
            m = re.match(r'r(?P<base_depth>.*)_(?P<base_channel>.*)_(?P<arch>.*)', arch_code)
            base_depth, base_channel = map(int, m.groups()[:-1])
            arch = m.group('arch')
            return dict(base_depth=base_depth, base_channel=base_channel, arch=arch)
        except Exception:
            raise ValueError('Cannot parse arch code {}'.format(arch_code))

    @property
    def flops_ratio(self):
        """Get the flops ratio of generated archtecture and base archtecture.

        :return: flops ratio
        :rtype: float
        """
        return round(self.size_info['FLOPs'] / self.base_flops, 3)

    @classmethod
    def sample(cls,
               method='random',
               base_depth=50,
               base_arch=None,
               sampled_archs=None,
               flops_constraint=None,
               EA_setting=None,
               fore_part=None,
               max_sample_num=100000,
               **kwargs
               ):
        """Sample a new archtecture to train.

        :param method: search method
        :type method: str
        :param base_depth: base depth
        :type base_depth: int
        :param base_arch: base arch code
        :type base_arch: str
        :param sampled_archs: search space of arch
        :type sampled_archs: list
        :param flops_constraint: flops constraint
        :type flops_constraint: list
        :param EA_setting: ea setting
        :type EA_setting: dict
        :param fore_part: fore part
        :type fore_part: Module
        :param max_sample_num: max sample nums
        :type max_sample_num: int
        :return: model dict
        """
        if sampled_archs is None:
            sampled_archs = []
        if EA_setting is None:
            EA_setting = dict(num_mutate=3)
        if flops_constraint is None:
            low_flops, high_flops = 0, float('inf')
        else:
            low_flops, high_flops = flops_constraint
        sample_num = 0
        discard = ListDict()
        params = {}
        while sample_num < max_sample_num:
            sample_num += 1
            if method == 'random':
                with_neck = params.get('with_neck')
                params.update(cls.random_sample(with_neck))
            elif method == 'EA':
                params.update(cls.EA_sample(base_arch=base_arch, **EA_setting))
            elif method == 'proposal':
                params.update(cls.arch_decoder(arch_code=base_arch))
            else:
                raise ValueError('Unrecognized sample method {}.')
            net = cls(**params, base_depth=base_depth, fore_part=fore_part)
            exist = net.name in sampled_archs + discard['arch']
            success = low_flops <= net.flops_ratio <= high_flops
            flops_info = '{}({})'.format(net.flops, net.flops_ratio)
            if exist:
                continue
            if success:
                return dict(arch=net, discard=discard)
            else:
                discard.append(dict(arch=net.name, flops=flops_info))
        return None

    @classmethod
    def random_sample(cls, base_channel, base_depth, with_neck=True):
        """Random sample a model arch.

        :return: model dict (dict)
        :rtype: dict
        """
        if with_neck:
            num_reduction, num_stage = 3, 4
        else:
            num_reduction, num_stage = 2, 3
        arch_space = cls.attr_space['arch']
        length = random.randint(*arch_space['num_block'])
        arch = ['1'] * length
        position = np.random.choice(length, size=num_reduction, replace=False)
        for p in position:
            arch[p] = '2'
        insert = np.random.choice(length - 1, size=num_stage - 1, replace=False)
        insert = [i + 1 for i in insert]
        insert = reversed(sorted(insert))
        for i in insert:
            arch.insert(i, '-')
        return dict(base_channel=base_channel, base_depth=base_depth, arch=''.join(arch))

    @classmethod  # noqa: C901
    def EA_sample(cls, base_arch, num_mutate=3, **kwargs):
        """Use ea to sample a model.

        :return: model dict
        :rtype: dict
        """

        def _chwidth(cls, cur_channel):
            """Return new channel number.

            :param cur_channel: num of current channel number
            :type cur_channel: int
            :return: channel num (int)
            :rtype: int
            """
            index = cls.base_channels.index(cur_channel)
            candidate = [i for i in range(index - 1, index + 2) if 0 <= i < len(cls.base_channels) and i != index]
            channel = cls.base_channels[random.choice(candidate)]
            ops.append('chwidth:{}->{}'.format(cur_channel, channel))
            return channel

        def _insert(arch):
            """Return new arch code.

            :param arch: current arch code
            :type arch: str
            :return: arch code (str)
            :rtype: str
            """
            idx = np.random.randint(low=0, high=len(arch))
            arch.insert(idx, '1')
            ops.append('insert:{}'.format(idx))
            return arch

        def _remove(arch):
            """Return new arch code.

            :param arch: current arch code
            :type arch: str
            :return: arch code (str)
            :rtype: str
            """
            ones_index = [i for i, char in enumerate(arch) if char == '1']
            idx = random.choice(ones_index)
            arch.pop(idx)
            ops.append('remove:{}'.format(idx))
            return arch

        def _swap(arch, R):
            """Return new arch code.

            :param arch: current arch code
            :type arch: str
            :return: arch code (str)
            """
            while True:
                not_ones_index = [i for i, char in enumerate(arch) if char != '1']
                idx = random.choice(not_ones_index)
                r = random.randint(1, R)
                direction = -r if random.random() > 0.5 else r
                try:
                    arch[idx], arch[idx + direction] = arch[idx + direction], arch[idx]
                    break
                except Exception:
                    logging.debug("Arch is not match, continue.")
                    continue
            ops.append('swap:{}&{}'.format(idx, idx + direction))
            return arch

        def is_valid(arch):
            """Return if the arch in search space.

            :param arch: current arch code
            :type arch: str
            :return: if the model is valid (bool)
            """
            stages = arch.split('-')
            length = 0
            for stage in stages:
                if len(stage) == 0:
                    return False
                length += len(stage)
            return min_block <= length <= max_block

        arch_space = cls.attr_space['arch']
        min_block, max_block = arch_space['num_block']
        params = cls.arch_decoder(base_arch)
        base_channel, base_arch = params.get('base_channel'), params.get('arch')
        while True:
            ops = []
            new_arch = list(base_arch)
            new_channel = base_channel
            try:
                if random.random() > 0.5:
                    new_channel = _chwidth(base_channel)
                for i in range(num_mutate):
                    op_idx = np.random.randint(low=0, high=3)
                    if op_idx == 0:
                        new_arch = _insert(new_arch)
                    elif op_idx == 1:
                        new_arch = _remove(new_arch)
                    elif op_idx == 2:
                        R = num_mutate // 2
                        new_arch = _swap(new_arch, R)
                    else:
                        raise Exception('operation index out of range')
            except Exception:
                logging.debug("Arch is not match, continue.")
                continue
            new_arch = ''.join(new_arch)
            if is_valid(new_arch) and (new_arch != base_arch or new_channel != base_channel):
                break
        params.update(dict(base_channel=new_channel, arch=new_arch))
        return params

    @property
    def config(self):
        """Return config dict.

        :return: config dict (str)
        :rtype: str
        """
        config = OrderedDict(
            type='ResNetVariantDet',
            arch=self.arch,
            base_depth=self.base_depth,
            base_channel=self.base_channel,
            num_stages=self.num_stages,
            strides=self.strides,
            dilations=self.dilations,
            out_indices=self.out_indices,
            frozen_stages=self.frozen_stages,
            zero_init_residual=self.zero_init_residual,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            out_channels=self.out_channels,
            style='pytorch')
        return config
