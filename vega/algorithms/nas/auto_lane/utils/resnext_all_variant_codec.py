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

"""The ResNeXt_Variant for encode."""
import re
from collections import OrderedDict
from vega.algorithms.nas.auto_lane.utils.str2dict import dict2str
from vega.algorithms.nas.auto_lane.utils.resnet_variant_codec import ResNet_Variant


class ResNeXt_all_Variant(ResNet_Variant):
    """ResNeXt_all_Variant.

    :param base_depth: the base_depth of arch
    :type base_depth: int
    :param base_channel: base channel num
    :type base_channel: int
    :param arch: base arch
    :type arch: str
    :param base_width: base width
    :type base_width: int
    """

    type = 'ResNeXt_all_Variant'
    module_space = {'backbone': 'ResNeXt_all_Variant'}
    id_attrs = ['base_channel', 'arch', 'base_depth', 'groups', 'base_width']

    _block_setting = {18: ('BasicBlock', 8),
                      34: ('BasicBlock', 16),
                      50: ('Bottleneck', 16),
                      101: ('Bottleneck', 33)}

    _base_strides = (1, 2, 2, 2)
    _base_dilations = (1, 1, 1, 1)
    _base_out_indices = (0, 1, 2, 3)

    def __init__(self,
                 base_depth,
                 base_channel,
                 arch,
                 *args,
                 **kwargs):
        super().__init__(base_depth=base_depth,
                         base_channel=base_channel,
                         arch=arch,
                         *args, **kwargs)

        self.base_width = base_channel // 2  # base_width
        self.groups = 2  # base_channel // 2

    def __str__(self):
        """Repr."""
        return self.arch_code

    @property
    def arch_code(self):
        """Return arch code.

        :return: arch code (str)
        :rtype: str
        """
        return 'x{}({}x{}d)_{}_{}'.format(self.base_depth, self.groups, self.base_width,
                                          self.base_channel, self.arch)

    @property
    def base_flops(self):
        """Retrun flops of base model.

        :return: base flops (int)
        :rtype: int
        """
        pass

    @staticmethod
    def arch_decoder(arch_code: str):
        """Retrun params of the model.

        :param arch_code: arch code of the model
        :type arch_code: str
        :return: model dict (dict)
        :rtype: dict
        """
        base_arch_code = {18: 'x18(2x32d)_64_1-21-21-21',
                          34: 'x34(2x32d)_64_111-2111-211111-211',
                          50: 'x50(32x4d)_64_111-2111-211111-211',
                          101: 'x101(32x4d)_64_111-2111-21111111111111111111111-211'}
        if arch_code.startswith('ResNeXt_all'):
            m = re.match(r'ResNeXt_all(?P<base_depth>.*)\((?P<groups>.*)x(?P<base_width>.*)d\)', arch_code)
            base_depth = int(m.group('base_depth'))
            arch_code = base_arch_code[base_depth]
        try:
            m = re.match(r'x(?P<base_depth>.*)\((?P<groups>.*)x(?P<base_width>.*)d\)_(?P<base_channel>.*)_(?P<arch>.*)',
                         arch_code)
            base_depth, groups, base_width, base_channel = map(int, m.groups()[:-1])
            arch = m.group('arch')
            return dict(base_depth=base_depth, groups=groups, base_width=base_width,
                        base_channel=base_channel, arch=arch)
        except Exception:
            raise ValueError('Cannot parse arch code {}'.format(arch_code))

    @property
    def config(self):
        """Return config of the model.

        :return: model config string (str)
        :rtype: str
        """
        config = OrderedDict(
            type='ResNeXt_all_Variant',
            arch=self.arch,
            base_depth=self.base_depth,
            base_channel=self.base_channel,
            groups=self.groups,
            base_width=self.base_width,
            num_stages=self.num_stages,
            strides=self.strides,
            dilations=self.dilations,
            out_indices=self.out_indices,
            frozen_stages=self.frozen_stages,
            zero_init_residual=False,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            out_channels=self.out_channels,
            style='pytorch')
        return config
