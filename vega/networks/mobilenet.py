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

"""This is SearchSpace for blocks."""
from vega.modules.module import Module
from vega.modules.connections import OutlistSequential
from vega.common import ClassFactory, ClassType
from vega.modules.operators import conv_bn_relu6
from vega.modules.blocks.micro_decoder import InvertedResidual
from vega import is_torch_backend


def _make_divisible(v, divisor, min_value=None):
    """Make value divisible."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@ClassFactory.register(ClassType.NETWORK)
class MobileNetV3Tiny(Module):
    """Create MobileNetV3Tiny SearchSpace."""

    inverted_residual_setting = [
        [1.0, 9, 1],
        [4.0, 14, 2],
        [3.0, 14, 1],
        [3.0, 24, 2],
        [3.0, 24, 1],
        [3.0, 24, 1],
        [6.0, 48, 2],
        [2.5, 48, 1],
        [2.3, 48, 1],
        [2.3, 48, 1],
        [6.0, 67, 1],
        [6.0, 67, 1],
        [6.0, 96, 2],
        [6.0, 96, 1],
        [6.0, 96, 1],
        [6.0, 96, 1]]

    def __init__(self, load_path=None):
        """Construct MobileNetV3Tiny class.

        :param load_path: path for saved model
        """
        super(MobileNetV3Tiny, self).__init__()
        input_channel = 9
        features = [conv_bn_relu6(
            inchannel=3, outchannel=input_channel, kernel=3, stride=2)]
        for _, lst in enumerate(self.inverted_residual_setting):
            output_channel = lst[1]
            features.append(InvertedResidual(
                inp=input_channel, oup=output_channel, stride=lst[2], expand_ratio=lst[0]))
            input_channel = output_channel
        self.block = OutlistSequential(*features, out_list=[3, 6, 13, 17])
        if load_path is not None and is_torch_backend():
            import torch
            self.load_state_dict(torch.load(load_path), strict=False)


@ClassFactory.register(ClassType.NETWORK)
class MobileNetV2Tiny(Module):
    """Create MobileNetV3Tiny SearchSpace."""

    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    def __init__(self, load_path=None, width_mult=1.0, round_nearest=8):
        """Construct MobileNetV3Tiny class.

        :param load_path: path for saved model
        """
        super(MobileNetV2Tiny, self).__init__()
        input_channel = 32
        input_channel = _make_divisible(
            input_channel * width_mult, round_nearest)
        features = [conv_bn_relu6(3, input_channel, 3, 2)]
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(
                    inp=input_channel, oup=output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel
        self.features = OutlistSequential(*features[:18], out_list=[3, 6, 13, 17])
        if load_path is not None and is_torch_backend():
            import torch
            self.load_state_dict(torch.load(load_path), strict=False)
