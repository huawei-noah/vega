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

"""This is SearchSpace for preprocess."""
from vega.modules.module import Module
from vega.common import ClassFactory, ClassType
from vega.modules.operators import ops


@ClassFactory.register(ClassType.NETWORK)
class PreOneStem(Module):
    """Class of one stem convolution.

    :param desc: description of PreOneStem
    :type desc: Config
    """

    def __init__(self, init_channels, stem_multi):
        """Init PreOneStem."""
        super(PreOneStem, self).__init__()
        self._c_curr = init_channels * stem_multi
        self.conv2d = ops.Conv2d(3, self._c_curr, 3, padding=1, bias=False)
        self.batchNorm2d = ops.BatchNorm2d(self._c_curr)

    @property
    def output_channel(self):
        """Get Output channel."""
        return self._c_curr

    def call(self, x):
        """Forward function of PreOneStem."""
        x = super().call(x)
        return x, x
