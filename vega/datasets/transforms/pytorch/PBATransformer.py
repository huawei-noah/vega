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

"""This is a class for PBATransformer."""
import numpy as np
from vega.common import ClassFactory, ClassType
from ..Cutout import Cutout


@ClassFactory.register(ClassType.TRANSFORM)
class PBATransformer(object):
    """Applies PBATransformer to 'img'.

    The PBATransformer operation combine 15 transformer operations to convert image.
    :param para_array: parameters of the operation specified as an Array.
    :type para_array: array
    """

    transforms = dict()

    def __init__(self, para_array, operation_names, **kwargs):
        """Construct the PBATransformer class."""
        self.para_array = para_array
        self.operation_names = operation_names
        self.split_policy(self.para_array)

    def split_policy(self, raw_policys):
        """Decode raw_policys, get the name, probability and level of each operation.

        :param raw_policys: raw policys which got from .csv file
        :type raw_policys: array
        """
        split = len(raw_policys) // 2
        if split % 2 == 1:
            raise ValueError(
                'set raw_policys illegal, length of raw_policys should be even number!')
        self.policys = self.decode_policys(raw_policys[:split])
        self.policys += self.decode_policys(raw_policys[split:])

    def decode_policys(self, raw_policys):
        """Decode raw_policys, get the name, probability and level of each operation.

        :param raw_policys: raw policys which hasn't been decoded, a list of int number
        :type raw_policys: list
        :return: policys which has been decoded, a list of [name, probability, level] of each operation,
        :rtype: list
        """
        policys = []
        for i, operation_name in enumerate(self.operation_names):
            policys.append((operation_name, int(raw_policys[2 * i]) / 10., int(raw_policys[2 * i + 1])))
        return policys

    def __call__(self, img):
        """Call function of PBATransformer.

        :param img: input image
        :type img: numpy or tensor
        :return: the image after transform
        :rtype: numpy or tensor
        """
        count = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
        policys = self.policys
        np.random.shuffle(policys)
        whether_cutout = [0, 0]
        for policy in policys:
            if count == 0:
                break
            if len(policy) != 3:
                raise ValueError(
                    'set policy illegal, policy should be (op, prob, mag)!')
            op, prob, mag = policy
            if np.random.random() > prob:
                continue
            else:
                count -= 1
                if op == "Cutout":
                    if whether_cutout[0] == 0:
                        whether_cutout[0] = mag
                    else:
                        whether_cutout[1] = mag
                    continue
                operation = ClassFactory.get_cls(ClassType.TRANSFORM, op)
                current_operation = operation(mag)
                img = current_operation(img)

        from torchvision.transforms import functional as F
        if F._is_pil_image(img):
            img = F.to_tensor(img)

        img = Cutout(8)(img)
        for i in whether_cutout:
            if i:
                img = Cutout(i)(img)
        return img
