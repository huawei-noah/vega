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

"""This is the class for SR dataset."""
from vega.common import ClassFactory, ClassType
from .div2k import DIV2K


@ClassFactory.register(ClassType.DATASET)
class Set5(DIV2K):
    """Set5 dataset, its class interface is same as DIV2K."""

    pass


@ClassFactory.register(ClassType.DATASET)
class Set14(DIV2K):
    """Set14 dataset, its class interface is same as DIV2K."""

    pass


@ClassFactory.register(ClassType.DATASET)
class BSDS100(DIV2K):
    """BSDS100 dataset, its class interface is same as DIV2K."""

    pass
