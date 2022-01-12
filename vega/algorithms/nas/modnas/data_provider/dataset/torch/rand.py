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

"""Random tensor dataset."""
import torch
from torch.utils.data import TensorDataset
from modnas.registry.dataset import register


def get_data_shape(shape):
    """Return tensor shape in data."""
    if shape in [None, 'nil', 'None']:
        return []
    elif isinstance(shape, int):
        return [shape]
    return shape


def get_dtype(dtype):
    """Return tensor dtype in data."""
    if dtype == 'float':
        return torch.float32
    elif dtype == 'int':
        return torch.int64
    else:
        return None


def get_random_data(shape, dtype, drange):
    """Return random tensor data of given shape and dtype."""
    data = torch.Tensor(*shape)
    if drange in [None, 'nil', 'None']:
        data.normal_()
    else:
        data.uniform_(drange[0], drange[1])
    return data.to(dtype=get_dtype(dtype))


@register
def RandData(data_spec, data_size=128):
    """Return random tensor data."""
    data = []
    for dshape, dtype, drange in data_spec:
        data.append(get_random_data([data_size] + get_data_shape(dshape), dtype, drange))
    dset = TensorDataset(*data)
    return dset
