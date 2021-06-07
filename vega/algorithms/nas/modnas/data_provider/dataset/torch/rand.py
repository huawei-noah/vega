# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
