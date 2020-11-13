# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Backend Register."""
import os
from zeus import register_zeus


def set_data_format():
    """Set data format for tensorflow."""
    from zeus.common.general import General
    if General.data_format is None:
        if General.device_category == 'GPU':
            General.data_format = 'channels_first'
        elif General.device_category == 'NPU':
            General.data_format = 'channels_last'


def set_backend(backend='pytorch', device_category='GPU'):
    """Set backend.

    :param backend: backend type, default pytorch
    :type backend: str
    """
    if "BACKEND_TYPE" in os.environ:
        return

    # CUDA visible
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'GPU'
    elif 'NPU-VISIBLE-DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'NPU'
        os.environ['ORIGIN_RANK_TABLE_FILE'] = os.environ['RANK_TABLE_FILE']
        os.environ['ORIGIN_RANK_SIZE'] = os.environ['RANK_SIZE']

    # device
    if device_category is not None:
        os.environ['DEVICE_CATEGORY'] = device_category

    # backend
    if backend == 'pytorch':
        os.environ['BACKEND_TYPE'] = 'PYTORCH'
    elif backend == 'tensorflow':
        os.environ['BACKEND_TYPE'] = 'TENSORFLOW'
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
    elif backend == 'mindspore':
        os.environ['BACKEND_TYPE'] = 'MINDSPORE'
    else:
        raise Exception('backend must be pytorch, tensorflow or mindspore')
    set_data_format()
    register_zeus(backend)

    # vega
    import vega.core.search_algs.ps_differential
    import vega.algorithms

    from zeus.common.config_serializable import backup_configs
    backup_configs()
