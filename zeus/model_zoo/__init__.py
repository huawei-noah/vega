# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import and register torch vision model automatically."""

from .model_zoo import ModelZoo


def register_modelzoo(backend):
    """Import and register modelzoo automatically."""
    if backend != "pytorch":
        return
    from .torch_vision_model import import_all_torchvision_models
    import logging
    try:
        import_all_torchvision_models()
    except Exception as e:
        logging.warn("Failed to import torchvision models, msg={}".format(str(e)))
