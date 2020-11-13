# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import and register network automatically."""


def register_networks(backend):
    """Import and register network automatically."""
    from .network_desc import NetworkDesc
    from .adelaide import AdelaideFastNAS
    from .erdb_esr import ESRN
    from .mobilenet import MobileNetV3Tiny, MobileNetV2Tiny
    from .mobilenetv3 import MobileNetV3Small, MobileNetV3Large
    from .necks import FPN
    from . import resnet
    from . import quant
    from . import mtm_sr
    from . import super_network
    from . import resnet_det
    from . import resnet_general
    from . import resnext_det
    from . import xt_model
    from . import text_cnn
    from . import faster_rcnn
    if backend == "pytorch":
        from . import pytorch
    elif backend == "tensorflow":
        from . import tensorflow
    elif backend == "mindspore":
        from . import mindspore
