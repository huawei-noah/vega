# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torchvision networks and models."""
import os
from torchvision import models as torchvision_models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from vega.search_space.networks import NetworkFactory
from vega.search_space.networks import NetTypes
from vega.core.common import TaskOps, DefaultConfig


def import_all_torchvision_models():
    """Import all torchvision networks and models."""
    for _name in dir(torchvision_models):
        if not _name.startswith("_"):
            _cls = getattr(torchvision_models, _name)
            NetworkFactory.register_custom_cls(NetTypes.TORCH_VISION_MODEL, _cls)
    NetworkFactory.register_custom_cls(NetTypes.TORCH_VISION_MODEL, fasterrcnn_resnet50_fpn)


def set_torch_home():
    """Set TORCH_HOME to local path."""
    task = TaskOps(DefaultConfig().data.general)
    full_path = os.path.abspath("{}/torchvision_models".format(task.model_zoo_path))
    os.environ['TORCH_HOME'] = full_path


def get_torchvision_model_file(model_name):
    """Get torchvison model file name.

    :param model_name: the name of model, eg. resnet18.
    :type modle_name: str.
    :return: model file name.
    :rtype: string.

    """
    models = {
        'vgg11': 'vgg11-bbd30ac9.pth',
        'vgg13': 'vgg13-c768596a.pth',
        'vgg16': 'vgg16-397923af.pth',
        'vgg19': 'vgg19-dcbb9e9d.pth',
        'vgg11_bn': 'vgg11_bn-6002323d.pth',
        'vgg13_bn': 'vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'vgg16_bn-6c64b313.pth',
        'vgg19_bn': 'vgg19_bn-c79401a0.pth',
        'squeezenet1_0': 'squeezenet1_0-a815701f.pth',
        'squeezenet1_1': 'squeezenet1_1-f364aa15.pth',
        'shufflenetv2_x0.5': 'shufflenetv2_x0.5-f707e7126e.pth',
        'shufflenetv2_x1.0': 'shufflenetv2_x1-5666bf0f80.pth',
        'resnet18': 'resnet18-5c106cde.pth',
        'resnet34': 'resnet34-333f7ec4.pth',
        'resnet50': 'resnet50-19c8e357.pth',
        'resnet101': 'resnet101-5d3b4d8f.pth',
        'resnet152': 'resnet152-b121ed2d.pth',
        'resnext50_32x4d': 'resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'wide_resnet101_2-32ee1156.pth',
        'mobilenet_v2': 'mobilenet_v2-b0353104.pth',
        "mnasnet0_5": "mnasnet0.5_top1_67.592-7c6cb539b9.pth",
        "mnasnet1_0": "mnasnet1.0_top1_73.512-f206786ef8.pth",
        'inception_v3_google': 'inception_v3_google-1a9a5a14.pth',
        'googlenet': 'googlenet-1378be20.pth',
        'densenet121': 'densenet121-a639ec97.pth',
        'densenet169': 'densenet169-b2777c0a.pth',
        'densenet201': 'densenet201-c1103571.pth',
        'densenet161': 'densenet161-8d451a50.pth',
        'alexnet': 'alexnet-owt-4df8aa71.pth',
        'fasterrcnn_resnet50_fpn': 'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        'fasterrcnn_resnet50_fpn_coco': 'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        'keypointrcnn_resnet50_fpn_coco': 'keypointrcnn_resnet50_fpn_coco-9f466800.pth',
        'maskrcnn_resnet50_fpn_coco': 'maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        'fcn_resnet101_coco': 'fcn_resnet101_coco-7ecb50ca.pth',
        'deeplabv3_resnet101_coco': 'deeplabv3_resnet101_coco-586e9e4e.pth',
        'r3d_18': 'r3d_18-b3b3357e.pth',
        'mc3_18': 'mc3_18-a90a0ba3.pth',
        'r2plus1d_18': 'r2plus1d_18-91a641e6.pth',
    }
    if model_name in models:
        return models[model_name]
    else:
        return None
