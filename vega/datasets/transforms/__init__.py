# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Lazy import transforms."""

from vega.common.class_factory import ClassFactory

ClassFactory.lazy_register("vega.datasets.transforms", {
    # common
    "AutoAugment": ["AutoAugment"],
    "AutoContrast": ["AutoContrast"],
    "BboxTransform": ["BboxTransform"],
    "Brightness": ["Brightness"],
    "Color": ["Color"],
    "Compose": ["Compose", "ComposeAll"],
    "Compose_pair": ["Compose_pair"],
    "Contrast": ["Contrast"],
    "Cutout": ["Cutout"],
    "Equalize": ["Equalize"],
    "RandomCrop_pair": ["RandomCrop_pair"],
    "RandomHorizontalFlip_pair": ["RandomHorizontalFlip_pair"],
    "RandomMirrow_pair": ["RandomMirrow_pair"],
    "RandomRotate90_pair": ["RandomRotate90_pair"],
    "RandomVerticallFlip_pair": ["RandomVerticallFlip_pair"],
    "RandomColor_pair": ["RandomColor_pair"],
    "RandomRotate_pair": ["RandomRotate_pair"],
    "Rescale_pair": ["Rescale_pair"],
    "Normalize_pair": ["Normalize_pair"],
    # GPU only
    "ImageTransform": ["ImageTransform"],
    "Invert": ["Invert"],
    "MaskTransform": ["MaskTransform"],
    "Posterize": ["Posterize"],
    "Rotate": ["Rotate"],
    "SegMapTransform": ["SegMapTransform"],
    "Sharpness": ["Sharpness"],
    "Shear_X": ["Shear_X"],
    "Shear_Y": ["Shear_Y"],
    "Solarize": ["Solarize"],
    "Translate_X": ["Translate_X"],
    "Translate_Y": ["Translate_Y"],
    "RandomGaussianBlur_pair": ["RandomGaussianBlur_pair"],
    "RandomHorizontalFlipWithBoxes": ["RandomHorizontalFlipWithBoxes"],
    "Resize": ["Resize"],
    "RandomCrop": ["RandomCrop"],
    "RandomHorizontalFlip": ["RandomHorizontalFlip"],
    "Normalize": ["Normalize"],
    "ToTensor": ["ToTensor"],
})

ClassFactory.lazy_register("vega.datasets.transforms.pytorch", {
    "Numpy2Tensor": ["Numpy2Tensor"],
    "PBATransformer": ["PBATransformer"],
    "ToPILImage_pair": ["ToPILImage_pair"],
    "ToTensor_pair": ["ToTensor_pair", "PILToTensorAll", "ToTensorAll"],
})

ClassFactory.lazy_register("mmdet.datasets", {
    "extra_aug": ["PhotoMetricDistortion", "Expand", "ExtraAugmentation"],
})
