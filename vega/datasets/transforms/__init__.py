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
