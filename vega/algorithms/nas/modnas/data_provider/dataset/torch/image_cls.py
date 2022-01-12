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

"""Datasets for image classification."""

from typing import Callable, Optional, Dict, List, Any
import os
import torch
import numpy as np
from torchvision import transforms, datasets
from modnas.registry.dataset import register
from torch.utils.data.dataset import Dataset


_metadata = {
    'cifar10': {
        'mean': [0.49139968, 0.48215827, 0.44653124],
        'stddev': [0.24703233, 0.24348505, 0.26158768],
    },
    'cifar100': {
        'mean': [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
        'stddev': [0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
    },
    'mnist': {
        'mean': [0.13066051707548254],
        'stddev': [0.30810780244715075],
    },
    'fashionmnist': {
        'mean': [0.28604063146254594],
        'stddev': [0.35302426207299326],
    },
    'imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'stddev': [0.229, 0.224, 0.225],
    },
}


_default_metadata = {
    'mean': [0.5, 0.5, 0.5],
    'stddev': [0, 0, 0],
}


_dsets = {
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'mnist': datasets.MNIST,
    'fashionmnist': datasets.FashionMNIST,
    'imagenet': datasets.ImageFolder,
    'image': datasets.ImageFolder,
}


_train_transforms: Dict[str, Callable] = {
    'cifar10':
    lambda: [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()],
    'cifar100':
    lambda: [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()],
    'mnist':
    lambda: [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)],
    'fashionmnist':
    lambda:
    [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
     transforms.RandomVerticalFlip()],
    'imagenet':
    lambda resize_scale=0.08: [
        transforms.RandomResizedCrop(224, scale=(resize_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
    ],
    'image':
    lambda resize_scale=0.08: [
        transforms.RandomResizedCrop(224, scale=(resize_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
    ],
}


_valid_transforms: Dict[str, Callable] = {
    'imagenet': lambda: [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ],
    'image': lambda: [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ],
}


class Cutout(object):
    """Apply Cutout on dataset."""

    def __init__(self, length, seed=11235):
        self.length = length
        self.rng = np.random.RandomState(seed)

    def __call__(self, img):
        """Return image with Cutout applied."""
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = self.rng.randint(h)
        x = self.rng.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


@register
def ImageClsData(dataset: str,
                 root: str,
                 valid: bool = False,
                 mean: Optional[List[float]] = None,
                 stddev: Optional[List[float]] = None,
                 cutout: int = 0,
                 jitter: bool = False,
                 transform_args: Optional[Dict[str, Any]] = None,
                 to_tensor: bool = True) -> Dataset:
    """Return dataset for image classification."""
    dataset = dataset.lower()
    dset = _dsets.get(dataset)
    if dset is None:
        raise ValueError('unsupported dataset: {}'.format(dataset))
    meta = _metadata.get(dataset, _default_metadata)
    mean = meta['mean'] if mean is None else mean
    stddev = meta['stddev'] if stddev is None else stddev
    transf_all = _valid_transforms if valid else _train_transforms
    transf = transf_all.get(dataset, lambda: [])(**(transform_args or {}))
    if jitter is True or jitter == 'strong':
        transf.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
    elif jitter == 'normal':
        transf.append(transforms.ColorJitter(brightness=32. / 255., saturation=0.5))
    if to_tensor:
        transf.extend([transforms.ToTensor(), transforms.Normalize(mean, stddev)])
    if cutout > 0:
        transf.append(Cutout(cutout))
    os.makedirs(root, exist_ok=True)
    if dset == datasets.ImageFolder:
        data = dset(root, transform=transforms.Compose(transf))
    else:
        data = dset(root, train=(not valid), transform=transforms.Compose(transf), download=True)
    return data
