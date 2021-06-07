# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Datasets for image classification."""
import os
import numpy as np
import torch
from torchvision import transforms, datasets
from modnas.registry.dataset import register


def get_metadata(dataset):
    """Return dataset metadata."""
    if dataset == 'cifar10':
        mean = [0.49139968, 0.48215827, 0.44653124]
        stddev = [0.24703233, 0.24348505, 0.26158768]
    elif dataset == 'cifar100':
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        stddev = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    elif dataset == 'mnist':
        mean = [0.13066051707548254]
        stddev = [0.30810780244715075]
    elif dataset == 'fashionmnist':
        mean = [0.28604063146254594]
        stddev = [0.35302426207299326]
    elif dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        stddev = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        stddev = [0, 0, 0]
    return {
        'mean': mean,
        'stddev': stddev,
    }


_train_transforms = {
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

_valid_transforms = {
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

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """Return image with Cutout applied."""
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

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
def ImageClsData(dataset,
                 root,
                 valid=False,
                 mean=None,
                 stddev=None,
                 cutout=0,
                 jitter=False,
                 transform_args=None,
                 to_tensor=True):
    """Return dataset for image classification."""
    dataset = dataset.lower()
    meta = get_metadata(dataset)
    mean = meta['mean'] if mean is None else mean
    stddev = meta['stddev'] if stddev is None else stddev
    os.makedirs(root, exist_ok=True)
    if dataset == 'cifar10':
        dset = datasets.CIFAR10
    elif dataset == 'cifar100':
        dset = datasets.CIFAR100
    elif dataset == 'mnist':
        dset = datasets.MNIST
    elif dataset == 'fashionmnist':
        dset = datasets.FashionMNIST
    elif dataset == 'imagenet':
        dset = datasets.ImageFolder
    elif dataset == 'image':
        dset = datasets.ImageFolder
    else:
        raise ValueError('unsupported dataset: {}'.format(dataset))
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

    if dset == datasets.ImageFolder:
        data = dset(root, transform=transforms.Compose(transf))
    else:
        data = dset(root, train=(not valid), transform=transforms.Compose(transf), download=True)
    return data
