# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Cifar10 dataset."""
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from .common.dataset import Dataset
from vega.datasets.pytorch.transforms import Compose
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps


@ClassFactory.register(ClassType.DATASET)
class Cifar10(CIFAR10, Dataset):
    """This is a class for Cifar10 dataset.

    :param mode: `trian`,`val` or `test`, defaults to `train`
    :type mode: str, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    def __init__(self, **kwargs):
        """Construct the Cifar10 class."""
        Dataset.__init__(self, **kwargs)
        CIFAR10.__init__(self, root=self.args.data_path, train=self.train,
                         transform=Compose(self.transforms.__transform__), download=self.args.download)

    @property
    def input_channels(self):
        """Input channel number of the cifar10 image.

        :return: the channel number
        :rtype: int
        """
        _shape = self.data.shape
        _input_channels = 3 if len(_shape) == 4 else 1
        return _input_channels

    @property
    def input_size(self):
        """Input size of cifar10 image.

        :return: the input size
        :rtype: int
        """
        _shape = self.data.shape
        return _shape[1]

    def _init_sampler(self):
        """Init sampler used to cifar10.

        :raises ValueError: the mode should be train, val or test, if not, will raise ValueError
        :return: a sampler method
        :rtype: if the mode is test, rerutrn None, else rerurn a sampler object
        """
        if self.mode == 'test' or self.args.train_portion == 1:
            return None
        self.args.shuffle = False
        num_train = 50000
        indices = list(range(num_train))
        split = int(np.floor(self.args.train_portion * num_train))
        if self.mode == 'train':
            return SubsetRandomSampler(indices[:split])
        elif self.mode == 'val':
            return SubsetRandomSampler(indices[split:num_train])
        else:
            raise ValueError('the mode should be train, val or test')
