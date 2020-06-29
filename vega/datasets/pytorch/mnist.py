# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Mnist dataset."""
from copy import deepcopy
from torchvision.datasets import MNIST
from .common.dataset import Dataset
from vega.datasets.pytorch.transforms import Compose
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps


@ClassFactory.register(ClassType.DATASET)
class Mnist(MNIST, Dataset):
    """This is a class for Mnist dataset, wchich is the subclass of MNIST and Dateset.

    :param mode: trian,val or test , defaults to 'train'
    :type mode: str, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    def __init__(self, **kwargs):
        """Construct the Mnist class."""
        Dataset.__init__(self, **kwargs)
        MNIST.__init__(self, root=self.args.data_path, train=self.train,
                       transform=Compose(self.transforms.__transform__), download=self.args.download)

    @property
    def input_channels(self):
        """Input channel number of the Mnist image.

        :return: the channel number
        :rtype: int
        """
        _shape = self.data.shape
        _input_channels = 3 if len(_shape) == 4 else 1
        return _input_channels

    @property
    def input_size(self):
        """Input size of Mnist image.

        :return: the input size
        :rtype: int
        """
        _shape = self.data.shape
        return _shape[1]
