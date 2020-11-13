# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Imagenet dataset."""
from torchvision.datasets import ImageFolder
from .utils.dataset import Dataset
from zeus.datasets.transforms import Compose
from zeus.common import ClassFactory, ClassType
from zeus.common import FileOps
from zeus.datasets.conf.imagenet import ImagenetConfig


@ClassFactory.register(ClassType.DATASET)
class Imagenet(ImageFolder, Dataset):
    """This is a class for Imagenet dataset, wchich is the subclass of ImageNet and Dateset.

    :param mode: `train`,`val` or `test`, defaults to `train`
    :type mode: str, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = ImagenetConfig()

    def __init__(self, **kwargs):
        """Construct the Imagenet class."""
        Dataset.__init__(self, **kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        split = 'train' if self.mode == 'train' else 'val'
        local_data_path = FileOps.join_path(self.args.data_path, split)
        ImageFolder.__init__(self, root=local_data_path, transform=Compose(self.transforms.__transform__))

    @property
    def input_channels(self):
        """Input channel number of the Imagenet image.

        :return: the channel number
        :rtype: int
        """
        return 3

    @property
    def input_size(self):
        """Input size of Imagenet image.

        :return: the input size
        :rtype: int
        """
        return self.shape[1]
