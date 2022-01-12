# -*- coding: utf-8 -*-

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

"""This is a class for Mnist dataset."""

from torchvision.datasets import MNIST
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.datasets.conf.mnist import MnistConfig
from .dataset import Dataset


@ClassFactory.register(ClassType.DATASET)
class Mnist(MNIST, Dataset):
    """This is a class for Mnist dataset, wchich is the subclass of MNIST and Dataset.

    :param mode: train,val or test , defaults to 'train'
    :type mode: str, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = MnistConfig()

    def __init__(self, **kwargs):
        """Construct the Mnist class."""
        Dataset.__init__(self, **kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        MNIST.__init__(self, root=self.args.data_path, train=self.train,
                       transform=self.transforms, download=self.args.download)

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
