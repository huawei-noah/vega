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

"""This is a class for Cifar100 dataset."""

import os
import numpy as np
from PIL import Image
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.datasets.conf.cifar100 import Cifar100Config
from .dataset import Dataset


@ClassFactory.register(ClassType.DATASET)
class Cifar100(Dataset):
    """This is a class for Cifar100 dataset.

    :param mode: `train`,`val` or `test`, defaults to `train`
    :type mode: str, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = Cifar100Config()

    def __init__(self, **kwargs):
        """Construct the Cifar10 class."""
        Dataset.__init__(self, **kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        is_train = self.mode == 'train' or self.mode == 'val' and self.args.train_portion < 1
        self.base_folder = 'cifar-100-python'
        if is_train:
            files_list = ["train"]
        else:
            files_list = ['test']

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in files_list:
            file_path = os.path.join(self.args.data_path, self.base_folder, file_name)
            entry = FileOps.load_pickle(file_path, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: tuple
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        """Get the length of the dataset.

        :return: the length of the dataset
        :rtype: int
        """
        return len(self.data)

    @property
    def input_channels(self):
        """Input channels of the cifar100 image.

        :return: the input channels
        :rtype: int
        """
        _shape = self.data.shape
        _input_channels = 3 if len(_shape) == 4 else 1
        return _input_channels

    @property
    def input_size(self):
        """Input size of cifar100 image.

        :return: the input size
        :rtype: int
        """
        _shape = self.data.shape
        return _shape[1]
