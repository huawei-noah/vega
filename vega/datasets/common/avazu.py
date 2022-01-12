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

"""This is a class for Avazu dataset."""
import logging
import numpy as np
from vega.common import FileOps
from vega.datasets.conf.avazu import AvazuConfig
from vega.common import ClassFactory, ClassType
from .utils.avazu_util import AVAZUDataset
from .dataset import Dataset


@ClassFactory.register(ClassType.DATASET)
class AvazuDataset(Dataset):
    """This is a class for Avazu dataset.

    :param train: if the mode is train or not, defaults to True
    :type train: bool
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = AvazuConfig()

    def __init__(self, **kwargs):
        """Construct the AvazuDataset class."""
        super(AvazuDataset, self).__init__(**kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        logging.info("init new avazu_dataset finish. 0721 debug.")

    @property
    def data_loader(self):
        """Dataloader arrtribute which is a unified interface to generate the data.

        :return: a batch data
        :rtype: dict, list, optional
        """
        return AvazuLoader(args=self.args,
                           gen_type=self.mode,
                           batch_size=self.args.batch_size,
                           random_sample=self.args.random_sample,
                           shuffle_block=self.args.shuffle_block,
                           dir_path=self.args.data_path)


class AvazuLoader(AVAZUDataset):
    """Avazu dataset's data loader."""

    def __init__(self, args=None, gen_type="train", batch_size=2000, random_sample=False,
                 shuffle_block=False, dir_path="./"):
        """Construct avazu_loader class."""
        self.args = args
        AVAZUDataset.__init__(self, dir_path=dir_path)
        self.gen_type = gen_type
        self.batch_size = batch_size
        self.random_sample = random_sample
        self.shuffle_block = shuffle_block

    def __iter__(self):
        """Iterate method for AvazuLoader."""
        return self.batch_generator(gen_type=self.gen_type,
                                    batch_size=self.batch_size,
                                    random_sample=self.random_sample,
                                    shuffle_block=self.shuffle_block)

    def __len__(self):
        """Calculate the length of avazu dataset, thus, number of batch."""
        if self.gen_type == "train":
            return int(np.ceil(1.0 * self.args.train_size / self.args.batch_size))
        else:
            return int(np.ceil(1.0 * self.args.test_size / self.args.batch_size))
