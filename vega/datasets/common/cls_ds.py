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

"""This is a class for classification dataset."""

import random
import os
import PIL
import vega
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.datasets.conf.cls_ds import ClassificationDatasetConfig
from .dataset import Dataset


@ClassFactory.register(ClassType.DATASET)
class ClassificationDataset(Dataset):
    """This is a class for Classification dataset.

    :param mode: `train`,`val` or `test`, defaults to `train`
    :type mode: str, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = ClassificationDatasetConfig()

    def __init__(self, **kwargs):
        """Construct the classification class."""
        Dataset.__init__(self, **kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        sub_path = os.path.abspath(os.path.join(self.args.data_path, self.mode))
        if self.args.train_portion != 1.0 and self.mode == "val":
            sub_path = os.path.abspath(os.path.join(self.args.data_path, "train"))
        if self.args.train_portion == 1.0 and self.mode == "val" and not os.path.exists(sub_path):
            sub_path = os.path.abspath(os.path.join(self.args.data_path, "test"))
        if not os.path.exists(sub_path):
            raise("dataset path is not existed, path={}".format(sub_path))
        self._load_file_indexes(sub_path)
        self._load_data()
        self._shuffle()

    def _load_file_indexes(self, sub_path):
        self.classes = [_file for _file in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, _file))]
        if not self.classes:
            raise("data folder has not sub-folder, path={}".format(sub_path))
        self.n_class = len(self.classes)
        self.classes.sort()
        self.file_indexes = []
        for _cls in self.classes:
            _path = os.path.join(sub_path, _cls)
            self.file_indexes += [(_cls, os.path.join(_path, _file)) for _file in os.listdir(_path)]
        if not self.file_indexes:
            raise("class folder has not image, path={}".format(sub_path))
        self.args.n_images = len(self.file_indexes)
        self.data = None

    def __len__(self):
        """Get the length of the dataset.

        :return: the length of the dataset
        :rtype: int
        """
        return len(self.file_indexes)

    def __getitem__(self, index):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: dict, {'data': xx, 'mask': xx, 'name': name}
        """
        if self.args.cached:
            (label, _, image) = self.data[index]
        else:
            (label, _file) = self.file_indexes[index]
            image = self._load_image(_file)
        image = self.transforms(image)
        n_label = self.classes.index(label)
        return image, n_label

    def _load_data(self):
        if not self.args.cached:
            return
        # TODO read file multi thread
        self.data = [(_cls, _file, self._load_image(_file)) for (_cls, _file) in self.file_indexes]

    def _load_image(self, image_file):
        img = PIL.Image.open(image_file)
        img = img.convert("RGB")
        return img

    def _to_tensor(self, data):
        if vega.is_torch_backend():
            import torch
            return torch.tensor(data)
        elif vega.is_tf_backend():
            import tensorflow as tf
            return tf.convert_to_tensor(data)

    def _shuffle(self):
        if self.args.cached:
            random.shuffle(self.data)
        else:
            random.shuffle(self.file_indexes)
