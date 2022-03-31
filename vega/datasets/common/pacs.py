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

"""This is a class for Pacs dataset."""
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.datasets.conf.pacs import PacsConfig
from .dataset import Dataset


@ClassFactory.register(ClassType.DATASET)
class Pacs(Dataset):
    """This is a class for Pacs dataset.

    :param mode: `train`,`val` or `test`, defaults to `train`
    :type mode: str, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = PacsConfig()

    def __init__(self, **kwargs):
        """Construct the Pacs class."""
        Dataset.__init__(self, **kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        targetdomain = self.args.targetdomain
        domain = ['cartoon', 'art_painting', 'photo', 'sketch']
        if self.mode == "train":
            domain.remove(targetdomain)
        else:
            domain = [targetdomain]
        full_data = []
        label_name = []
        full_concept = []
        for k, domain_name in enumerate(domain):
            split_path = os.path.join(self.args.split_path, domain_name + '_all' + '.txt')
            images, labels = self._dataset_info(split_path)
            concept = [k] * len(labels)
            full_data.extend(images)
            label_name.extend(labels)
            full_concept.extend(concept)

        classes = list(set(label_name))
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        full_label = [class_to_idx[x] for x in label_name]
        self.data = full_data
        self.label = full_label
        self.concept = full_concept

    def __getitem__(self, index):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: tuple
        """
        data, label, concept = self.data[index], self.label[index], self.concept[index]
        img = Image.open(data).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        if self.args.task == 'nas_ood':
            return {'input':img, 'target': label, 'concept': concept}, label
        return img, (label, concept)

    def _dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            path = os.path.join(self.args.data_path, row[0])
            path = path.replace('\\', '/')
            file_names.append(path)
            labels.append(int(row[1]))
        return file_names, labels

    def __len__(self):
        """Get the length of the dataset.

        :return: the length of the dataset
        :rtype: int
        """
        return len(self.data)

    @property
    def input_channels(self):
        """Input channel number of the pacs image.

        :return: the channel number
        :rtype: int
        """
        _shape = self.data.shape
        _input_channels = 3 if len(_shape) == 4 else 1
        return _input_channels

    @property
    def input_size(self):
        """Input size of pacs image.

        :return: the input size
        :rtype: int
        """
        _shape = self.data.shape
        return _shape[1]

