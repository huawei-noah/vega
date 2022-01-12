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

"""This is the class for DIV2K dataset."""
import os
import os.path
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.datasets.conf.div2k import DIV2KConfig
from .utils import div2k_util as util
from .dataset import Dataset


@ClassFactory.register(ClassType.DATASET)
class DIV2K(Dataset):
    """Construct the class of DIV2K dataset, which is a subclass of Dataset.

    :param train: if the mdoe is train or false, defaults to True
    :type train: bool, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = DIV2KConfig()

    def __init__(self, **kwargs):
        """Construct the DistributorBaseClass class.."""
        super(DIV2K, self).__init__(**kwargs)
        self.dataset_init()

    def _init_transforms(self):
        """Initialize transform used to Cityscapes dataset.

        :return: return the default transforms method used to DIV2K, usually the train dataset
        and the test dataset is different
        :rtype: list
        """
        if self.train:
            result = list()
            if self.args.crop is not None:
                result.append(self._get_cls("RandomCrop_pair")(self.args.crop, self.args.upscale))
            if self.args.hflip:
                result.append(self._get_cls("RandomHorizontalFlip_pair")())
            if self.args.vflip:
                result.append(self._get_cls("RandomVerticallFlip_pair")())
            if self.args.rot90:
                result.append(self._get_cls("RandomRotate90_pair")())
            return result
        else:
            result = list()
            if self.args.crop is not None:
                result.append(self._get_cls("RandomCrop_pair")(self.args.crop, self.args.upscale))
            return result

    def _get_cls(self, _name):
        return ClassFactory.get_cls(ClassType.TRANSFORM, _name)

    def dataset_init(self):
        """Costruct method, which will load some dataset information."""
        self.args.root_HR = FileOps.download_dataset(self.args.root_HR)
        self.args.root_LR = FileOps.download_dataset(self.args.root_LR)
        if self.args.subfile is not None:
            with open(self.args.subfile) as f:  # lmdb format has no self.args.subfile
                file_names = sorted([line.rstrip('\n') for line in f])
                self.datatype = util.get_files_datatype(file_names)
                self.paths_HR = [os.path.join(self.args.root_HR, file_name) for file_name in file_names]
                self.paths_LR = [os.path.join(self.args.root_LR, file_name) for file_name in file_names]
        else:
            self.datatype = util.get_datatype(self.args.root_LR)
            self.paths_LR = util.get_paths_from_dir(self.args.root_LR)
            self.paths_HR = util.get_paths_from_dir(self.args.root_HR)

        if self.args.save_in_memory:
            self.imgs_LR = [self._read_img(path) for path in self.paths_LR]
            self.imgs_HR = [self._read_img(path) for path in self.paths_HR]

    def __len__(self):
        """Get the length of the dataset.

        :return: the length of the dataset
        :rtype: int
        """
        return len(self.paths_HR)

    def _read_img(self, path):
        """Read image from path according to the datatype.

        :path: path of the image
        :type path: str
        :return: image in np array, np.uint8, HWC, BGR
        :rtype: np.array
        """
        if self.datatype == 'pkl':
            return util.read_img_pkl(path)
        else:
            return util.read_img_img(path)

    def __getitem__(self, index):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: dict, {'LR': xx, 'HR': xx, 'LR_path': xx, 'HR_path': xx}
        """
        LR_path, HR_path = self.paths_LR[index], self.paths_HR[index]
        if self.args.save_in_memory:
            img_LR, img_HR = self.imgs_LR[index], self.imgs_HR[index]
        else:
            img_LR, img_HR = self._read_img(LR_path), self._read_img(HR_path)
        img_LR, img_HR = self.transforms(img_LR, img_HR)
        img_LR, img_HR = util.np_to_tensor(img_LR), util.np_to_tensor(img_HR)
        # float32, CHW, BGR, [0,1]
        img_LR = img_LR / self.args.value_div
        img_HR = img_HR / self.args.value_div
        return img_LR, img_HR
