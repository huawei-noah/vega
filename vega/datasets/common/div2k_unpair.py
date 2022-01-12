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

"""This is the class for DIV2K dataset for the unpaird setting."""
import logging
import os
import os.path
import random
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.datasets.conf.div2k import DIV2KConfig
from .dataset import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ]


def is_image_file(filename):
    """If a file is image mode or not.

    :param filename: the file's name
    :type filename: str
    :return: True or False
    :rtype: bool
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


@ClassFactory.register(ClassType.DATASET)
class Div2kUnpair(Dataset):
    """This is the class of DIV2K dataset for unpaired setting, which is a subclass of Dataset.

    :param train: if the mdoe is train or false, defaults to True
    :type train: bool, optional
    :param self.args: the config the dataset need, defaults to None
    :type self.args: yml, py or dict
    """

    config = DIV2KConfig()

    def make_dataset(self, dir, max_dataset_size=float("inf")):
        """Generate a list of images.

        :param dir: path of the image
        :type dir: str
        :param max_dataset_size: The maximum length of dataset, default:"inf"
        :type max_dataset_size: float
        :return: a list of training or testing images
        :rtype: list
        """
        images = []
        if not os.path.isdir(dir):
            raise TypeError('%s is not a valid directory' % dir)
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_dataset_size, len(images))]

    def __init__(self, **kwargs):
        """Initialize method."""
        super(Div2kUnpair, self).__init__(**kwargs)
        self.dataset_init()

    def dataset_init(self):
        """Initialize dataset."""
        self.args.HR_dir = FileOps.download_dataset(self.args.HR_dir)
        self.args.LR_dir = FileOps.download_dataset(self.args.LR_dir)
        self.Y_paths = sorted(
            self.make_dataset(self.args.LR_dir, float("inf"))) if self.args.LR_dir is not None else None
        self.HR_paths = sorted(
            self.make_dataset(self.args.HR_dir, float("inf"))) if self.args.HR_dir is not None else None

        self.trans_norm = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        for i in range(len(self.HR_paths)):
            file_name = os.path.basename(self.HR_paths[i])
            if (file_name.find("0401") >= 0):
                logging.info("We find the possion of NO. 401 in the HR patch NO. {}".format(i))
                self.HR_paths = self.HR_paths[:i]
                break
        for i in range(len(self.Y_paths)):
            file_name = os.path.basename(self.Y_paths[i])
            if (file_name.find("0401") >= 0):
                logging.info("We find the possion of NO. 401 in the LR patch NO. {}".format(i))
                self.Y_paths = self.Y_paths[i:]
                break

        self.Y_size = len(self.Y_paths)
        if self.train:
            self.load_size = self.args.load_size
            self.crop_size = self.args.crop_size
            self.upscale = self.args.upscale
            self.augment_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
            self.HR_transform = transforms.RandomCrop(int(self.crop_size * self.upscale))
            self.LR_transform = transforms.RandomCrop(self.crop_size)

    def __getitem__(self, index):
        """Get an item of the dataset according to the index.

        :param index: a random integer for data indexing
        :type index: int
        :return: returns a dictionary that contains A, B, and HR
        :rtype: dict, {'X': xx, 'Y': xx, 'HR': xx}
        """

        def pil_flow(img_paths, train):
            """Return pil image.

            :param img_paths: the images' paths
            :type img_paths: list
            :param train: if current mode is train or not
            :type train: bool
            :return: return a image list, in which per image is after some necessary image processing.
            :rtype:  list
            """
            imgs = [Image.open(img_path).convert('RGB') for img_path in img_paths]
            if not train:
                return imgs

            HR_img = self.augment_transform(imgs[1])
            imgs[1] = HR_img
            return imgs

        def tensor_flow(imgs):
            """Convert image with pil mode to tensor mode.

            :param imgs: List of images to be converted
            :type imgs: list
            :param useY: if using Y channel or not
            :type useY: bool
            :param weakly: if using weakly-paired setting or not
            :type weakly: bool
            :param useWhitening: if using whitening mode or not
            :type useWhitening: bool
            :return: list of tensors
            :rtype: list
            """
            imgs_tensor = []
            for i, img in enumerate(imgs):
                img_tensor = transforms.functional.to_tensor(img)
                imgs_tensor.append(img_tensor)
            return imgs_tensor

        if not self.train:
            index_Y = index
        else:
            index_Y = random.randint(0, self.Y_size - 1)
        Y_path = self.Y_paths[index_Y]
        img_paths = [Y_path]

        if self.HR_paths is not None:
            HR_path = self.HR_paths[index % len(self.HR_paths)]
            img_paths.append(HR_path)
        imgs = pil_flow(img_paths, self.train)
        imgs_tensor = tensor_flow(imgs)

        if len(imgs) > 1:
            if not self.train:
                return {'Y': imgs_tensor[0], 'HR': imgs_tensor[1]}
            else:
                HR_img = imgs[1]
                if self.upscale == 1:
                    X_img = HR_img
                else:
                    size = (int(HR_img.size[0] / self.upscale), int(HR_img.size[1] / self.upscale))
                    X_img = HR_img.resize(size, Image.BICUBIC)
                X_tensor = tensor_flow([X_img])
                return {'X': X_tensor[0], 'Y': imgs_tensor[0], 'HR': imgs_tensor[1]}
        elif len(imgs) == 1:
            return {'Y': imgs_tensor[0]}

    def __len__(self):
        """Return the total number of images in the dataset.

        :return: the length of the dataset
        :rtype: int
        """
        return len(self.HR_paths)
