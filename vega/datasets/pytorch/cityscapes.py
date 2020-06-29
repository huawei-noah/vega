# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is the class of Cityscapes dataset."""
import os
import os.path as osp
import cv2
import torch
import numpy as np
import glob
import pickle
from . import transforms
from .common.dataset import Dataset
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps


@ClassFactory.register(ClassType.DATASET)
class Cityscapes(Dataset):
    """Class of Cityscapes dataset, which is subclass of Dateset.

    Two types of data are supported:
        1) Image with extensions in 'jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP'
        2) pkl with extensions in 'pkl', 'pt', 'pth'. Image pkl should be in format of HWC, with bgr as the channels
    To use this dataset, provide either: 1) data_dir and label_dir; or 2) root_dir and list_file
    :param train: if the mdoe is train or false, defaults to True
    :type train: bool, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    def __init__(self, **kwargs):
        """Construct the Cityscapes class."""
        super(Cityscapes, self).__init__(**kwargs)
        self.dataset_init()

    def _init_transforms(self):
        """Initialize transforms."""
        result = list()
        if "Rescale" in self.args:
            import logging
            logging.info(str(dict(**self.args.Rescale)))
            result.append(transforms.Rescale_pair(**self.args.Rescale))
        if "RandomMirror" in self.args and self.args.RandomMirror:
            result.append(transforms.RandomHorizontalFlip_pair())
        if "RandomColor" in self.args:
            result.append(transforms.RandomColor_pair(**self.args.RandomColor))
        if "RandomGaussianBlur" in self.args:
            result.append(transforms.RandomGaussianBlur_pair(**self.args.RandomGaussianBlur))
        if "RandomRotation" in self.args:
            result.append(transforms.RandomRotate_pair(**self.args.RandomRotation))
        if "Normalization" in self.args:
            result.append(transforms.Normalize_pair(**self.args.Normalization))
        if "RandomCrop" in self.args:
            result.append(transforms.RandomCrop_pair(**self.args.RandomCrop))
        return result

    def dataset_init(self):
        """Construct method.

        If both data_dir and label_dir are provided, then use data_dir and label_dir
        Otherwise use root_dir and list_file.
        """
        if "data_dir" in self.args and "label_dir" in self.args:
            self.data_files = sorted(glob.glob(osp.join(self.args.data_dir, "*")))
            self.label_files = sorted(glob.glob(osp.join(self.args.label_dir, "*")))
        else:
            if "root_dir" not in self.args or "list_file" not in self.args:
                raise Exception("You must provide a root_dir and a list_file!")
            with open(osp.join(self.args.root_dir, self.args.list_file)) as f:
                lines = f.readlines()
            self.data_files = [None] * len(lines)
            self.label_files = [None] * len(lines)
            for i, line in enumerate(lines):
                data_file_name, label_file_name = line.strip().split()
                self.data_files[i] = osp.join(self.args.root_dir, data_file_name)
                self.label_files[i] = osp.join(self.args.root_dir, label_file_name)

        datatype = self._get_datatype()
        if datatype == "image":
            self.read_fn = self._read_item_image
        else:
            self.read_fn = self._read_item_pickle

    def __len__(self):
        """Get the length of the dataset.

        :return: the length of the dataset
        :rtype: int
        """
        return len(self.data_files)

    def __getitem__(self, index):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: dict, {'data': xx, 'mask': xx, 'name': name}
        """
        image, label = self.read_fn(index)
        image_name = self.data_files[index].split("/")[-1].split(".")[0]
        image, label = self.transforms(image, label)
        item = {'data': image, 'mask': label, "name": image_name}
        return self._numpy_to_tensor(item)

    @staticmethod
    def _numpy_to_tensor(item):
        """Convert data and mask from numpy to tensor.

        :param item: an item of dataset in np.array
        :type item: {'data': np.float, 'mask': np.uint8}
        :return: an item of dataset in torch.tensor
        :rtype dict: {'data': torch.float32, 'mask': torch.long}
        """
        item['data'] = torch.from_numpy(np.transpose(item['data'], [2, 0, 1])).float()
        item['mask'] = torch.from_numpy(item['mask']).long()
        return item

    @staticmethod
    def _get_datatype_files(file_paths):
        """Check file extensions in file_paths to decide whether they are images or pkl.

        :param file_paths: a list of file names
        :type file_paths: list of str
        :return image, pkl or None according to the type of files
        :rtype: str
        """
        IMG_EXTENSIONS = {'jpg', 'JPG', 'jpeg', 'JPEG',
                          'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP'}
        PKL_EXTENSIONS = {'pkl', 'pt', 'pth'}

        file_extensions = set(data_file.split('.')[-1] for data_file in file_paths)
        if file_extensions.issubset(IMG_EXTENSIONS):
            return "image"
        elif file_extensions.issubset(PKL_EXTENSIONS):
            return "pkl"
        else:
            raise Exception("Invalid file extension")

    def _get_datatype(self):
        """Check the datatype of all data.

        :return image, pkl or None
        :rtype: str
        """
        type_data = self._get_datatype_files(self.data_files)
        type_labels = self._get_datatype_files(self.label_files)

        if type_data == type_labels:
            return type_data
        else:
            raise Exception("Images and masks must be both image or pkl!")

    def _read_item_image(self, index):
        """Read image and label in "image" format.

        :param index: index
        :type index: int
        :return: image in np.array, HWC, bgr; label in np.array, HW
        :rtype: tuple of np.array
        """
        image = cv2.imread(self.data_files[index], cv2.IMREAD_COLOR)
        label = cv2.imread(self.label_files[index], cv2.IMREAD_GRAYSCALE)
        return image, label

    def _read_item_pickle(self, index):
        """Read image and label in "pkl" format.

        :param index: index
        :type index: int
        :return: image in np.array, HWC, bgr; label in np.array, HW
        :rtype: tuple of np.array
        """
        with open(self.data_files[index], "rb") as file:
            image = pickle.load(file)
        with open(self.label_files[index], "rb") as file:
            label = pickle.load(file)
        return image, label
