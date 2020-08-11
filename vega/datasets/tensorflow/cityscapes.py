# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class of ImageNet Dataset."""
import os.path as osp
import tensorflow as tf
from .. import transforms
from vega.core.common.config import obj2config
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import FileOps
from .dataset import Dataset
from vega.datasets.conf.city_scapes import CityscapesConfig
import cv2


@ClassFactory.register(ClassType.DATASET)
class Cityscapes(Dataset):
    """This is a class for CityscapesTF dataset.

    :param data_dir: Cityscapes data directory
    :type data_dir: str
    :param image_size: input image size
    :type image_size: int
    :param batch_size: batch size
    :type batch_size: int
    :param mode: dataset mode, train or val
    :type mode: str
    :param drop_remainder: whether to drop data remainder
    :type drop_remainder: bool, default False
    """

    config = CityscapesConfig()

    def __init__(self, **kwargs):
        super(Cityscapes, self).__init__(**kwargs)
        config = obj2config(getattr(self.config, self.mode))
        config.update(self.args)
        self.args = config
        self.root_dir = self.args['root_dir']
        self.image_size = self.args.Rescale.size
        self.list_file = self.args.list_file
        self.batch_size = self.args.get('batch_size', 1)
        self.num_parallel_batches = self.args.get('num_parallel_batches', 1)
        self.drop_remainder = self.args.get('drop_remainder', False)

        self.transforms = self._init_transforms()
        self.root_dir = FileOps.download_dataset(self.root_dir)
        self._init_data_files()

    def _init_data_files(self):
        """Initialize data settings."""
        with open(osp.join(self.root_dir, self.list_file)) as f:
            lines = f.readlines()
        self.data_files = [None] * len(lines)
        self.label_files = [None] * len(lines)
        for i, line in enumerate(lines):
            data_file_name, label_file_name = line.strip().split()
            self.data_files[i] = osp.join(self.root_dir, data_file_name)
            self.label_files[i] = osp.join(self.root_dir, label_file_name)
        self.num_images = len(lines)

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

    def _read_image_label(self, datafile, labelfile):
        """Read image and label from files."""
        image = cv2.imread(datafile.decode(), cv2.IMREAD_COLOR)
        label = cv2.imread(labelfile.decode(), cv2.IMREAD_GRAYSCALE)
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label

    def _resize_function(self, image, label):
        """Resize image and label to setting size."""
        label = tf.expand_dims(label, -1)
        image.set_shape([None, None, 3])
        label.set_shape([None, None, 1])
        image = tf.image.resize_images(image, [self.image_size, self.image_size])
        label = tf.image.resize_images(label, [self.image_size, self.image_size])
        label = tf.squeeze(label, [-1])
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int32)
        return image, label

    def data_map_func(self, datafile, labelfile):
        """Map function from original files to tf dataset."""
        data_list = tf.py_func(self._read_image_label, [datafile, labelfile], [tf.uint8, tf.uint8])
        return self._resize_function(data_list[0], data_list[1])

    def input_fn(self):
        """Define input_fn used by Tensorflow Estimator."""
        dataset = tf.data.Dataset.from_tensor_slices((self.data_files, self.label_files))
        if self.world_size > 1:
            dataset = dataset.shard(self.world_size, self.rank)
        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=self.num_images)
            dataset = dataset.repeat()

        dataset = dataset.map(self.data_map_func)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset
