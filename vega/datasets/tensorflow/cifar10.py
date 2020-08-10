# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Cifar10 dataset."""
import os
import tensorflow as tf
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import FileOps
from .dataset import Dataset
from vega.datasets.conf.cifar10 import Cifar10Config


@ClassFactory.register(ClassType.DATASET)
class Cifar10(Dataset):
    """This is a class for Cifar10TF dataset.

    :param data_dir: cifar10 data directory
    :type data_dir: str
    :param batch_size: batch size
    :type batch_size: int
    :param mode: dataset mode, train or val
    :type mode: str
    :param num_parallel_batches: number of parallel batches
    :type num_parallel_batches: int, default 1
    :param repeat_num: batch repeat number
    :param repeat_num: int, default 5
    :param padding: data padding in image preprocess
    :type padding: int, default 8
    :param fp16: whether to use fp16
    :type fp16: bool, default False
    :param drop_remainder: whether to drop data remainder
    :type drop_remainder: bool, default False
    """

    config = Cifar10Config()

    def __init__(self, **kwargs):
        """Init Cifar10."""
        super(Cifar10, self).__init__(**kwargs)
        self.data_path = FileOps.download_dataset(self.args.data_path)
        self.num_parallel_batches = self.args.num_parallel_batches
        self.train_portion = self.args.train_portion
        self.dtype = tf.float16 if self.args.fp16 is True else tf.float32
        self.num_channels = 3
        self.height = 32
        self.width = 32
        self.single_data_bytes = self.height * self.width * self.num_channels + 1
        self.num_images = self.args.num_images
        if self.train_portion != 1:
            if self.mode == 'train':
                self.num_images = int(self.num_images * self.train_portion)
            elif self.mode == 'val':
                self.num_images = int(self.args.num_images_train * (1 - self.train_portion))
        self.drop_remainder = self.args.drop_last
        self.single_data_size = [self.num_channels, self.height, self.width]
        if self.mode == 'train':
            self.padding = self.args.padding

    @property
    def data_files(self):
        """Cifar10 data files of type bin."""
        data_files = []
        is_training = self.mode == 'train' or self.mode == 'val' and self.train_portion != 1
        if is_training:
            for i in range(1, 6):
                data_files.append(os.path.join(self.data_path, 'data_batch_{}.bin'.format(i)))
        else:
            data_files.append(os.path.join(self.data_path, 'test_batch.bin'))
        return data_files

    def data_map_func(self, value):
        """CIFAR-10 data map function from raw data."""
        data_vector = tf.decode_raw(value, tf.uint8)
        input = self.generate_input(data_vector[1:self.single_data_bytes])
        label = self.generate_label(data_vector[0])
        return input, label

    def generate_input(self, input_vector):
        """Generate an input image from the raw vector."""
        raw_input = tf.reshape(input_vector, self.single_data_size)
        input = tf.cast(tf.transpose(raw_input, [1, 2, 0]), tf.float32)
        if self.mode == 'train':
            input = tf.image.resize_image_with_crop_or_pad(input,
                                                           self.height + self.padding,
                                                           self.width + self.padding)
            input = tf.random_crop(input, [self.height, self.width, self.num_channels])
            input = tf.image.random_flip_left_right(input)
        input = tf.image.per_image_standardization(input)
        input = tf.cast(input, self.dtype)
        return input

    def generate_label(self, label_vector):
        """Generate a label from the raw element."""
        label = tf.cast(label_vector, tf.int32)
        return label

    def split_train_data(self, dataset):
        """Split dataset."""

        def val_filter_func(x, y):
            return x >= self.args.num_images_train - self.num_images

        def train_filter_func(x, y):
            return x < self.num_images

        if self.mode == 'train':
            return dataset.enumerate().filter(train_filter_func).map(lambda x, y: y)
        elif self.mode == 'val':
            return dataset.enumerate().filter(val_filter_func).map(lambda x, y: y)

    def input_fn(self):
        """Define input_fn used by Tensorflow Estimator."""
        dataset = tf.data.FixedLengthRecordDataset(self.data_files, self.single_data_bytes)
        if self.train_portion != 1:
            dataset = self.split_train_data(dataset)
        if self.world_size > 1:
            dataset = dataset.shard(self.world_size, self.rank)
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=self.num_images)
            dataset = dataset.repeat()
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(self.data_map_func,
                                          batch_size=self.batch_size,
                                          num_parallel_batches=self.num_parallel_batches,
                                          drop_remainder=self.drop_remainder)
        )
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        return dataset
