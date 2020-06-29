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
import functools
import tensorflow as tf
from tensorflow.contrib.data.python.ops import threadpool
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import FileOps


@ClassFactory.register(ClassType.DATASET)
class Cifar10TF(object):
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

    def __init__(self, data_dir, batch_size, mode, num_parallel_batches=1,
                 repeat_num=5, padding=8, fp16=False, drop_remainder=False):
        """Init Cifar10TF."""
        self.data_dir = FileOps.download_dataset(data_dir)
        self.batch_size = batch_size
        self.mode = mode
        self.num_parallel_batches = num_parallel_batches
        self.repeat_num = repeat_num
        self.dtype = tf.float16 if fp16 is True else tf.float32
        self.num_channels = 3
        self.height = 32
        self.width = 32
        self.single_data_bytes = self.height * self.width * self.num_channels + 1
        self.num_images_train = 50000
        self.num_images_valid = 10000
        self.drop_remainder = drop_remainder
        self.single_data_size = [self.num_channels, self.height, self.width]
        self.padding = padding

    @property
    def data_files(self):
        """Cifar10 data files of type bin."""
        data_files = []
        if self.mode == 'train':
            for i in range(1, 6):
                data_files.append(os.path.join(self.data_dir, 'data_batch_{}.bin'.format(i)))
        elif self.mode == 'val':
            data_files.append(os.path.join(self.data_dir, 'test_batch.bin'))
        else:
            raise Exception('mode must be train or val.')
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

    def set_shapes(self, batch_size, images, labels):
        """Statically set the batch_size dimension."""
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size])))
        return images, labels

    @property
    def data_len(self):
        """Return dataset length of train or valid."""
        if self.mode == 'train':
            len = self.num_images_train // self.batch_size
        else:
            len = self.num_images_valid // self.batch_size
        return len

    def input_fn(self, params):
        """Define input_fn used by Tensorflow Estimator."""
        dataset = tf.data.FixedLengthRecordDataset(self.data_files, self.single_data_bytes)
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=self.num_images_train)
        dataset = dataset.repeat(self.repeat_num)
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(self.data_map_func,
                                          batch_size=self.batch_size,
                                          num_parallel_batches=self.num_parallel_batches,
                                          drop_remainder=self.drop_remainder)
        )
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        return dataset
