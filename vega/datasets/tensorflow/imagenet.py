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

"""This is a class of ImageNet Dataset."""
import os
import tensorflow as tf
from official.r1.resnet.imagenet_preprocessing import preprocess_image
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.datasets.conf.imagenet import ImagenetConfig
from ..common.dataset import Dataset


@ClassFactory.register(ClassType.DATASET)
class Imagenet(Dataset):
    """This is a class for Imagenet dataset.

    :param data_dir: Imagenet data directory
    :type data_dir: str
    :param image_size: input imagenet size
    :type image_size: int
    :param batch_size: batch size
    :type batch_size: int
    :param mode: dataset mode, train or val
    :type mode: str
    :param fp16: whether to use fp16
    :type fp16: bool, default False
    :param num_parallel_batches: number of parallel batches
    :type num_parallel_batches: int, default 8
    :param drop_remainder: whether to drop data remainder
    :type drop_remainder: bool, default False
    :param transpose_input: whether to transpose input dimention
    :type transpose_input: bool, default false
    """

    config = ImagenetConfig()

    def __init__(self, **kwargs):
        """Init Cifar10."""
        super(Imagenet, self).__init__(**kwargs)
        self.data_path = FileOps.download_dataset(self.args.data_path)
        self.fp16 = self.args.fp16
        self.num_parallel_batches = self.args.num_parallel_batches
        self.image_size = self.args.image_size
        self.drop_remainder = self.args.drop_last
        if self.data_path == 'null' or not self.data_path:
            self.data_path = None
        self.num_parallel_calls = self.args.num_parallel_calls

    def _record_parser(self, raw_record):
        """Parse dataset function."""
        features_dict = {
            'image/encoded': tf.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        }
        parsed = tf.parse_single_example(raw_record, features_dict)
        image_buffer = parsed['image/encoded']
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        image = preprocess_image(image_buffer=image_buffer,
                                 bbox=bbox,
                                 output_height=self.image_size,
                                 output_width=self.image_size,
                                 num_channels=3,
                                 is_training=self.train)
        image = tf.cast(image, dtype=tf.float16 if self.fp16 else tf.float32)
        label = tf.cast(parsed['image/class/label'], dtype=tf.int32) - 1
        return image, label

    def _read_raw_data(self, data_file):
        """Read raw data."""
        dataset = tf.data.TFRecordDataset(data_file, buffer_size=8 * 1024 ** 2)
        return dataset

    def input_fn(self):
        """Define input_fn used by Tensorflow Estimator."""
        data_files = os.path.join(
            self.data_path, 'train/train-*' if self.mode == 'train' else 'val/val-*')
        dataset = tf.data.Dataset.list_files(data_files, shuffle=False)
        if self.world_size > 1:
            dataset = dataset.shard(self.world_size, self.rank)

        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=1024)
            dataset = dataset.repeat()

        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            self._read_raw_data, cycle_length=self.num_parallel_calls, sloppy=True))

        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                self._record_parser, batch_size=self.args.batch_size,
                num_parallel_batches=self.num_parallel_batches, drop_remainder=self.drop_remainder))

        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset
