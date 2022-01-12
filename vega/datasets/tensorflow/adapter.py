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

"""This is a base class of the dataset."""

import logging
import tensorflow as tf
import vega
from vega.common.general import General


class TfAdapter(object):
    """This is the base class of the dataset, which is a subclass of `TaskOps`.

    The Dataset provide several basic attribute like dataloader, transform and sampler.
    """

    dtype_map = {"torch.float32": tf.float32,
                 "float32": tf.float32,
                 "torch.float16": tf.float32,
                 "float16": tf.float32,
                 "float64": tf.double,
                 "torch.int32": tf.int32,
                 "int32": tf.int32,
                 "torch.int64": tf.int64,
                 "int64": tf.int64,
                 "int": tf.int64}

    def __init__(self, dataset):
        self.dataset = dataset
        self.args = dataset.args
        self._num_examples = len(self.dataset) if hasattr(self.dataset, "__len__") else self.args.get('num_images')
        self.data_index = list(range(self._num_examples))
        if self.args.get('train_portion', 1.0) < 1:
            split = int(self.args.train_portion * self._num_examples)
            if self.dataset.mode == 'train':
                self.data_index = self.data_index[:split]
                self._num_examples = split
            elif self.dataset.mode == 'val':
                self.data_index = self.data_index[split:]
                self._num_examples = self._num_examples - split
        self.repeat_ratio = self.args.get('repeat_ratio', 1.)
        self.is_detection = self.args.get("is_detection", False)
        self.is_spatiotemporal = self.args.get('is_spatiotemporal')

    def _get_dataset_info(self):
        """Get the data shape."""
        if self.is_detection:
            return
        item = self.dataset[0]
        if self.is_spatiotemporal:
            self.feature_shape = [v.shape if v is not None else v for v in item]
        if isinstance(item, (list, tuple)):
            self.image_pos, self.label_pos = 0, 1
        elif isinstance(item, dict):
            keys = list(item.keys())
            self.image_pos, self.label_pos = keys[0], keys[1]
        else:
            raise ValueError
        image = item[self.image_pos]
        label = item[self.label_pos]
        self.fixed_size = self.args.get("fixed_size", True)
        self.data_format = General.data_format
        self.image_shape = list(image.shape)
        try:
            self.label_shape = list(label.shape)
        except Exception:
            self.label_shape = 1

        try:
            self.image_dtype = str(image.dtype)
        except Exception:
            logging.debug('Falied to get image dtype.')
        try:
            self.label_dtype = str(label.dtype)
        except Exception:
            self.label_dtype = "int"

        self.image_dtype_tf = self.dtype_map[self.image_dtype]
        self.label_dtype_tf = self.dtype_map[self.label_dtype]

    def _get_item(self, images_index, label_index):
        """Get one item of the dataset."""
        item = self.dataset[images_index]
        if self.is_spatiotemporal:
            return item[0], item[1], item[2], item[3], item[4], item[5]
        if not self.is_detection:
            image = item[self.image_pos]
            label = item[self.label_pos]
            return image, label
        else:
            image = item[0]
            img_meta = image.get("img_meta")
            return image.get("img"), image.get("gt_bboxes"), image.get("gt_bboxes_ignore"), \
                image.get("gt_labels_ignore"), image.get("gt_labels"), \
                img_meta.get("ori_shape"), img_meta.get("img_shape"), \
                img_meta.get("pad_shape"), img_meta.get("scale_factor"), \
                img_meta.get("flip"), item[1]

    def _resize_image_label(self, image, label):
        """Resize the image and label."""
        if len(self.image_shape) == 3:
            img_channel = self.image_shape[0]
            image.set_shape([img_channel, None, None])
        elif len(self.image_shape) == 2:
            img_channel = 1
            image.set_shape([img_channel, None, None])
        else:
            image.set_shape(self.image_shape)

        if self.label_shape == 1:
            label.set_shape(self.label_shape)
        elif len(self.label_shape) == 3:
            label_channel = self.label_shape[0]
            label.set_shape([label_channel, None, None])
        else:
            label_channel = 1
            label.set_shape([label_channel, None, None])

        return image, label

    def data_map_func(self, images_index, label_index):
        """Apply data map function from raw data."""
        if self.is_spatiotemporal:
            feature, spatial_mx, temporal_mx, mean, std, label = tf.numpy_function(
                self._get_item, [images_index, label_index],
                [tf.float64, tf.float64, tf.float64, tf.float64, tf.float64, tf.float64])
            feature.set_shape(self.feature_shape[0])
            spatial_mx.set_shape(self.feature_shape[1])
            temporal_mx.set_shape(self.feature_shape[2])
            label.set_shape(self.feature_shape[-1])
            return (feature, spatial_mx, temporal_mx), (mean, std, label)
        if not self.is_detection:
            image, label = tf.numpy_function(self._get_item,
                                             [images_index, label_index],
                                             [self.image_dtype_tf, self.label_dtype_tf])
            if self.fixed_size:
                image.set_shape(self.image_shape)
                label.set_shape(self.label_shape)
            else:
                image, label = self._resize_image_label(image, label)

            try:
                label = tf.squeeze(label)
            except Exception:
                logging.debug('Falied to get label.')
            if self.label_dtype == "int":
                label = tf.cast(label, tf.int32)
            if self.data_format == "channels_last":
                try:
                    image = tf.transpose(image, [1, 2, 0])
                    label = tf.transpose(label, [1, 2, 0])
                except Exception:
                    logging.debug('Falied to transpose.')
        else:
            img, gt_bboxes, gt_bboxes_ignore, gt_labels_ignore, gt_labels, \
                ori_shape, img_shape, pad_shape, scale_factor, flip, target = tf.numpy_function(
                    self._get_item,
                    [images_index, label_index],
                    [tf.float32, tf.float32, tf.float32, tf.float32, tf.int64,
                        tf.int64, tf.int64, tf.int64, tf.float64, tf.bool, tf.int64])
            image = dict()
            img_meta = dict()
            img_meta["ori_shape"] = ori_shape
            img_meta["img_shape"] = img_shape
            img_meta["pad_shape"] = pad_shape
            img_meta["scale_factor"] = scale_factor
            img_meta["flip"] = flip
            image["img"] = img
            image["gt_bboxes"] = gt_bboxes
            image["gt_bboxes_ignore"] = gt_bboxes_ignore
            image["gt_labels"] = gt_labels
            image["gt_labels_ignore"] = gt_labels_ignore
            image["img_meta"] = img_meta
            label = target

        return image, label

    def __len__(self):
        """Return dataset length of train or valid."""
        if self.dataset.mode == 'train':
            len = self._num_examples // self.args.batch_size
            if self.dataset.world_size > 1:
                len = len // self.dataset.world_size
            len = int(len * self.repeat_ratio)
        else:
            len = self._num_examples // self.args.batch_size
        return len

    def input_fn(self):
        """Return the next `batch_size` examples from this data set."""
        if hasattr(self.dataset, "input_fn"):
            return self.dataset.input_fn()
        self._get_dataset_info()
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data_index, self.data_index))
        if self.dataset.mode == 'train' and self.dataset.world_size > 1:
            dataset = dataset.shard(self.dataset.world_size, self.dataset.rank)
        if self.dataset.mode == 'train':
            dataset = dataset.repeat()
        if self.args.shuffle:
            dataset = dataset.shuffle(buffer_size=self._num_examples)

        if vega.is_npu_device():
            # esr cannot adapt to num_parallel_calls on NPU
            dataset = dataset.map(self.data_map_func)
            dataset = dataset.batch(
                batch_size=self.args.batch_size, drop_remainder=self.args.drop_last)
        else:
            dataset = dataset.map(self.data_map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(
                batch_size=self.args.batch_size, drop_remainder=self.args.drop_last)
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset

    @property
    def loader(self):
        """Dataloader arrtribute which is a unified interface to generate the data.

        :return: a batch data
        :rtype: dict, list, optional
        """
        return self
