# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Coco TFRecords dataset."""
import os
import tensorflow as tf
from official.vision.detection.dataloader import tf_example_decoder
from official.vision.detection.utils import box_utils
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import FileOps


@ClassFactory.register(ClassType.DATASET)
class CocoDataset(object):
    """This is a class for Coco TFRecords dataset.

    :param data_dir: Coco TFRecords data directory
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
        """Init CocoTF."""
        self.data_dir = FileOps.download_dataset(data_dir)
        self.batch_size = batch_size
        self.mode = mode
        self.num_parallel_batches = num_parallel_batches
        self.repeat_num = repeat_num
        self.dtype = tf.float16 if fp16 is True else tf.float32
        self.drop_remainder = drop_remainder
        self._include_mask = False
        self._dataset_fn = tf.data.TFRecordDataset

    @property
    def _file_pattern(self):
        """Coco data files of type TFRecords."""
        if self.mode == 'train':
            file_pattern = os.path.join(self.data_dir, 'coco_train.record-?????-of-00100')
        elif self.mode == 'val':
            file_pattern = os.path.join(self.data_dir, 'coco_val.record-?????-of-00100')
        elif self.mode == 'test':
            file_pattern = os.path.join(self.data_dir, 'coco_testdev.record-?????-of-00100')
        else:
            raise Exception('mode must be train, val or test.')
        return file_pattern

    def _parse_single_example(self, example):
        """Parse a single serialized tf.Example proto.

        Args:
        example: a serialized tf.Example proto string.
        Returns:
        A dictionary of groundtruth with the following fields:
            source_id: a scalar tensor of int64 representing the image source_id.
            height: a scalar tensor of int64 representing the image height.
            width: a scalar tensor of int64 representing the image width.
            boxes: a float tensor of shape [K, 4], representing the groundtruth
            boxes in absolute coordinates with respect to the original image size.
            classes: a int64 tensor of shape [K], representing the class labels of
            each instances.
            is_crowds: a bool tensor of shape [K], indicating whether the instance
            is crowd.
            areas: a float tensor of shape [K], indicating the area of each
            instance.
            masks: a string tensor of shape [K], containing the bytes of the png
            mask of each instance.
        """
        decoder = tf_example_decoder.TfExampleDecoder(
            include_mask=self._include_mask)
        decoded_tensors = decoder.decode(example)

        image = decoded_tensors['image']
        image_size = tf.shape(image)[0:2]
        boxes = box_utils.denormalize_boxes(
            decoded_tensors['groundtruth_boxes'], image_size)
        groundtruths = {
            'source_id': tf.string_to_number(
                decoded_tensors['source_id'], out_type=tf.int64),
            'height': decoded_tensors['height'],
            'width': decoded_tensors['width'],
            'num_detections': tf.shape(decoded_tensors['groundtruth_classes'])[0],
            'boxes': boxes,
            'classes': decoded_tensors['groundtruth_classes'],
            'is_crowds': decoded_tensors['groundtruth_is_crowd'],
            'areas': decoded_tensors['groundtruth_area'],
        }
        if self._include_mask:
            groundtruths.update({
                'masks': decoded_tensors['groundtruth_instance_masks_png'],
            })
        return groundtruths

    def input_fn(self, params):
        """Define input_fn used by Tensorflow Estimator."""
        dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                lambda filename: self._dataset_fn(filename).prefetch(1),
                cycle_length=32,
                sloppy=False))
        dataset = dataset.map(self._parse_single_example, num_parallel_calls=self.num_parallel_batches)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        return dataset
