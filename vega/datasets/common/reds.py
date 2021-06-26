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
import numpy as np
import random
from .dataset import Dataset
from vega.common import ClassFactory, ClassType
from vega.datasets.conf.reds import REDSConfig
from vega.datasets.common.utils.reds_util import read_file, imfrombytes, img2tensor, augment, paired_random_crop


@ClassFactory.register(ClassType.DATASET)
class REDS(Dataset):
    """This is a class for reds dataset."""

    config = REDSConfig()

    def __init__(self, **kwargs):
        """Construct the Cifar10 class."""
        Dataset.__init__(self, **kwargs)
        self.train = self.mode == 'train'
        self.gt_root, self.lq_root = self.args['dataroot_gt'], self.args['dataroot_lq']
        if self.args['num_frame'] % 2 != 1:
            raise Exception('num_frame should be odd number, but got {}'.format(self.args["num_frame"]))
        self.num_frame = self.args['num_frame']
        self.num_half_frames = self.args['num_frame'] // 2

        self.keys = []
        with open(self.args['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend(
                    [f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if self.train:
            if self.args['val_partition'] == 'REDS4':
                val_partition = ['000', '011', '015', '020']
            elif self.args['val_partition'] == 'official':
                val_partition = [f'{v:03d}' for v in range(240, 270)]
            else:
                raise ValueError(
                    f'Wrong validation partition {self.args["val_partition"]}.'
                    f"Supported ones are ['official', 'REDS4'].")
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # temporal augmentation configs
        self.interval_list = self.args.get('interval_list', [1])
        self.random_reverse = self.args.get('random_reverse', False)

    def __getitem__(self, index):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: array of numpy, array of numpy
        """
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)

        # determine the neighboring frames
        if self.train:
            interval = random.choice(self.interval_list)
        else:
            interval = 1

        # ensure not exceeding the borders
        frame_name = f'{center_frame_idx:08d}'
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        if self.train:
            # each clip has 100 frames starting from 0 to 99
            while (start_frame_idx < 0) or (end_frame_idx > 99):
                center_frame_idx = random.randint(0, 99)
                start_frame_idx = center_frame_idx - self.num_half_frames * interval
                end_frame_idx = center_frame_idx + self.num_half_frames * interval
            frame_name = f'{center_frame_idx:08d}'
            neighbor_list = list(
                range(center_frame_idx - self.num_half_frames * interval,
                      center_frame_idx + self.num_half_frames * interval + 1,
                      interval))
            # random reverse
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
        else:
            neighbor_list = []
            for i in range(start_frame_idx, end_frame_idx + 1):
                idx = i
                if i < 0:
                    idx = center_frame_idx + self.num_half_frames - i
                elif i > 99:
                    idx = (center_frame_idx - self.num_half_frames) - (i - 99)
                neighbor_list.append(idx)

        if len(neighbor_list) != self.num_frame:
            raise Exception('Wrong length of neighbor list: {}'.format(len(neighbor_list)))

        # get the GT frame (as the center frame)
        img_gt_path = os.path.join(self.gt_root, clip_name, f'{frame_name}.png')
        img_bytes = read_file(img_gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            img_lq_path = os.path.join(self.lq_root, clip_name, f'{neighbor:08d}.png')
            img_bytes = read_file(img_lq_path)
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        if self.train:
            scale = self.args['scale']
            gt_size = self.args['gt_size']
            # randomly crop
            img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale,
                                                 img_gt_path)

            # augmentation - flip, rotate
            img_lqs.append(img_gt)
            img_results = augment(img_lqs, self.args['use_flip'], self.args['use_rot'])
            img_lqs, img_gt = img_results[0:-1], img_results[-1]

        img_lqs = img2tensor(img_lqs)
        img_lqs = np.stack(img_lqs, axis=0)
        img_gt = img2tensor(img_gt)

        return img_lqs, img_gt

    def __len__(self):
        """Get the length of dataset."""
        return len(self.keys)
