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

"""This script is used to process the Avazu dataset."""
from __future__ import division
from __future__ import print_function
import logging
import os
import numpy as np
import torch


class BaseDataset():
    """This is the basic class of the AVAZU dataset.

    :param int block_size : raw data files will be partitioned into 'num_of_parts' blocks, each block has
    'block_size' samples
    :param int train_num_of_files: train raw data files will be partitioned into 'train_num_of_parts' blocks
    :param int test_num_files: test raw data files will be partitioned into 'test_num_of_parts' blocks
    :param int  train_size :the size of the train data, equal to train_pos_samples + train_neg_samples
    :param int test_size: the size of the test data, equal to test_pos_sample + test_neg_samples
    :param int pos_train_samples: the positive sampler number in the train set
    :param int pos_test_samples:  the positive sampler number in the test set
    :param int neg_train_samples:the negative sampler number in the train set
    :param int train_pos_ratio:the ratio of the positive sampler in the train set
    :param float test_pos_ratio:the ratio of the positive sampler in the test set
    :param bool initialized:decide whether to process (into hdf) or not
    :param intmax_length:different from # fields, in case some field has more than one value
    :param int num_of_feats:dimension of whole feature space
    :param list[str] feat_names: used as column names when creating hdf5 files
    :param int feat_min:sometimes independent feature maps are needed, i.e. each field has an independent feature map
    starting at index 0, feat_min is used to increment the index of feature maps and produce a unified
    index feature map
    :param int feat_sizes:sizes of feature maps,feat_min[i] = sum(feat_sizes[:i])
    :param str raw_data_dir:the original data is stored at raw_data_dir
    :param str feature_data_dir:raw_to_feature() will process raw data and produce libsvm-format feature files,
    and feature engineering is done here
    :param npy_data_dir:feature_to_npy() will convert feature files into npy tables, according to block_size
    """

    block_size = None
    train_num_of_files = 0
    test_num_of_files = 0
    train_size = 0
    test_size = 0

    train_pos_ratio = 0
    test_pos_ratio = 0
    initialized = 0
    max_length = 0
    num_of_feats = 0
    feat_names = None
    feat_min = None
    feat_sizes = None
    raw_data_dir = None
    npy_data_dir = None

    pos_train_samples = 0
    pos_test_samples = 0
    neg_train_samples = 0
    neg_test_samples = 0

    X_train = None
    y_train = None
    X_valid = None
    y_valid = None
    X_test = None
    y_test = None

    def summary(self):
        """Summarize the data set."""
        logging.info(self.__class__.__name__, 'data set summary:')
        logging.info('train set: ', self.train_size)
        logging.info('\tpositive samples: ', self.pos_train_samples)
        logging.info('\tnegative samples: ', self.neg_train_samples)
        logging.info('\tpositive ratio: ', self.train_pos_ratio)
        logging.info('test size:', self.test_size)
        logging.info('\tpositive samples: ', self.pos_test_samples)
        logging.info('\tnegative samples: ', self.neg_test_samples)
        logging.info('\tpositive ratio: ', self.test_pos_ratio)
        logging.info('input max length = %d, number of categories = %d' %
                     (self.max_length, self.num_of_feats))

    def _iterate_npy_files_(self, gen_type='train', num_of_files=None, shuffle_block=False, offset=0):
        """Iterate among hdf files(blocks). when the whole data set is finished, the iterator restarts.

        from the beginning, thus the data stream will never stop.
        :param gen_type: could be `train`, `valid`, or `test`. when gen_type=`train` or `valid`,
        this file iterator will go through the train set
        :type gen_type: str, optional
        :param num_of_parts: the file number, defaults to None
        :type num_of_parts: int
        :param shuffle_block:shuffle block files at every round, defaults to False
        :type shuffle_block: bool, optional
        :param offset: the start position to read, defaults to 0
        :type offset: int
        :return: input_hdf_file_name, output_hdf_file_name, finish_flag
        :rtype: tuple
        """
        gen_type = gen_type.lower()
        if num_of_files and offset >= num_of_files:
            raise ValueError("offset is supposed to be in range [0, %d)" % num_of_files)
        if gen_type == 'train' or gen_type == 'valid':
            file_prefix = 'train'
        elif gen_type == 'val' or gen_type == 'test':
            file_prefix = 'test'
        if num_of_files is None:
            yield os.path.join(self.npy_data_dir, file_prefix + '_input.npy'), \
                os.path.join(self.npy_data_dir, file_prefix + '_output.npy'), True
        else:
            logging.info("generating {} files from offset {}".format(gen_type, offset))
            parts = np.arange(num_of_files)[offset:]
            if shuffle_block:
                for i in range(int(shuffle_block)):
                    np.random.shuffle(parts)
            for i, p in enumerate(parts):
                yield os.path.join(self.npy_data_dir, file_prefix + '_input_part_' + str(p) + '.npy'), \
                    os.path.join(self.npy_data_dir, file_prefix + '_output_part_' + str(p) + '.npy'), \
                    i + 1 == len(parts)

    def batch_generator(self, gen_type='train', batch_size=None, pos_ratio=None, num_of_parts=None, val_ratio=None,
                        random_sample=False, shuffle_block=False, split_fields=False, on_disk=True):
        """Genetate a batch_size data.

        :param str gen_type: `train`, `valid`, or `test`.  the valid set is partitioned
        from train set dynamically, defaults to `train`
        :param int batch_size: batch_size, defaults to None
        :param float pos_ratio: default value is decided by the dataset, which means you
        don't want to change, defaults to None
        :param int  num_of_parts: file number, defaults to None
        :param float val_ratio: fraction of valid set from train set, defaults to None
        :param bool random_sample: if True, will shuffle, defaults to False
        :param bool shuffle_block: shuffle file blocks at every round, defaults to False
        :param bool split_fields: if True, returned values will be independently indexed,
        else using unified index, defaults to False
        :param bool on_disk: Whether the data is on disk or not, defaults to True
        """
        if batch_size is None:
            batch_size = max(int(1 / self.train_pos_ratio),
                             int(1 / self.test_pos_ratio)) + 1
        if val_ratio is None:
            val_ratio = 0.0
        gen_type = gen_type.lower()
        if num_of_parts is None:
            if gen_type == 'train' or gen_type == 'valid':
                if self.train_num_of_files is not None:
                    num_of_parts = self.train_num_of_files
            elif gen_type == 'val' or gen_type == 'test':
                if self.test_num_of_files is not None:
                    num_of_parts = self.test_num_of_files
        if on_disk:
            logging.info('on disk...')
            return self.process_data(gen_type, num_of_parts, shuffle_block, batch_size, random_sample)

    def process_data(self, gen_type, num_of_files, shuffle_block, batch_size, random_sample):
        """Process data on disk.

        :param str gen_type: `train`, `valid`, or `test`.  the valid set is partitioned
        from train set dynamically, defaults to `train`
        :param int batch_size: batch_size, defaults to None
        :param int  num_of_files: file number, defaults to None
        :param bool random_sample: if True, will shuffle, defaults to False
        :param bool shuffle_block: shuffle file blocks at every round, defaults to False
        """
        for f_in, f_out, block_finished in self._iterate_npy_files_(gen_type, num_of_files, shuffle_block):
            x_all = np.load(f_in)
            y_all = np.load(f_out)
            data_gen = self.generator(
                x_all, y_all, batch_size, shuffle=random_sample)
            finished = False

            while not finished:
                x, y, finished = next(data_gen)
                x_id = torch.LongTensor(x)
                y = torch.FloatTensor(y).squeeze(1)
                yield [x_id, y]

    @staticmethod
    def generator(X, y, batch_size, shuffle=True):
        """Batch data generator from in-memory array-like object `X` and `y`.

        :param X: 2d-array of feature_id and feature_val(optional).
        :type X: numpy array
        :param y: 1d-array of label
        :type y: numpy array
        :param batch_size: output batches' size
        :type batch_size: int
        :param shuffle: whether to shuffle during batch data generation, defaults to True
        :type shuffle: bool, optional
        :return: a batch of `X` and `y`, and a flag indicating whether `X` and `y` are exhausted
        :rtype: tuple
        """
        num_of_batches = np.ceil(1. * X.shape[0] / batch_size)
        counter = 0
        finished = False
        sample_index = np.arange(X.shape[0])
        if shuffle:
            for i in range(int(shuffle)):
                np.random.shuffle(sample_index)
        if X.shape[0] > 0:
            while True:
                batch_idx = sample_index[batch_size * counter:batch_size * (counter + 1)]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                counter += 1
                if counter == num_of_batches:
                    counter = 0
                    finished = True
                yield X_batch, y_batch, finished
        else:
            raise ValueError('Shape of data must be bigger than 0.')

    @staticmethod
    def split_pos_neg(X, y):
        """Split positive and negative samples from `X` and `y`.

        :param X: 2d-array of feature_id and feature_val(optional)
        :type X: numpy array
        :param y: 1d-array of label
        :type y: numpy array
        :return: splited `X` and `y`
        :rtype: same as `X` and `y`
        """
        pos_idx = (y == 1).reshape(-1)
        X_pos, y_pos = X[pos_idx], y[pos_idx]
        X_neg, y_neg = X[~pos_idx], y[~pos_idx]
        return X_pos, y_pos, X_neg, y_neg

    def __str__(self):
        """Construct method."""
        return self.__class__.__name__


class AVAZUDataset(BaseDataset):
    """This is the AVAZUDataset to genereate to hadle the dataset.

    :param dir_path: the data path, defaults to '../Avazu'
    :type dir_path: str
    """

    def __init__(self, dir_path='../Avazu'):
        """Construct method."""
        self.block_size = self.args.block_size
        self.train_num_of_files = self.args.train_num_of_files
        self.test_num_of_files = self.args.test_num_of_files
        self.train_size = self.args.train_size
        self.test_size = self.args.test_size
        self.pos_train_samples = self.args.pos_train_samples
        self.pos_test_samples = self.args.pos_test_samples
        self.neg_train_samples = self.args.neg_train_samples
        self.neg_test_samples = self.args.neg_test_samples
        self.train_pos_ratio = self.args.train_pos_ratio
        self.test_pos_ratio = self.args.test_pos_ratio
        self.initialized = self.args.initialized
        self.max_length = self.args.max_length
        self.num_of_feats = self.args.num_of_feats
        self.feat_names = self.args.feat_names
        self.feat_names = self.args.feat_names
        self.npy_data_dir = os.path.join(dir_path, 'npy/')
