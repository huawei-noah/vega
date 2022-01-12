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

"""This is a class for Spatiotemporal dataset."""
import os
import logging
import glob
import numpy as np
import pandas as pd
from vega.common import ClassFactory, ClassType
from vega.datasets.common.dataset import Dataset
from vega.datasets.conf.st import SpatiotemporalDatasetConfig
from sklearn.model_selection import train_test_split


@ClassFactory.register(ClassType.DATASET)
class SpatiotemporalDataset(Dataset):
    """Spatiotemporal Dataset."""

    config = SpatiotemporalDatasetConfig()

    __data_cache__ = None

    def __init__(self, **kwargs):
        super(SpatiotemporalDataset, self).__init__(**kwargs)
        if self.__data_cache__ is None:
            self._load_data()
        self.data = self.__data_cache__.get(self.mode)
        self._load_st_mx()
        self.inputs = self.data[:, :self.args.n_his, :, :]
        pred_frame = self.args.n_his + self.args.n_pred
        self.target = self.data[:, pred_frame - 1:pred_frame, :, :]

    def _load_data(self):
        data = self._read_from_csv(self.args.data_path)
        logging.info(data.describe())
        data = self._remove_cst_columns(data)
        self.num_features = data.shape[1]
        data_seq = data.values
        mean = np.mean(data_seq)
        std = np.std(data_seq)
        self.mean = mean
        self.std = std
        data = (data_seq - mean) / std
        data = self._fetch_time_series(data, self.args.n_his + self.args.n_pred)
        data = self._train_test_split(data)
        self.__data_cache__ = data

    def _load_st_mx(self):
        adj_mxs = self._load_adjacency_matrices()
        self.spatial_mx, self.temporal_mx = adj_mxs.get('adj_raw_sp_mx'), adj_mxs.get('adj_raw_fs_mx')

    def __len__(self):
        """Get the length of the dataset.

        :return: the length of the dataset
        :rtype: int
        """
        return len(self.data)

    def __getitem__(self, index):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: dict, {'data': xx, 'mask': xx, 'name': name}
        """
        return self.inputs[index], self.spatial_mx, self.temporal_mx, self.mean, self.std, self.target[index]

    def _read_from_csv(self, data_path, index=True):
        index_col_ = 0 if index else None
        return pd.read_csv(data_path, index_col=index_col_, engine='python')

    def _remove_cst_columns(self, df):
        std_dev_threshold = 0.05
        df = df.copy().loc[:, df.std() > std_dev_threshold]
        return df

    def _load_adjacency_matrices(self):
        adj_mxs = {}
        data_dir = os.path.dirname(self.args.data_path)
        for file in glob.glob(os.path.join(data_dir, 'graph_data', r'*.csv')):
            df = pd.read_csv(file, index_col=[0])
            name = os.path.basename(file).replace('.csv', '')
            adj_mxs[name] = df.values
        return adj_mxs

    def _train_test_split(self, data):
        train, test = train_test_split(data, random_state=40, train_size=1 - self.args.test_portion)
        train, val = train_test_split(train, random_state=40, train_size=self.args.train_portion)
        self._log_data_info('train dataset', train)
        self._log_data_info('val dataset', val)
        self._log_data_info('test dataset', test)
        return {'train': train, 'val': val, 'test': test}

    def _log_data_info(self, name, data):
        logging.info("{} shape:{}, max:{}, min:{}, mean:{}, std:{}".format(
            name, data.shape, data.max(), data.min(), data.mean(), data.std()))

    def _fetch_time_series(self, data_seq, n_frame):
        """Fetch time series data according to periods and freq."""
        n_slots = data_seq.shape[0] - n_frame
        num_features = data_seq.shape[1]
        tmp_seq = np.zeros((n_slots, n_frame, num_features, 1))
        for i in range(n_slots):
            tmp_seq[i, :, :, :] = np.reshape(data_seq[i:i + n_frame, :], [n_frame, num_features, 1])
        return tmp_seq
