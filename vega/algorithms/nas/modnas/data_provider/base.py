# -*- coding:utf-8 -*-

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

"""Base DataProvider."""
from modnas.utils.logging import get_logger


class DataProviderBase():
    """Base DataProvider class."""

    logger = get_logger('data_provider')

    def get_next_train_batch(self):
        """Return the next train batch."""
        return next(self.get_train_iter())

    def get_next_valid_batch(self):
        """Return the next validate batch."""
        return next(self.get_valid_iter())

    def get_train_iter(self):
        """Return train iterator."""
        raise NotImplementedError

    def get_valid_iter(self):
        """Return validate iterator."""
        raise NotImplementedError

    def reset_train_iter(self):
        """Reset train iterator."""
        raise NotImplementedError

    def reset_valid_iter(self):
        """Reset validate iterator."""
        raise NotImplementedError

    def get_num_train_batch(self, epoch: int):
        """Return number of train batches in current epoch."""
        raise NotImplementedError

    def get_num_valid_batch(self, epoch: int):
        """Return number of validate batches in current epoch."""
        raise NotImplementedError
