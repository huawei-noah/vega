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

"""Default DataProvider with Iterable."""
from typing import List, Optional, Any, Collection, Iterator
from modnas.registry.data_provider import register
from ..base import DataProviderBase


@register
class DefaultDataProvider(DataProviderBase):
    """Default DataProvider with dataloader."""

    def __init__(self, train_loader: Collection, valid_loader: Optional[Collection]) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.no_valid_warn = True
        self.reset_train_iter()
        self.reset_valid_iter()

    def get_next_train_batch(self) -> List[Any]:
        """Return the next train batch."""
        if self.train_loader is None:
            self.logger.error('no train loader')
            return None
        try:
            trn_batch = next(self.get_train_iter())
        except StopIteration:
            self.reset_train_iter()
            trn_batch = next(self.get_train_iter())
        return trn_batch

    def get_next_valid_batch(self) -> List[Any]:
        """Return the next validate batch."""
        if self.valid_loader is None:
            if self.no_valid_warn:
                self.logger.warning('no valid loader, returning training batch instead')
                self.no_valid_warn = False
            return self.get_next_train_batch()
        try:
            val_batch = next(self.get_valid_iter())
        except StopIteration:
            self.reset_valid_iter()
            val_batch = next(self.get_valid_iter())
        return val_batch

    def get_train_iter(self) -> Iterator:
        """Return train iterator."""
        return self.train_iter or iter([])

    def get_valid_iter(self) -> Iterator:
        """Return validate iterator."""
        return self.valid_iter or iter([])

    def reset_train_iter(self) -> None:
        """Reset train iterator."""
        self.train_iter = None if self.train_loader is None else iter(self.train_loader)

    def reset_valid_iter(self) -> None:
        """Reset validate iterator."""
        self.valid_iter = None if self.valid_loader is None else iter(self.valid_loader)

    def get_num_train_batch(self, epoch: int) -> int:
        """Return number of train batches in current epoch."""
        return 0 if self.train_loader is None else len(self.train_loader)

    def get_num_valid_batch(self, epoch: int) -> int:
        """Return number of validate batches in current epoch."""
        return 0 if self.valid_loader is None else len(self.valid_loader)
