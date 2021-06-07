# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Default DataProvider with dataloader."""
from ..base import DataProviderBase
from modnas.registry.data_provider import register


@register
class DefaultDataProvider(DataProviderBase):
    """Default DataProvider with dataloader."""

    def __init__(self, train_loader, valid_loader):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_iter = None
        self.valid_iter = None
        self.no_valid_warn = True
        self.reset_train_iter()
        self.reset_valid_iter()

    def get_next_train_batch(self):
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

    def get_next_valid_batch(self):
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

    def get_train_iter(self):
        """Return train iterator."""
        return self.train_iter

    def get_valid_iter(self):
        """Return validate iterator."""
        return self.valid_iter

    def reset_train_iter(self):
        """Reset train iterator."""
        self.train_iter = None if self.train_loader is None else iter(self.train_loader)

    def reset_valid_iter(self):
        """Reset validate iterator."""
        self.valid_iter = None if self.valid_loader is None else iter(self.valid_loader)

    def get_num_train_batch(self, epoch):
        """Return number of train batches in current epoch."""
        return 0 if self.train_loader is None else len(self.train_loader)

    def get_num_valid_batch(self, epoch):
        """Return number of validate batches in current epoch."""
        return 0 if self.valid_loader is None else len(self.valid_loader)
