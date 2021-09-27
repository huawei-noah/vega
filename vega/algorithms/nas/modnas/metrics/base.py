# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Implementation of Metrics interface."""
from modnas.utils.logging import get_logger
from typing import Any


class MetricsBase():
    """Base Metrics class."""

    logger = get_logger('metrics')
    cur_estim = None

    def __init__(self) -> None:
        self.estim = MetricsBase.get_estim()

    def __call__(self, *args, **kwargs) -> Any:
        """Return metrics output."""
        raise NotImplementedError

    @staticmethod
    def get_estim() -> Any:
        """Get current Estimator."""
        return MetricsBase.cur_estim

    @staticmethod
    def set_estim(estim: Any) -> None:
        """Set current Estimator."""
        MetricsBase.cur_estim = estim
