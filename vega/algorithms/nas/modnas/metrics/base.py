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

"""Implementation of Metrics interface."""
from typing import Any
from modnas.utils.logging import get_logger


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
