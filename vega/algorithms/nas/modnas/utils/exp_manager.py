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

"""Experiment file manager."""
import os
import time
from typing import Optional
from .logging import get_logger


class ExpManager():
    """Experiment file manager class."""

    logger = get_logger('exp_manager')

    def __init__(self, name: str, root_dir: str = 'exp', subdir_timefmt: Optional[str] = None) -> None:
        if subdir_timefmt is None:
            root_dir = os.path.join(root_dir, name)
        else:
            root_dir = os.path.join(root_dir, name, time.strftime(subdir_timefmt, time.localtime()))
        self.root_dir = os.path.realpath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.logger.info('exp dir: {}'.format(self.root_dir))

    def subdir(self, *args) -> str:
        """Return subdir in current root dir."""
        subdir = os.path.join(self.root_dir, *args)
        os.makedirs(subdir, exist_ok=True)
        return subdir

    def join(self, *args) -> str:
        """Join root dir and subdir path."""
        return os.path.join(self.subdir(*args[:-1]), args[-1])
