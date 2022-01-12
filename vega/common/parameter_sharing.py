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
"""This is Search on Network."""
import json
import logging
import hashlib
from threading import Lock
import vega
from vega.common import TaskOps, FileOps
from vega.common.utils import singleton

_lock = Lock()


def calculated_uuid(value):
    """Create uuid by static names."""
    value = str(json.dumps(value)) if isinstance(value, dict) else str(value)
    return hashlib.sha256(value.encode()).hexdigest()  # hash(value)


def add_share_file_path(uuid, file_name):
    """Share file path."""
    global _lock
    with _lock:
        ParameterSharing().__shared_params__[uuid] = file_name
        return uuid


def pop_share_file_path(uuid):
    """Pop Shared file path."""
    global _lock
    with _lock:
        cls = ParameterSharing()
        if not cls.__shared_params__:
            return None
        file_name = cls.__shared_params__.pop(uuid)
        result = FileOps.join_path(cls.sharing_dir, file_name)
        cls.__popped_files__.append(result)
        return result


@singleton
class ParameterSharing(object):
    """Parameter sharing class."""

    __shared_params__ = {}
    __popped_files__ = []

    def __init__(self):
        self.sharing_dir = FileOps.join_path(TaskOps().local_base_path, 'parameter_sharing')
        FileOps.make_dir(self.sharing_dir)

    def push(self, model, name):
        """Push state dict and save into files."""
        uuid = calculated_uuid(model.to_desc() if hasattr(model, "to_desc") else str(model))
        file_name = "{}_{}.{}".format(name, uuid, 'pth' if vega.is_torch_backend() else 'ckpt')
        saved_file_path = FileOps.join_path(self.sharing_dir, file_name)
        self._save(model, saved_file_path)
        add_share_file_path(uuid, saved_file_path)
        logging.info("push shared weight file uuid:{}".format(uuid))
        return saved_file_path

    def pop(self, desc):
        """Pop one file path."""
        if not self.__shared_params__:
            return
        uuid = calculated_uuid(desc)
        logging.info("pop shared weight file uuid:{}".format(uuid))
        return pop_share_file_path(uuid)

    def _save(self, model, file_name):
        if vega.is_torch_backend():
            import torch
            torch.save(model.state_dict(), file_name)
        elif vega.is_ms_backend():
            from mindspore.train.serialization import save_checkpoint
            save_checkpoint(model, file_name)

    def _remove(self, file_path):
        FileOps.remove(file_path)

    def remove(self):
        """Remove file has been popped."""
        while self.__popped_files__:
            self._remove(self.__popped_files__.pop())

    def clear(self):
        """Clear all shared params and remove files."""
        self.__shared_params__ = {}
        self.__popped_files__ = []
        self._remove(self.sharing_dir)
