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

"""Defined Conf for Pipeline."""
import os
import glob
from vega.common import ConfigSerializable
from vega.common import FileOps, Config
from vega.common.general import TaskConfig


class ModelConfig(ConfigSerializable):
    """Default Model config for Pipeline."""

    type = None
    model_desc = None
    model_desc_file = None
    pretrained_model_file = None
    head = None
    models_folder = None

    @classmethod
    def from_dict(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        t_cls = super(ModelConfig, cls).from_dict(data, skip_check)
        if data.get("models_folder") and not data.get('model_desc'):
            folder = data.models_folder.replace("{local_base_path}",
                                                os.path.join(TaskConfig.local_base_path, TaskConfig.task_id))
            pattern = FileOps.join_path(folder, "desc_*.json")
            desc_file = glob.glob(pattern)[0]
            t_cls.model_desc = Config(desc_file)
        elif data.get("model_desc_file") and not data.get('model_desc'):
            model_desc_file = data.get("model_desc_file").replace(
                "{local_base_path}", os.path.join(TaskConfig.local_base_path, TaskConfig.task_id))
            t_cls.model_desc = Config(model_desc_file)
        if data.get("pretrained_model_file"):
            pretrained_model_file = data.get("pretrained_model_file").replace(
                "{local_base_path}", os.path.join(TaskConfig.local_base_path, TaskConfig.task_id))
            t_cls.pretrained_model_file = pretrained_model_file
        return t_cls
