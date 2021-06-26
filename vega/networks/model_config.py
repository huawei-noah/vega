# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
    num_classes = None
    getter = None

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
