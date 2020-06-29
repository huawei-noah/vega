# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Contains Default and User configuration."""
import glob
import os
from copy import deepcopy
from .config import Config
from .task_ops import TaskOps
from .utils import singleton, update_dict


@singleton
class DefaultConfig(object):
    """Load and save default config from default yaml files."""

    __data__ = {}

    @classmethod
    def load(self, cfg_path):
        """Load default configs from vega/config dir.

        :param cfg_path: default yaml file path
        """
        files = glob.glob(cfg_path + r'/*/*.yml') + glob.glob(cfg_path + r'/*/*/*.yml')
        for _file in files:
            cfg_dict = Config(_file)
            if cfg_dict is None:
                continue
            DefaultConfig.__data__ = update_dict(cfg_dict, DefaultConfig.__data__)
        # set default task_id
        DefaultConfig.__data__['general']['task']['task_id'] = TaskOps.__task_id__
        # load default darts config
        darts_cifar10_template_file = os.path.join(cfg_path, "darts", "darts_cifar10.json")
        DefaultConfig.__data__["default_darts_cifar10_template"] = Config(darts_cifar10_template_file)
        darts_imagenet_template_file = os.path.join(cfg_path, "darts", "darts_imagenet.json")
        DefaultConfig.__data__["default_darts_imagenet_template"] = Config(darts_imagenet_template_file)

    @property
    def data(self):
        """Return cfg dict."""
        return Config(DefaultConfig.__data__)

    @data.setter
    def data(self, value):
        """Set cfg value."""
        DefaultConfig.__data__ = update_dict(value, DefaultConfig.__data__)


@singleton
class UserConfig(object):
    """Load user config from user file and merge config with default config."""

    __data__ = None

    def load(self, cfg_path):
        """Load config from file and merge config dict with default config.

        :param cfg_path: user config file path
        """
        if cfg_path is None:
            raise ValueError("config path can't be None or empty")
        self.__data__ = Config(cfg_path)
        defaults = DefaultConfig().data
        for pipe_name, child in self.__data__.items():
            if isinstance(child, dict):
                if pipe_name == 'pipeline':
                    continue
                elif pipe_name == 'general':
                    child.update(update_dict(child, Config(defaults.get('general'))))
                else:
                    for _, step_item in child.items():
                        UserConfig().merge_reference(step_item)
                    UserConfig().merge_default(child, defaults)
        if "general" not in self.__data__:
            self.__data__.general = defaults.get('general')

    @property
    def data(self):
        """Return cfg dict."""
        return self.__data__

    @staticmethod
    def merge_reference(child):
        """Merge config with reference the specified config with ref item."""
        if not isinstance(child, dict):
            return
        ref = child.get('ref')
        if not ref:
            return
        ref_dict = deepcopy(UserConfig().data)
        for key in ref.split('.'):
            ref_dict = ref_dict.get(key)
        exclude_keys = ['callbacks', 'lazy_built']
        for key in exclude_keys:
            if key in ref_dict:
                ref_dict.pop(key)
        ref_dict = update_dict(child, ref_dict)
        child = update_dict(ref_dict, child)

    @staticmethod
    def merge_default(child, defaults):
        """Merge default by class name and group name.

        :param child: child dict
        :param defaults: default dict
        """
        if not isinstance(child, dict):
            return
        for group_name, item in child.items():
            if group_name == 'type':
                continue
            if not isinstance(item, dict):
                continue
            cls_name = item.get('type')
            if cls_name is None:
                continue
            default_cfg = deepcopy(defaults.get(cls_name))
            if default_cfg is None:
                continue
            item.update(update_dict(item, default_cfg))
