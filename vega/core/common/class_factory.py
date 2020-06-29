# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Management class registration and bind configuration properties, provides the type of class supported."""
import logging
from copy import deepcopy
from inspect import isfunction

from .config import Config
from .utils import update_dict
from .user_config import UserConfig, DefaultConfig


class ClassType(object):
    """Const class saved defined class type."""

    TRAINER = 'trainer'
    METRIC = 'trainer.metric'
    OPTIM = 'trainer.optim'
    LR_SCHEDULER = 'trainer.lr_scheduler'
    LOSS = 'trainer.loss'
    EVALUATOR = 'evaluator'
    GPU_EVALUATOR = 'evaluator.gpu_evaluator'
    HAVA_D_EVALUATOR = 'evaluator.hava_d_evaluator'
    DAVINCI_MOBILE_EVALUATOR = 'evaluator.davinci_mobile_evaluator'
    SEARCH_ALGORITHM = 'search_algorithm'
    SEARCH_SPACE = 'search_space'
    PIPE_STEP = 'pipe_step'
    GENERAL = 'general'
    HPO = 'hpo'
    DATASET = 'dataset'
    TRANSFORM = 'dataset.transforms'
    CALLBACK = 'trainer.callback'


class ClassFactory(object):
    """A Factory Class to manage all class in vega need to register with config."""

    __configs__ = None
    __registry__ = {}

    @classmethod
    def register(cls, type_name=ClassType.GENERAL, alias=None):
        """Register class into registry.

        :param type_name: type_name: type name of class registry
        :param alias: alias of class name
        :return: wrapper
        """

        def wrapper(t_cls):
            """Register class with wrapper function.

            :param t_cls: class need to register
            :return: wrapper of t_cls
            """
            t_cls_name = alias if alias is not None else t_cls.__name__
            if type_name not in cls.__registry__:
                cls.__registry__[type_name] = {t_cls_name: t_cls}
            else:
                if t_cls_name in cls.__registry__:
                    raise ValueError(
                        "Cannot register duplicate class ({})".format(t_cls_name))
                cls.__registry__[type_name].update({t_cls_name: t_cls})
            return t_cls

        return wrapper

    @classmethod
    def register_cls(cls, t_cls, type_name=ClassType.GENERAL, alias=None):
        """Register class with type name.

        :param t_cls: class need to register.
        :param type_name: type name.
        :param alias: class name.
        :return:
        """
        t_cls_name = alias if alias is not None else t_cls.__name__
        if type_name not in cls.__registry__:
            cls.__registry__[type_name] = {t_cls_name: t_cls}
        else:
            if t_cls_name in cls.__registry__:
                raise ValueError(
                    "Cannot register duplicate class ({})".format(t_cls_name))
            cls.__registry__[type_name].update({t_cls_name: t_cls})
        return t_cls

    @classmethod
    def is_exists(cls, type_name, cls_name=None):
        """Determine whether class name is in the current type group.

        :param type_name: type name of class registry
        :param cls_name: class name
        :return: True/False
        """
        if cls_name is None:
            return type_name in cls.__registry__
        return type_name in cls.__registry__ and cls_name in cls.__registry__.get(type_name)

    @classmethod
    def get_cls(cls, type_name, t_cls_name=None, bing_cfg=True):
        """Get class and bind config to class.

        :param type_name: type name of class registry
        :param t_cls_name: class name
        :return:t_cls
        """
        if not cls.is_exists(type_name, t_cls_name):
            raise ValueError("can't find class type {} class name {} in class registry".format(type_name, t_cls_name))
        # create instance without configs
        if t_cls_name is None:
            t_cls_type = cls.__configs__
            for _type_name in type_name.split('.'):
                t_cls_type = t_cls_type.get(_type_name)
            t_cls_name = t_cls_type.get('type')
        if t_cls_name is None:
            raise ValueError(
                "can't find class: {} with class type: {} in registry".format(t_cls_name, type_name))
        t_cls = cls.__registry__.get(type_name).get(t_cls_name)
        if bing_cfg:
            try:
                cls.bind_cfg(t_cls, type_name)
            except Exception as ex:
                logging.warning("Failed to bind cfg of class=%s in type_name=%s, ex=%s", t_cls.__name__, type_name,
                                str(ex))
        return t_cls

    @classmethod
    def set_current_step(cls, configs):
        """Set config's current step in a pipeline.

        :param configs: config dict need to set
        """
        if not isinstance(configs, dict):
            raise TypeError("configs should be a dict")
        cls.__configs__ = deepcopy(configs)

    @classmethod
    def bind_cfg(cls, t_cls, type_name):
        """Set the configuration information required for the creation of a class.

        :param t_cls: class instance need to bind
        :param type_name: type name in registry
        :return: t_cls
        """
        cls_dict = deepcopy(cls.get_cfg_dict(type_name, t_cls.__name__))
        if isfunction(t_cls) or type_name == ClassType.LOSS:
            cls_dict.pop('type')
            return t_cls(**cls_dict)
        else:
            if UserConfig().data is not None and 'general' in UserConfig().data and cls_dict is not None:
                cls_dict.update(UserConfig().data.general)
            # Merge config with base class
            cls_dict = cls.merge_base_cfg(cls_dict, type_name, t_cls)
            if not cls_dict:
                return t_cls
            setattr(t_cls, 'cfg', deepcopy(Config(cls_dict)))
            return t_cls

    @classmethod
    def get_cfg_dict(cls, type_name, name):
        """Get config dict from default and user config.

        Hierarchical configuration binding is supported if the type name is included.
        Indicates that other classes are applied to the class, and the system will customize the parser structure
        to bind the configuration to the corresponding class
        :param type_name: type name in registry
        :param name: class name
        :return: config dict of current cls
        """
        cls_dict = deepcopy(DefaultConfig().data.get(name))
        if cls.__configs__ is not None:
            current_type_dict = cls.__configs__.copy()
            current_type_name = None
            for _type_name in type_name.split('.'):
                if not current_type_dict or _type_name not in current_type_dict:
                    break
                current_type_dict = current_type_dict.get(_type_name)
                current_type_name = current_type_dict.get('type')
            if current_type_name == name or current_type_name == 'FineGrainedSpace':
                cls_dict = current_type_dict
        return cls_dict

    @classmethod
    def merge_base_cfg(cls, cls_dict, type_name, t_cls):
        """Merge config with parent class, configuration in the parent class can be inherited.

        If it is multiple inheritance, configuration will overwritten according to init sequence.
        Configuration with the same name will apply the first one.
        The parent class must also be registered in the ClassFactory.
        :param cls_dict: config dict of cls
        :param type_name: type name in registry of class
        :param t_cls: class instance
        :return: merged dict
        """
        base_cls_list = t_cls.__bases__
        merged_dict = deepcopy(cls_dict)
        for base_cls in base_cls_list:
            if base_cls and cls.is_exists(type_name, base_cls.__name__):
                base_cfg = deepcopy(cls.get_cfg_dict(type_name, base_cls.__name__))
                if not base_cfg:
                    continue
                exclude_keys = ['loss', 'metric', 'lr_scheduler', 'optim', 'model_desc', 'transforms']
                merged_dict = update_dict(cls_dict, base_cfg, exclude_keys)
        return merged_dict
