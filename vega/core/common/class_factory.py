# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Management class registration and bind configuration properties, provides the type of class supported."""
from copy import deepcopy
from inspect import isfunction, isclass


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
    DATASET = 'dataset'
    TRANSFORM = 'dataset.transforms'
    CALLBACK = 'trainer.callback'
    CONFIG = 'CONFIG'
    CODEC = 'search_algorithm.codec'


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
    def register_from_package(cls, package, type_name=ClassType.GENERAL):
        """Register all public class from package.

        :param t_cls: class need to register.
        :param type_name: type name.
        :param alias: class name.
        :return:
        """
        for _name in dir(package):
            if _name.startswith("_"):
                continue
            _cls = getattr(package, _name)
            if not isclass(_cls) and not isfunction(_cls):
                continue
            ClassFactory.register_cls(_cls, type_name)

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
    def get_cls(cls, type_name, t_cls_name=None):
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
        return t_cls

    @classmethod
    def get_instance(cls, type_name, params=None):
        """Get instance."""
        try:
            if not params:
                return
            t_cls_name = params.pop('type')
            t_cls = cls.get_cls(type_name, t_cls_name)
            return t_cls(**params) if params else t_cls()
        except Exception as ex:
            raise Exception("Can't get instance for params:{}, ex={}".format(params, ex))

    @classmethod
    def set_current_step(cls, configs):
        """Set config's current step in a pipeline.

        :param configs: config dict need to set
        """
        if not isinstance(configs, dict):
            raise TypeError("configs should be a dict")
        cls.__configs__ = deepcopy(configs)
