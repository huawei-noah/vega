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
from enum import Enum
from inspect import isfunction, isclass, signature as sig

logger = logging.getLogger(__name__)


class ClassType(object):
    """Const class saved defined class type."""

    TRAINER = 'trainer'
    METRIC = 'trainer.metric'
    OPTIMIZER = 'trainer.optimizer'
    LR_SCHEDULER = 'trainer.lr_scheduler'
    LOSS = 'trainer.loss'
    EVALUATOR = 'evaluator'
    HOST_EVALUATOR = 'evaluator.host_evaluator'
    DEVICE_EVALUATOR = 'evaluator.device_evaluator'
    SEARCH_ALGORITHM = 'search_algorithm'
    PIPE_STEP = 'pipe_step'
    GENERAL = 'general'
    DATASET = 'dataset'
    TRANSFORM = 'dataset.transforms'
    CALLBACK = 'trainer.callback'
    CONFIG = 'CONFIG'
    CODEC = 'search_algorithm.codec'
    QUOTA = 'quota'
    NETWORK = "network"
    PRETRAINED_HOOK = 'pretrained_hook'
    SEARCHSPACE = 'searchspace'
    PACKAGE = "package"
    GENERATOR = "generator"


class SearchSpaceType(Enum):
    """Const class saved defined Search Space type."""

    CONNECTIONS = 'connections'

    @classmethod
    def contains(cls, item):
        """Use the contains method to replace the in operation."""
        for _item in cls:
            if isinstance(item, str):
                if _item.value == item:
                    return True
            else:
                if _item.value == item.value:
                    return True
        return False


class ClassFactory(object):
    """A Factory Class to manage all class need to register with config."""

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
            if SearchSpaceType.contains(type_name):
                cls.register_cls(t_cls, ClassType.NETWORK, t_cls_name)
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
        registered = type_name in cls.__registry__ and cls_name in cls.__registry__.get(type_name)
        if not registered:
            cls._import_pkg(type_name, cls_name)
        registered = type_name in cls.__registry__ and cls_name in cls.__registry__.get(type_name)
        return registered

    @classmethod
    def _import_pkg(cls, type_name, cls_name):
        type_cls = "{}:{}".format(type_name, cls_name)
        pkg = cls.__registry__.get(ClassType.PACKAGE).get(
            type_cls) or cls.__registry__.get(ClassType.PACKAGE).get(cls_name)
        if pkg:
            __import__(pkg)

    @classmethod
    def get_cls(cls, type_name, t_cls_name=None):
        """Get class and bind config to class.

        :param type_name: type name of class registry
        :param t_cls_name: class name
        :return:t_cls
        """
        if t_cls_name is None:
            from vega.datasets.conf.dataset import DatasetConfig
            from vega.evaluator.conf import EvaluatorConfig
            if type_name == ClassType.DATASET:
                t_cls_name = DatasetConfig.type
            elif type_name == ClassType.EVALUATOR:
                t_cls_name = EvaluatorConfig.type
            else:
                pass
        if t_cls_name is None:
            raise ValueError("can't find class. class type={}".format(type_name))
        # lazy load class
        if not cls.is_exists(type_name, t_cls_name) and t_cls_name is not None:
            cls._import_pkg(type_name, t_cls_name)
        # verify class
        if not cls.is_exists(type_name, t_cls_name):
            raise ValueError("can't find class type {} class name {} in class registry".format(type_name, t_cls_name))
        # get class
        t_cls = cls.__registry__.get(type_name).get(t_cls_name)
        return t_cls

    @classmethod
    def get_instance(cls, type_name, params=None, **kwargs):
        """Get instance."""
        _params = deepcopy(params)
        if not _params:
            return
        if isinstance(_params, str):
            _params = dict(type=_params)
        t_cls_name = _params.pop('type')
        if kwargs:
            _params.update(kwargs)
        t_cls = cls.get_cls(type_name, t_cls_name)
        if type_name != ClassType.NETWORK:
            return t_cls(**_params) if _params else t_cls()
        # remove extra params
        params_sig = sig(t_cls.__init__).parameters
        instance = cls._create_instance_params(params_sig, _params, t_cls)
        if not instance:
            extra_param = {k: v for k, v in _params.items() if k not in params_sig}
            _params = {k: v for k, v in _params.items() if k not in extra_param}
            try:
                instance = t_cls(**_params) if _params else t_cls()
            except Exception as ex:
                logging.error("Failed to create instance:{}".format(t_cls))
                raise ex
            for k, v in extra_param.items():
                setattr(instance, k, v)
        return instance

    @classmethod
    def _create_instance_params(cls, params_sig, _params, t_cls):
        try:
            has_args = any('*' in str(v) and not str(v).startswith('**') for v in params_sig.values())
            has_kwargs = any('**' in str(v) for v in params_sig.values())
            if has_args and not has_kwargs:
                return t_cls(*list(_params.values())) if list(_params.values()) else t_cls()
            if not has_args and has_kwargs:
                return t_cls(**_params) if _params else t_cls()
            if has_args and has_kwargs:
                if _params and list(_params.values()):
                    return t_cls(*list(_params.values()), **_params)
                if _params and not list(_params.values()):
                    return t_cls(**_params)
                if not _params and list(_params.values()):
                    return t_cls(*list(_params.values()))
                return t_cls()
        except Exception as ex:
            logging.error("Failed to create instance:{}".format(t_cls))
            raise ex

    @classmethod
    def lazy_register(cls, base_pkg, pkg_cls_dict):
        """Get instance."""
        for pkg, classes in pkg_cls_dict.items():
            for _cls in classes:
                cls.register_cls("{}.{}".format(base_pkg, pkg), ClassType.PACKAGE, _cls)
