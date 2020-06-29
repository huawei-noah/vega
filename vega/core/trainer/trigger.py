# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Compatibility code to be able to use extend the ability of the trainer."""

from functools import wraps
from vega.core.common.class_factory import ClassFactory


@ClassFactory.register('trainer.trigger')
class Trigger(object):
    """Provide extension points for trainer.

    User can customize the function processing process and register it in the extension point.
    When running train process, the customized function will be executed.
    """

    __triggers__ = {}

    @classmethod
    def register(cls, trigger_names):
        """Register trigger class into __triggers__.

        Multi-trigger can be registered with one name, activate them all by default.
        :param trigger_names: str or list: trigger name list
        """

        def wrapper(t_cls):
            """Return a wrapper for the decorate.

            :param t_cls: class need to register
            :return: wrapper func
            """
            if isinstance(trigger_names, list):
                trigger_list = trigger_names
            else:
                trigger_list = trigger_names
            for name in trigger_list:
                if name not in cls.__triggers__:
                    cls.__triggers__[name] = [t_cls]
                else:
                    cls.__triggers__[name].append(t_cls)

        return wrapper

    @classmethod
    def activate(cls, name):
        """Activate triggers into registry by trigger name.

        :param name: trigger name
        :return: wrapper
        """

        def decorate(fn):
            """Return a decorate has args.

            :param fn: function need to register
            :return: decorate func
            """

            @wraps(fn)
            def wrapper(*args, **kwargs):
                """Return a wrapper for the decorate.

                :param args: dynamic args
                :param kwargs: dynamic kwargs
                :return: wrapper func
                """
                trigger_cls_list = cls.__triggers__.get(name)
                if not trigger_cls_list:
                    return fn(*args, **kwargs)
                # intersection triggers with triggers in config.yml
                if hasattr(cls, 'cfg') and cls.cfg.triggers is not None:
                    activate_triggers = list(set(trigger_cls_list).intersection(cls.cfg.triggers))
                else:
                    activate_triggers = trigger_cls_list
                for trigger_cls in activate_triggers:
                    trigger_cls().before(*args, **kwargs)
                result = fn(*args, **kwargs)
                for trigger_cls in activate_triggers:
                    trigger_cls().after(*args, **kwargs)
                return result

            return wrapper

        return decorate

    def before(self, *args, **kwargs):
        """Executor before func executed. This is abstract function need to implement.

        :param args: dynamic args
        :param kwargs: dynamic kwargs
        """
        pass

    def after(self, *args, **kwargs):
        """Executor after func executed. This is abstract function need to implement.

        :param args: dynamic args
        :param kwargs: dynamic kwargs
        """
        pass
