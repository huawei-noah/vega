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

"""Management class registration and bind configuration properties, provides the type of class supported."""


class ClassFactory(object):
    """A Factory Class to manage all class need to register with config."""

    __registry__ = {}

    @classmethod
    def register(cls):
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
            t_cls_name = t_cls.__name__
            if t_cls_name not in cls.__registry__:
                cls.__registry__[t_cls_name] = t_cls
            else:
                raise ValueError("Cannot register duplicate class ({})".format(t_cls_name))

            return t_cls

        return wrapper

    @classmethod
    def get_cls(cls, cls_name=None):
        """Get class from the classfactory.

        :param cls_name: the name of the class
        :type cls_name: str
        """
        t_cls = cls.__registry__.get(cls_name)
        return t_cls
