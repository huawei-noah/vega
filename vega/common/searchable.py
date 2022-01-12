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
from vega.common.utils import singleton


@singleton
class SearchableRegister(object):
    """Searchable Register class."""

    __types__ = []
    __searchable_classes__ = {}
    __hooks__ = []

    def init(self):
        """Init items."""
        self.__types__ = []
        self.__searchable_classes__ = {}
        self.__hooks__ = []
        return self

    def has_searchable(self):
        """Check searchable is not None."""
        return self.__searchable_classes__

    def search_space(self):
        """Get all search space."""
        res = []
        for v in self.__searchable_classes__.values():
            search_space = v.search_space
            if isinstance(search_space, list):
                res.extend(search_space)
            else:
                res.append(v.search_space())
        return res

    def add_space(self, name, module):
        """Add search space."""
        for searchable in self.__types__:
            entity = searchable(name)
            if not entity.search_on(module):
                continue
            self.__searchable_classes__[name] = entity

    def register(self, searchable):
        """Register search space."""
        self.__types__.append(searchable)

    def update(self, desc):
        """Update."""
        res = {}
        for k, v in self.__searchable_classes__.items():
            sub_desc = desc.get(k)
            if not sub_desc:
                continue
            v.update(sub_desc)
            res[k] = sub_desc
        return res

    def active_searchable(self, name, module):
        """Active searchable function."""
        searchable = self.__searchable_classes__.get(name)
        if not searchable:
            return None
        return searchable(module)

    def active_search_event(self, model):
        """Active searchable event."""
        if not hasattr(model, "named_modules"):
            return model
        for name, m in model.named_modules():
            for hook in self.__hooks__:
                hook(model, name, self.active_searchable(name, m))
        return model

    def add_search_event(self, fn):
        """Add event into searchable class."""
        self.__hooks__.append(fn)


class Searchable(object):
    """Searchable base class."""

    _key = None
    _type = None
    _range = None

    def __init__(self, key, type=None, range=None):
        self.key = key or self._key
        self.type = type or self._type
        self.range = range or self._range
        self.desc = None

    def search_space(self):
        """Get search space."""
        return dict(key=self.key, type=self.type, range=self.range)

    def update(self, desc):
        """Update desc."""
        self.desc = desc

    def search_on(self, module):
        """Call search on function."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Call searchable."""
        raise NotImplementedError


def space(key=None, type=None, range=None):
    """Set class to singleton class."""

    def decorator(cls):
        """Get decorator."""
        cls._type = type
        cls._range = range
        cls._key = key
        SearchableRegister().register(cls)
        return cls

    return decorator


def change_space(hps):
    """Change space by hps."""
    all_searchable_cls = SearchableRegister().__types__
    for cls in all_searchable_cls:
        if cls._key == hps.get('key'):
            cls._type = hps.get('type')
            cls._range = hps.get('range')
