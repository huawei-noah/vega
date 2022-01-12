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

"""Utils for checking yml."""


class BaseChecking(object):
    """Check CheckBase for checking rules."""

    @classmethod
    def check_all(cls, attr_name, rules, checked_cls_name, config):
        """Check rules for attr."""
        for subclass in cls.__subclasses__():
            subclass.check(attr_name, rules, checked_cls_name, config)

    @classmethod
    def check(cls, attr_name, rules):
        """Check single rules for attr."""
        pass


class RequiredChecking(BaseChecking):
    """For checking Required."""

    @classmethod
    def check(cls, attr_name, rules, checked_cls_name, config):
        """Check required attr."""
        if "required" in rules:
            if attr_name not in config:
                raise Exception("{} attr is required in {}".format(
                    attr_name, checked_cls_name))
        else:
            if attr_name not in config:
                return False
        return True


class TypeCheck(BaseChecking):
    """for checking Type."""

    @classmethod
    def check(cls, attr_name, rules, checked_cls_name, config):
        """Check type attr."""
        if attr_name not in config:
            return
        if not isinstance(config[attr_name], rules["type"]):
            raise Exception("{} in {} attr must be type: {}".format(
                attr_name, checked_cls_name, rules["type"]))


class ScopeCheck(BaseChecking):
    """for checking Scope."""

    @classmethod
    def check(cls, attr_name, rules, checked_cls_name, config):
        """Check scope attr."""
        if attr_name not in config:
            return
        if "scope" in rules:
            if isinstance(rules["scope"], list):
                if config[attr_name] not in rules["scope"]:
                    raise Exception("The value of {} in {}attr must in: {}".format(
                        attr_name, checked_cls_name, rules["scope"]))


def valid_rule(cls, config, check_rules):
    """Check config."""
    if not check_rules:
        return
    cls_name = cls.__name__
    for attr_name, rules in check_rules.items():
        BaseChecking().check_all(attr_name, rules, cls_name, config)


def make_rules(adict, attr_name, if_required, types, scpoe=None):
    """Make new rule in dict for attr."""
    if attr_name not in adict:
        adict[attr_name] = {}
    adict[attr_name]['required'] = if_required
    adict[attr_name]['type'] = types
    if scpoe:
        adict[attr_name]['scope'] = scpoe
    return adict
