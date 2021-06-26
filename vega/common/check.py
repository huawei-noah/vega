# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
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
