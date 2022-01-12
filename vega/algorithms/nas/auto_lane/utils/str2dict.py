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

"""Utilities for the conversion between str and dict."""
import ast


def str_warp(str_):
    """Change input string.

    :param str_: input string
    :type str_: str

    :return: output string
    :rtype: str
    """
    if isinstance(str_, str) and not (str_.startswith('\'') and str_.endswith('\'')):
        return "'{}'".format(str_)
    else:
        return str_


def str2dict(str_):
    """Change str to dict.

    :param str_: input str
    :type str_: str

    :return: output dict
    :rtype: dict
    """
    if not isinstance(str_, str):
        raise TypeError('"str_" must be a string, not {}'.format(type(str_)))
    # to keep the keys order
    str_.replace('dict(', 'OrderedDict(')
    return ast.literal_eval(str_)


def dict2str(dict_, tab=0, format_first_line=False, in_one_line=False):
    """Change dict to str.

    :param dict_: dict
    :type dict_: dict

    :param tab: add n tabs before each line
    :type tab: int

    :param format_first_line:  whether add tab before the first line
    :type format_first_line: bool

    :param in_one_line: whether convert dict to str in one line
    :type in_one_line: bool

    :return: result str
    :rtype: str
    """
    if not isinstance(dict_, dict):
        print(dict_)
        raise TypeError('"dict_" must be either a dict (or an OrderedDict), not {}'.format(type(dict_)))
    attr = ''
    space = ' ' * 4
    if len(dict_.keys()) <= 2 and len(str(list(dict_.values()))) <= 50:
        in_one_line = True
    if in_one_line:
        tab = 0
        format_first_line = False
    if format_first_line:
        tab += 1
    separator = ' ' if in_one_line else '\n'
    for key, value in dict_.items():
        if not isinstance(value, dict):
            if isinstance(value, str):
                value = str_warp(value)
            attr += "{}{}={},{}".format(space * tab, key, value, separator)
        else:
            attr += "{}{}={},{}".format(space * tab, key,
                                        dict2str(value, tab=tab + 1, in_one_line=in_one_line), separator)
    if format_first_line:
        tab -= 1
    return "{}dict({}{})".format(format_first_line * tab * space, (not in_one_line) * '\n',
                                 attr[:len(attr) - len(separator) - 1])
