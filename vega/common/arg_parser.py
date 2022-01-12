# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Arg parser."""

import argparse


__all__ = ["argment_parser", "str2bool"]


def str2bool(value):
    """Convert string to boolean."""
    value = str(value)
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def argment_parser(desc=None):
    """Parser argment."""
    return argparse.ArgumentParser(description=desc, formatter_class=CustomHelpFormat)


class CustomHelpFormat(argparse.HelpFormatter):

    def _get_help_string(self, action):
        help = action.help
        if isinstance(action, argparse._HelpAction):
            return help

        items = []
        if action.choices:
            items.append(f"choices: {'|'.join(action.choices)}")

        if "%(default)" not in action.help and action.default is not None and action.default not in [True, False]:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    items.append("default: %(default)s")

        if items:
            help += " (" + ", ".join(items) + ")"
        return help

    def _format_action_invocation(self, action):
        if action.option_strings:
            return ", ".join(action.option_strings)
        else:
            return action.dest
