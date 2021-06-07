# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Arg parser."""

import argparse


__all__ = ["argment_parser"]


def argment_parser(desc=None):
    """Parser argment."""
    return argparse.ArgumentParser(description=desc, formatter_class=CustomHelpFormat)


class CustomHelpFormat(argparse.HelpFormatter):

    def _get_help_string(self, action):
        help = action.help
        if isinstance(action, argparse._HelpAction):
            return help

        items = []

        if action.type and action.default not in [True, False]:
            items.append(f"type: {action.type.__name__}")

        if action.choices:
            items.append(f"choices: {'|'.join(action.choices)}")

        if '%(default)' not in action.help and action.default is not None and action.default not in [True, False]:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    items.append('default: %(default)s')

        if items:
            help += " (" + ", ".join(items) + ")"
        return help

    def _format_action_invocation(self, action):
        if action.option_strings:
            return ", ".join(action.option_strings)
        else:
            return action.dest
