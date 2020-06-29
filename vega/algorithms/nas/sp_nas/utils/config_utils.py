# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Functions for saving config. by Huang Guowei."""

import json


class _encoder(json.JSONEncoder):
    def encode(self, obj):
        def _tuples(item):
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': item}
            if isinstance(item, list):
                return [_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: _tuples(value) for key, value in item.items()}
            else:
                return item

        return super(_encoder, self).encode(_tuples(obj))


def _decoder(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj


def dict_to_json(config_dict, result_file):
    """Save config dict to json.

    :param config_dict: config dictionary
    :type config_dict: dict
    :param result_file: save json file
    :type result_file: str
    """
    config_str = _encoder().encode(config_dict)
    with open(result_file, 'w') as f:
        f.write(config_str)
    return True


def json_to_dict(json_file):
    """Load dictionary from json file.

    :param json_file: json file
    :type str: string
    :return: dictionary
    :rtype: dict
    """
    with open(json_file, "r") as f:
        data = json.load(f, object_hook=_decoder)
    return data
