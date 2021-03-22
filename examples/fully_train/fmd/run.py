# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run example."""

from vega.tools.run_pipeline import run_pipeline


def _load_special_lib(config_file):
    import fmd


if __name__ == '__main__':
    run_pipeline(load_special_lib_func=_load_special_lib)
