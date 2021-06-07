# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run ModularNAS routines as main node."""
from modnas.utils.wrapper import run


def exec_main():
    """Run ModularNAS as main node."""
    override = [{
        'defaults': {
            'estim.*.main': True
        }
    }]
    return run(parse=True, override=override)


if __name__ == '__main__':
    exec_main()
