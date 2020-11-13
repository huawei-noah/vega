# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The MasterFactory method.

Create Master or LocalMaster.
"""
from zeus.common.general import General
from .local_master import LocalMaster
from .master import Master


def create_master(**kwargs):
    """Return a LocalMaster instance when run on local, else return a master instance."""
    if General._parallel:
        return Master(**kwargs)
    else:
        return LocalMaster(**kwargs)
