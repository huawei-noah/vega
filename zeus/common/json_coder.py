# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Utils tools."""

import json
from datetime import datetime
import logging
import numpy as np
from .consts import Status, DatatimeFormatString


logger = logging.getLogger(__name__)


class JsonEncoder(json.JSONEncoder):
    """Json encoder, encoder some special object."""

    def default(self, obj):
        """Override default function."""
        if isinstance(obj, datetime):
            return obj.strftime(DatatimeFormatString)
        elif isinstance(obj, Status):
            return obj.value
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            return json.JSONEncoder.default(self, obj)
