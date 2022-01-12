# -*- coding: utf-8 -*-

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
