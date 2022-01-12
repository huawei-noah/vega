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

"""The MasterFactory method.

Create Master or LocalMaster.
"""

import time
import logging
import traceback
from vega.common.general import General


logger = logging.getLogger(__name__)


__all__ = ["create_master", "shutdown_cluster"]
__master_instance__ = None


def create_master(**kwargs):
    """Return a LocalMaster instance when run on local, else return a master instance."""
    if General._parallel:
        global __master_instance__
        if __master_instance__:
            __master_instance__.restart(**kwargs)
            return __master_instance__
        else:
            from .master import Master
            __master_instance__ = Master(**kwargs)
            return __master_instance__
    else:
        from .local_master import LocalMaster
        return LocalMaster(**kwargs)


def shutdown_cluster():
    """Shutdown cluster."""
    global __master_instance__
    if not __master_instance__:
        time.sleep(2)
        return

    try:
        logger.info("Try to shutdown cluster.")
        __master_instance__.shutdown()
        time.sleep(12)
        logger.info("Cluster is shut down.")
    except Exception as e:
        logger.error(f"Pipeline's cluster shutdown error, message: {e}")
        logger.debug(traceback.format_exc())
