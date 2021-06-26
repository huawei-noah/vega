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
        logger.error("Pipeline's cluster shutdown error: {}".format(str(e)))
        logger.error(traceback.format_exc())
