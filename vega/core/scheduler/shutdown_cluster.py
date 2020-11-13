# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The shutdown_cluster method.

The shutdown_cluster method to shutdown cluster imminently.
"""
import logging
import traceback
from zeus.common.general import General


def shutdown_cluster():
    """Shutdown all distributed cluster."""
    # detect master is running
    if not General._parallel:
        return
    try:
        logging.info("Try to shutdown cluster.")
        from zeus.trainer.utils import load_master_ip
        from distributed import Client
        ip, port = load_master_ip()
        if ip is None or port is None:
            logging.info("Stand-alone mode, no need to shut down the cluster.")
            return
        shutdown_client = Client("{}:{}".format(ip, port))
        logging.info("Cluster will be shut down.")
        shutdown_client.shutdown()
        shutdown_client.close()
        del shutdown_client
        logging.info("Cluster is shut down.")
    except Exception as e:
        logging.error("Pipeline's cluster shutdown error: {}".format(str(e)))
        logging.error(traceback.format_exc())
