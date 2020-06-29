# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""vega run.py."""
import sys
import logging
import argparse
from .common.utils import init_log, lazy
from .common import Config, UserConfig, DefaultConfig
from .pipeline.pipeline import Pipeline
from .common.consts import ClusterMode
from .backend_register import set_backend

logger = logging.getLogger(__name__)


def run(cfg_path):
    """Run vega automl.

    :param cfg_path: config path.
    """
    if sys.version_info < (3, 6):
        sys.exit('Sorry, Python < 3.6 is not supported.')
    _init_env(cfg_path)
    _run_pipeline()


def set_zone(zone):
    """Set zone.

    :param zone: zone.
    """
    if zone == "huawei_yellow_zone":
        DefaultConfig().data = DefaultConfig().data.hw_y
    elif zone == "huawei_green_zone":
        DefaultConfig().data = DefaultConfig().data.hw_g
    else:
        logger.warn("Unavailable zoo, do nothing, zoo={}".format(zone))


def _init_env(cfg_path):
    """Init config and evn parameters.

    :param cfg_path: config file path
    """
    logging.getLogger().setLevel(logging.DEBUG)
    UserConfig().load(cfg_path)
    init_log(UserConfig().data.general.logger.level)
    cluster_args = env_args()
    UserConfig().data.env = cluster_args
    UserConfig().data.general.cluster_mode = ClusterMode.LocalCluster
    if cluster_args is None:
        if UserConfig().data.general.cluster.master_ip is None:
            UserConfig().data.general.cluster_mode = ClusterMode.Single
            env = init_local_cluster_args(UserConfig().data.general.cluster.master_ip,
                                          UserConfig().data.general.cluster.listen_port)
            UserConfig().data.env = env
        else:
            env = init_local_cluster_args(UserConfig().data.general.cluster.master_ip,
                                          UserConfig().data.general.cluster.listen_port,
                                          UserConfig().data.general.cluster.slaves)
            UserConfig().data.env = env
    set_backend(UserConfig().data.general.get('backend', 'pytorch'))


def _run_pipeline():
    """Run pipeline."""
    logging.info("-" * 48)
    logging.info("  task id: {}".format(UserConfig().data.general.task.task_id))
    logging.info("-" * 48)
    logger.info("configure: %s", str(UserConfig().data))
    logging.info("-" * 48)
    Pipeline().run()


@lazy
def env_args(args=None):
    """Call lazy function return args.

    :param args: args need to save and parse
    :return: args
    """
    return args


def init_local_cluster_args(master_ip, listen_port, slaves=None):
    """Initialize local_cluster."""
    try:
        if master_ip is None:
            master_ip = '127.0.0.1'
            env = Config({"init_method": "tcp://{}:{}".format(master_ip, listen_port),
                          "world_size": 1,
                          "rank": 0
                          })
        else:
            world_size = len(slaves) if slaves else 1
            env = Config({"init_method": "tcp://{}:{}".format(master_ip, listen_port),
                          "world_size": world_size,
                          "rank": 0,
                          "slaves": slaves,
                          })
        return env
    except Exception:
        raise ValueError("Init local cluster failed")
