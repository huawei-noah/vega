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
from .common.utils import init_log, lazy
from .common import Config, UserConfig
from .pipeline.pipeline import Pipeline
from .common.consts import ClusterMode
from .backend_register import set_backend
from .common.general import General
from vega.core.common.loader import load_conf_from_desc
from vega.core.pipeline.conf import PipelineConfig


logger = logging.getLogger(__name__)


def run(cfg_path):
    """Run vega automl.

    :param cfg_path: config path.
    """
    if sys.version_info < (3, 6):
        sys.exit('Sorry, Python < 3.6 is not supported.')
    _init_env(cfg_path)
    _run_pipeline()


def _init_env(cfg_path):
    """Init config and evn parameters.

    :param cfg_path: config file path
    """
    logging.getLogger().setLevel(logging.DEBUG)
    UserConfig().load(cfg_path)
    # load general
    if "general" in UserConfig().data:
        load_conf_from_desc(General, UserConfig().data.get("general"))
    init_log(General.logger.level)
    cluster_args = env_args()
    if cluster_args is None:
        if General.cluster.master_ip is None:
            General.cluster_mode = ClusterMode.Single
            cluster_args = init_local_cluster_args(General.cluster.master_ip,
                                                   General.cluster.listen_port)
        else:
            General.cluster_mode = ClusterMode.LocalCluster
            cluster_args = init_local_cluster_args(General.cluster.master_ip,
                                                   General.cluster.listen_port,
                                                   General.cluster.slaves)
    setattr(PipelineConfig, "steps", UserConfig().data.pipeline)
    General.env = cluster_args
    set_backend(General.backend, General.device_category)


def _run_pipeline():
    """Run pipeline."""
    logging.info("-" * 48)
    logging.info("  task id: {}".format(General.task.task_id))
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
            General.cluster.master_ip = master_ip
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
