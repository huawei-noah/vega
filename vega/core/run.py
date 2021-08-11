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
import json
import vega
from vega.common.utils import init_log, lazy
from vega.common import Config, UserConfig
from vega.common.task_ops import TaskOps
from .pipeline.pipeline import Pipeline
from vega import set_backend
from vega.common.general import General
from vega.core.pipeline.conf import PipelineConfig


logger = logging.getLogger(__name__)


def run(cfg_path):
    """Run vega automl.

    :param cfg_path: config path.
    """
    if sys.version_info < (3, 6):
        sys.exit('Sorry, Python < 3.6 is not supported.')
    _init_env(cfg_path)
    _adjust_config()
    _run_pipeline()


def _init_env(cfg_path):
    """Init config and evn parameters.

    :param cfg_path: config file path
    """
    logging.getLogger().setLevel(logging.DEBUG)
    UserConfig().load(cfg_path)
    # load general
    General.from_dict(UserConfig().data.get("general"), skip_check=False)
    init_log(level=General.logger.level,
             log_file="pipeline.log",
             log_path=TaskOps().local_log_path)
    General.env = env_args()
    if not General.env:
        General.env = init_cluster_args()
    setattr(PipelineConfig, "steps", UserConfig().data.pipeline)
    set_backend(General.backend, General.device_category)


def _run_pipeline():
    """Run pipeline."""
    logging.info("-" * 48)
    logging.info("  task id: {}".format(General.task.task_id))
    logging.info("-" * 48)
    logger.info("configure: %s", json.dumps(UserConfig().data, indent=4))
    logging.info("-" * 48)
    Pipeline().run()


def _adjust_config():
    vega.quota().adjuest_pipeline_by_runtime(UserConfig().data)


@lazy
def env_args(args=None):
    """Call lazy function return args.

    :param args: args need to save and parse
    :return: args
    """
    return args


def init_cluster_args():
    """Initialize local_cluster."""
    if not General.cluster.master_ip:
        master_ip = '127.0.0.1'
        General.cluster.master_ip = master_ip
        env = Config({
            "init_method": "tcp://{}:{}".format(master_ip, General.cluster.listen_port),
            "world_size": 1,
            "rank": 0
        })
    else:
        world_size = len(General.cluster.slaves) + 1 if General.cluster.slaves else 1
        env = Config({
            "init_method": "tcp://{}:{}".format(
                General.cluster.master_ip, General.cluster.listen_port),
            "world_size": world_size,
            "rank": 0,
            "slaves": General.cluster.slaves,
        })
    General.env = env
    return env
