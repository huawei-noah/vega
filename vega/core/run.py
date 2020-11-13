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
import yaml
import os
import logging
import json
from zeus.common.utils import init_log, lazy
from zeus.common import Config, UserConfig
from zeus.common.task_ops import TaskOps, FileOps
from .pipeline.pipeline import Pipeline
from .backend_register import set_backend
from zeus.common.general import General
from vega.core.pipeline.conf import PipelineConfig
from vega.core.quota import QuotaStrategy


logger = logging.getLogger(__name__)


def run(cfg_path):
    """Run vega automl.

    :param cfg_path: config path.
    """
    if sys.version_info < (3, 6):
        sys.exit('Sorry, Python < 3.6 is not supported.')
    _init_env(cfg_path)
    _backup_cfg(cfg_path)
    _adjust_config()
    _run_pipeline()


def _init_env(cfg_path):
    """Init config and evn parameters.

    :param cfg_path: config file path
    """
    logging.getLogger().setLevel(logging.DEBUG)
    UserConfig().load(cfg_path)
    # load general
    General.from_json(UserConfig().data.get("general"), skip_check=False)
    init_log(level=General.logger.level,
             log_path=TaskOps().local_log_path)
    cluster_args = env_args()
    if not cluster_args:
        cluster_args = init_local_cluster_args()
    setattr(PipelineConfig, "steps", UserConfig().data.pipeline)
    General.env = cluster_args
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
    if General.quota.runtime is None:
        return
    adjust_strategy = QuotaStrategy()
    config_new = adjust_strategy.adjust_pipeline_config(UserConfig().data)
    UserConfig().data = config_new


@lazy
def env_args(args=None):
    """Call lazy function return args.

    :param args: args need to save and parse
    :return: args
    """
    return args


def init_local_cluster_args():
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
        world_size = len(General.cluster.slaves) if General.cluster.slaves else 1
        env = Config({
            "init_method": "tcp://{}:{}".format(
                General.cluster.master_ip, General.cluster.listen_port),
            "world_size": world_size,
            "rank": 0,
            "slaves": General.cluster.slaves,
        })
    return env


def _backup_cfg(cfg_path):
    """Backup yml file.

    :param cfg_path: path of yml file.
    """
    if isinstance(cfg_path, str):
        output_path = FileOps.join_path(TaskOps().local_output_path, os.path.basename(cfg_path))
        FileOps.copy_file(cfg_path, output_path)
    else:
        output_path = FileOps.join_path(TaskOps().local_output_path, 'config.yml')
        with open(output_path, 'w') as f:
            f.write(yaml.dump(cfg_path))
