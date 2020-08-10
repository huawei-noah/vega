# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Default general."""
from datetime import datetime
from .consts import ClusterMode


class TaskConfig(dict):
    """Task Config."""

    task_id = datetime.now().strftime('%m%d.%H%M%S.%f')[:-3]
    local_base_path = "./tasks"
    output_subpath = "output"
    best_model_subpath = "best_model"
    log_subpath = "logs"
    result_subpath = "result"
    worker_subpath = "workers/[step_name]/[worker_id]"
    backup_base_path = None
    use_dloop = False


class ClusterConfig(object):
    """Cluster Config."""

    master_ip = None
    listen_port = 8000
    slaves = None


class Worker(object):
    """Worker Config."""

    # distributed = False
    timeout = 1000
    devices_per_job = -1
    eval_count = 10
    evaluate_timeout = 0.1


class Logger(object):
    """Logger Config."""

    level = 'info'


class ModelZoo(object):
    """Model Zoo Config."""

    model_zoo_path = None


class General(object):
    """General Config."""

    task = TaskConfig
    step_name = None
    worker_id = None
    logger = Logger
    backend = 'pytorch'
    device_category = 'GPU'
    cluster = ClusterConfig
    worker = Worker
    env = None
    model_zoo = ModelZoo
    cluster_mode = ClusterMode.LocalCluster
    calc_params_each_epoch = False
