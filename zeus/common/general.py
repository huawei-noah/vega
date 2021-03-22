# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Default general."""
import os
from datetime import datetime
from .config_serializable import ConfigSerializable


class TaskConfig(ConfigSerializable):
    """Task Config."""

    task_id = datetime.now().strftime('%m%d.%H%M%S.%f')[:-3]
    local_base_path = os.path.abspath("./tasks")
    output_subpath = "output"
    best_model_subpath = "best_model"
    log_subpath = "logs"
    result_subpath = "result"
    worker_subpath = "workers/[step_name]/[worker_id]"
    backup_base_path = None
    use_dloop = False


class ClusterConfig(ConfigSerializable):
    """Cluster Config."""

    master_ip = None
    listen_port = 8000
    slaves = None


class Worker(ConfigSerializable):
    """Worker Config."""

    # distributed = False
    timeout = 1000
    eval_count = 10
    evaluate_timeout = 0.1


class Logger(ConfigSerializable):
    """Logger Config."""

    level = 'info'


class Target(ConfigSerializable):
    """Target Config."""

    type = None
    value = None


class Restrict(ConfigSerializable):
    """Restriction Config."""

    flops = None
    latency = None
    params = None
    model_valid = None
    duration = {}
    trials = {}


class Affinity(ConfigSerializable):
    """Affinity Config."""

    type = None
    affinity_file = None
    affinity_value = None


class Strategy(ConfigSerializable):
    """Strategy Config."""

    runtime = None
    only_search = False


class QuotaConfig(ConfigSerializable):
    """Quota Config."""

    strategy = Strategy
    target = Target
    restrict = Restrict
    filter_rules = "(flops_params)"
    affinity = Affinity


class General(ConfigSerializable):
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
    calc_params_each_epoch = False
    dft = False
    workers_num = 1
    quota = QuotaConfig
    data_format = None
    # parallel
    parallel_search = False
    parallel_fully_train = False
    _parallel = False
    _resume = False
    devices_per_trainer = 1
    clean_worker_dir = True
    requires = []
