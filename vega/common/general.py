# -*- coding:utf-8 -*-

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

"""Default general."""
import os
import sys
from datetime import datetime
from vega.common.utils import get_available_port
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
    listen_port = get_available_port(min_port=28000, max_port=28999)
    slaves = []
    standalone_boot = False
    num_workers = 0
    num_nodes = 1
    num_workers_per_node = 1
    horovod = False         # read-only
    hccl = False            # read-only
    hccl_port = get_available_port(min_port=29000, max_port=29999)
    hccl_server_ip = None   # read-only
    enable_broadcast_buffers = False
    show_all_ranks = False


class Worker(ConfigSerializable):
    """Worker Config."""

    timeout = 365 * 24 * 3600     # 365 days
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


class General(ConfigSerializable):
    """General Config."""

    task = TaskConfig
    step_name = "pipestep"
    logger = Logger
    backend = 'pytorch'
    device_category = 'GPU'
    TF_CPP_MIN_LOG_LEVEL = 2
    cluster = ClusterConfig
    worker = Worker
    env = None
    calc_params_each_epoch = False
    dft = False
    workers_num = 1
    quota = None
    data_format = "channels_first"
    # parallel
    parallel_search = False
    parallel_fully_train = False
    _parallel = False
    _resume = False
    devices_per_trainer = 1
    clean_worker_dir = True
    requires = []
    message_port = None
    python_command = sys.executable or "python3"
    device_evaluate_before_train = True
    ms_execute_mode = 0  # 0-GRAPH_MODE 1-PYNATIVE_MODE
    dataset_sink_mode = True
    security = False
    skip_trainer_error = True
