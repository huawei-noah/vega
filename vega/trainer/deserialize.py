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

"""Deserialize worker."""

import os
from copy import deepcopy
from vega.common import FileOps


def _get_worker_config(worker):
    """Save worker config."""
    from vega.common.class_factory import ClassFactory
    from vega.common.general import General
    from vega.datasets.conf.dataset import DatasetConfig
    from vega.networks.model_config import ModelConfig
    from vega.evaluator.conf import EvaluatorConfig
    from vega.core.pipeline.conf import PipeStepConfig

    worker_config = {
        "class_factory": deepcopy(ClassFactory.__registry__),
        "general": General().to_dict(),
        "dataset": DatasetConfig().to_dict(),
        "model": ModelConfig().to_dict(),
        "trainer": worker.config.to_dict(),
        "evaluator": EvaluatorConfig().to_dict(),
        "pipe_step": PipeStepConfig().to_dict()
    }
    return worker_config


def pickle_worker(workers, id):
    """Pickle worker to file."""
    for index, worker in enumerate(workers):
        worker_config = _get_worker_config(worker)
        config_file = os.path.join(
            worker.get_local_worker_path(),
            f".{str(id)}.{str(index)}.config.pkl")
        FileOps.dump_pickle(worker_config, config_file)
        # pickle worker
        worker_file = os.path.join(
            worker.get_local_worker_path(),
            f".{str(id)}.{str(index)}.worker.pkl")
        FileOps.dump_pickle(worker, worker_file)


def load_config(config_file):
    """Load config from file."""
    # load General config (includes security setting)
    from vega.common.general import General
    General.security = False
    config = FileOps.load_pickle(config_file)
    General.from_dict(config["general"])

    # if security mode, reload config
    if General.security:
        config = FileOps.load_pickle(config_file)
    from vega.common.class_factory import ClassFactory
    from vega.common.general import General
    from vega.datasets.conf.dataset import DatasetConfig
    from vega.networks.model_config import ModelConfig
    from vega.trainer.conf import TrainerConfig
    from vega.evaluator.conf import EvaluatorConfig
    from vega.core.pipeline.conf import PipeStepConfig

    ClassFactory.__registry__ = config["class_factory"]
    DatasetConfig.from_dict(config["dataset"])
    ModelConfig.from_dict(config["model"])
    TrainerConfig.from_dict(config["trainer"])
    EvaluatorConfig.from_dict(config["evaluator"])
    PipeStepConfig.from_dict(config["pipe_step"])


def load_worker(worker_file):
    """Load worker from file."""
    worker = FileOps.load_pickle(worker_file)
    return worker
