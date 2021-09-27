# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Deserialize worker."""

import os
import pickle
from copy import deepcopy


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
        with open(config_file, "wb") as f:
            pickle.dump(worker_config, f)
        # pickle worker
        worker_file = os.path.join(
            worker.get_local_worker_path(),
            f".{str(id)}.{str(index)}.worker.pkl")
        with open(worker_file, "wb") as f:
            pickle.dump(worker, f)


def load_config(config_file):
    """Load config from file."""
    import pickle

    with open(config_file, 'rb') as f:
        config = pickle.load(f)

    from vega.common.class_factory import ClassFactory
    from vega.common.general import General
    from vega.datasets.conf.dataset import DatasetConfig
    from vega.networks.model_config import ModelConfig
    from vega.trainer.conf import TrainerConfig
    from vega.evaluator.conf import EvaluatorConfig
    from vega.core.pipeline.conf import PipeStepConfig

    ClassFactory.__registry__ = config["class_factory"]
    General.from_dict(config["general"])
    DatasetConfig.from_dict(config["dataset"])
    ModelConfig.from_dict(config["model"])
    TrainerConfig.from_dict(config["trainer"])
    EvaluatorConfig.from_dict(config["evaluator"])
    PipeStepConfig.from_dict(config["pipe_step"])


def load_worker(worker_file):
    """Load worker from file."""
    import pickle
    with open(worker_file, 'rb') as f:
        worker = pickle.load(f)
    return worker
