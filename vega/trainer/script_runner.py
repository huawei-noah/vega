# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Base Trainer."""

import logging
import subprocess
import traceback
import os
import pickle
import glob
from vega.common import init_log, Config
from vega.common.general import General
from vega.trainer.distributed_worker import DistributedWorker
from vega.trainer.conf import TrainerConfig
from vega.common.class_factory import ClassFactory, ClassType
from vega.trainer.utils import WorkerTypes

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.TRAINER)
class ScriptRunner(DistributedWorker):
    """Trainer base class."""

    def __init__(self, model=None, id=None, hps=None, model_desc=None, **kwargs):
        super().__init__()
        self.model_desc = model_desc
        self.worker_id = id
        self.config = TrainerConfig()
        self.hps = self._get_hps(hps)
        self.worker_type = WorkerTypes.TRAINER

    def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        init_log(level=General.logger.level,
                 log_file=f"{self.step_name}_worker_{self.worker_id}.log",
                 log_path=self.local_log_path)
        self._dump_trial_config()
        self._run_script()

    def _run_script(self):
        """Run script."""
        env = os.environ.copy()
        script = os.path.abspath(self.config.script)
        cmd = [General.python_command, script]
        if hasattr(self.config, "params") and self.config.params is not None:
            cmd = [General.python_command, self.config.params]
        try:
            proc = subprocess.Popen(cmd, env=env, cwd=self.get_local_worker_path())
            logger.info(f"start process, pid: {proc.pid}")
            proc.wait(timeout=General.worker.timeout)
        except Exception:
            logger.warn("Timeout worker has been killed.")
            logger.warn(traceback.print_exc())

    def _dump_trial_config(self):
        """Dump trial config."""
        data = {
            "general": General().to_dict(),
            "worker_id": self.worker_id,
            "model_desc": self.model_desc,
            "hps": self.hps,
            "epochs": self.config.epochs,
        }
        _file = os.path.join(self.get_local_worker_path(), ".trial")
        with open(_file, "wb") as f:
            pickle.dump(data, f)

    def _get_hps(self, hps):
        if hps is not None:
            pass
        elif self.config.hps_file is not None:
            hps_file = self.config.hps_file.replace("{local_base_path}", self.local_base_path)
            if os.path.isdir(hps_file):
                pattern = os.path.join(hps_file, "hps_*.json")
                hps_file = glob.glob(pattern)[0]
            hps = Config(hps_file)
            if "trainer" in hps:
                if "epochs" in hps["trainer"]:
                    hps["trainer"].pop("epochs")
                if "checkpoint_path" in hps["trainer"]:
                    hps["trainer"].pop("checkpoint_path")
        return hps
