# -*- coding: utf-8 -*-

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

"""Base Trainer."""

import logging
import subprocess
import traceback
import os
import glob
from vega.common import Config
from vega.common.general import General
from vega.common.wrappers import train_process_wrapper
from vega.trainer.distributed_worker import DistributedWorker
from vega.trainer.conf import TrainerConfig
from vega.common.class_factory import ClassFactory, ClassType
from vega.trainer.utils import WorkerTypes
from vega.common import FileOps

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

    @train_process_wrapper
    def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        try:
            self._dump_trial_config()
            self._run_script()
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(f"Failed to run script, message: {e}.")

    def _run_script(self):
        """Run script."""
        env = os.environ.copy()
        script = os.path.abspath(self.config.script)
        cmd = [General.python_command, script]
        if hasattr(self.config, "params") and self.config.params is not None:
            params = [f"--{k}={v}" for k, v in self.config.params.items()]
            cmd += params
        try:
            proc = subprocess.Popen(cmd, env=env, cwd=self.get_local_worker_path())
            logger.info(f"start process, pid: {proc.pid}")
            proc.wait(timeout=General.worker.timeout)
        except Exception as e:
            logger.warn(f"Timeout worker has been killed, message: {e}.")
            logger.debug(traceback.format_exc())

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
        FileOps.dump_pickle(data, _file)

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
