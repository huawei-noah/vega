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

"""TaskOps class.

tasks/<task_id>/
    +-- output
    |       +-- <config file>                           # backuped config file
    |       +-- <step name>
    |               +-- output.csv                      # step output data
    |               +-- report.csv                      # history file
    |               +-- desc_<worker id>.json           # model desc or hp desc
    |               +-- performance_<worker id>.pth     # performance file
    |               +-- model_<worker id>.pth           # pytorch model weight file
    |               +-- model_<worker id>               # tensorflow model
    +-- workers
    |       +-- <step name>
    |       |       +-- <worker id>
    |       |               +-- desc_<worker id>.json   # model desc or hp desc
    |       |               +-- model.ckpt              # checkpoint file
    |       |               +-- model_<worker id>[.pth] # best model
    |       |               +-- performance_<worker id>.json   # model performance
    |       +-- <step name>
    +-- temp
    +-- visual
    +-- logs
            +-- pipeline.log                            # main log
            +-- worker_<worker id>.log                  # worker log
            +-- evaluator_<worker id>.log               # evaluator log

properties:
    task_id

    output_subpath
    log_subpath
    result_subpath
    best_model_subpath

    local_base_path
    local_output_path
    local_log_path
    local_visual_path

    backup_base_path
    need_backup

    model_zoo_path
    temp_path

methods:
    get_worker_subpath(step_name=None, worker_id=None)
    get_local_worker_path(step_name=None, worker_id=None)

"""
import os
import logging
from .file_ops import FileOps
from .general import TaskConfig, General

logger = logging.getLogger(__name__)


class TaskOps(object):
    """This is a class containing the method of path."""

    task_cfg = TaskConfig()

    def __init__(self, task_id=None, step_name=None, worker_id=None):
        """Init TaskOps."""
        self.__task_id__ = self.task_cfg.task_id
        self._step_name = General.step_name
        if task_id:
            self.__task_id__ = task_id
        if step_name:
            self._step_name = step_name
        if worker_id:
            self.worker_id = worker_id

    @property
    def task_id(self):
        """Property: task_id."""
        return self.__task_id__

    @property
    def output_subpath(self):
        """Property: output sub-path."""
        return self.task_cfg.output_subpath

    @property
    def log_subpath(self):
        """Property: log sub-path."""
        return self.task_cfg.log_subpath

    @property
    def result_subpath(self):
        """Property: resutl sub-path."""
        return self.task_cfg.result_subpath

    @property
    def best_model_subpath(self):
        """Property: best model sub-path."""
        return self.task_cfg.best_model_subpath

    def get_worker_subpath(self, step_name=None, worker_id=None):
        """Get worker path relative to the local path.

        :param step_name: step name in the pipeline.
        :type step_name: str.
        :param worker_id: the worker's worker id.
        :type worker_id: str.
        :return: worker's path relative to the local path.
        :rtype: str.

        """
        if step_name is None and worker_id is None:
            step_name = self._step_name
            worker_id = self.worker_id
        _path = self.task_cfg.worker_subpath
        return _path.replace("[step_name]", str(step_name)).replace("[worker_id]", str(worker_id))

    @property
    def local_base_path(self):
        """Property: get the local base path.

        :return: local base path.
        :rtype: str.

        """
        _path = self.task_cfg.local_base_path
        if self.task_id:
            _path = FileOps.join_path(_path, self.task_id)
        FileOps.make_dir(_path)
        return os.path.abspath(_path)

    @property
    def local_output_path(self):
        """Property: get the local output path.

        :return: local output path.
        :rtype: str.

        """
        _base = self.local_base_path
        _path = FileOps.join_path(_base, self.output_subpath)
        FileOps.make_dir(_path)
        return _path

    @property
    def local_log_path(self):
        """Property: get the local log path.

        :return: local log path.
        :rtype: str.

        """
        _base = self.local_base_path
        _path = FileOps.join_path(_base, self.log_subpath)
        FileOps.make_dir(_path)
        return _path

    @property
    def local_visual_path(self):
        """Property: get the local path used to store training visual data.

        :return: local training visual data path.
        :rtype: str.

        """
        _base = self.local_base_path
        _path = FileOps.join_path(_base, "visual")
        FileOps.make_dir(_path)
        return _path

    @property
    def backup_base_path(self):
        """Property: get the remote s3 base path.

        :return: remote s3 base path.
        :rtype: str.

        """
        _path = self.task_cfg.backup_base_path
        if _path is None:
            return None
        if self.task_id:
            _path = FileOps.join_path(_path, self.task_id)
        return _path

    @property
    def need_backup(self):
        """Property: if need backup.

        :return: if need backup.
        :rtype: bool.

        """
        return self.task_cfg.backup_base_path is not None and self.task_cfg.backup_base_path != ""

    def get_local_worker_path(self, step_name=None, worker_id=None):
        """Get the local worker path.

        :param step_name: step name in the pipeline.
        :type step_name: str.
        :param worker_id: the worker's worker id.
        :type worker_id: str.
        :return: worker's local path.
        :rtype: str.

        """
        if step_name is None:
            step_name = self.step_name
        if worker_id is None:
            worker_id = self.worker_id
        _base = self.local_base_path
        _path = FileOps.join_path(_base, self.get_worker_subpath(str(step_name), str(worker_id)))
        FileOps.make_dir(_path)
        return _path

    @property
    def use_dloop(self):
        """Whether to use dloop in pipeline.

        :return: use dloop or not
        :rtype: bool
        """
        return self.task_cfg.use_dloop

    @property
    def temp_path(self):
        """Return model zoo path."""
        _path = FileOps.join_path(self.local_base_path, "temp")
        FileOps.make_dir(_path)
        return _path

    @property
    def step_name(self):
        """Return general step nmae."""
        return self._step_name

    @step_name.setter
    def step_name(self, value):
        """Setter: set step_name with value.

        :param value: step name
        :type value: str
        """
        self._step_name = value

    @property
    def step_path(self):
        """Return step path."""
        _path = FileOps.join_path(self.local_output_path, self.step_name)
        FileOps.make_dir(_path)
        return _path

    def backup(self):
        """Backup current folder."""
        if self.need_backup:
            FileOps.copy_folder(self.local_output_path,
                                FileOps.join_path(self.backup_base_path, self.output_subpath))

    def get_save_path(self, file_path=None):
        """Replace file path accorrding to names."""
        if 'local_base_path' in file_path:
            file_path = file_path.replace('{' + 'local_base_path' + '}', self.local_base_path)
        return file_path
