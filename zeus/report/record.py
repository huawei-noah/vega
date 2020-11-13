# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Report."""
import ast


class ReportRecord(object):
    """Record Class to record all data in one search loop."""

    def __init__(self, step_name=None, worker_id=None, **kwargs):
        self._step_name = step_name
        self._worker_id = worker_id
        self._desc = None
        self._performance = None
        self._checkpoint_path = None
        self._model_path = None
        self._weights_file = None
        self._info = None
        self._epoch = 0
        self._objectives = {}
        self._objective_keys = None
        self._rewards = None
        self._runtime = {}
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __hash__(self):
        """Override hash code."""
        return hash(self.uid)

    def __eq__(self, other):
        """Override eq func, step name and worker id is same."""
        return self.uid == other.uid

    def __repr__(self):
        """Override repr, output all record attrs."""
        return str(
            {'step_name': self._step_name, 'worker_id': self._worker_id, 'desc': self._desc, 'epoch': self._epoch,
             'performance': self._performance, 'checkpoint_path': self._checkpoint_path,
             'model_path': self._model_path, 'weights_file': self._weights_file, 'info': self._info,
             'objectives': self._objectives, '_objective_keys': self._objective_keys, 'rewards': self.rewards,
             'runtime': self._runtime})

    def __gt__(self, other):
        """Override gt for sorted according to performance attr."""
        return self.rewards > other.rewards

    @property
    def uid(self):
        """Uid for record. ReadOnly."""
        return '{}_{}_{}'.format(self.step_name, self.worker_id, self.epoch)

    @property
    def epoch(self):
        """Get epoch."""
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        """Set epoch."""
        self._epoch = value

    @property
    def step_name(self):
        """Get Step name."""
        return self._step_name

    @step_name.setter
    def step_name(self, value):
        """Set Step name."""
        self._step_name = value

    @property
    def worker_id(self):
        """Get worker id."""
        return self._worker_id

    @worker_id.setter
    def worker_id(self, value):
        """Set worker id."""
        self._worker_id = value

    @property
    def desc(self):
        """Get desc."""
        return self._desc

    @desc.setter
    def desc(self, value):
        """Set desc and parse value into dict."""
        if isinstance(value, str):
            value = ast.literal_eval(value)
        self._desc = value

    @property
    def performance(self):
        """Get performance."""
        return self._performance

    @performance.setter
    def performance(self, value):
        """Set performance and parse value into dict."""
        if isinstance(value, str):
            value = ast.literal_eval(value)
        self._performance = value

    @property
    def checkpoint_path(self):
        """Get checkpoint_path."""
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, value):
        """Set checkpoint_path and parse value into dict."""
        self._checkpoint_path = value

    @property
    def model_path(self):
        """Get model_path."""
        return self._model_path

    @model_path.setter
    def model_path(self, value):
        """Set model_path and parse value into dict."""
        self._model_path = value

    @property
    def weights_file(self):
        """Get weights file."""
        return self._weights_file

    @weights_file.setter
    def weights_file(self, value):
        """Set weights_file and parse value int dict."""
        self._weights_file = value

    @property
    def info(self):
        """Get rung id."""
        return self._info

    @info.setter
    def info(self, value):
        """Set rung id."""
        self._info = value

    @property
    def objectives(self):
        """Get objectives."""
        return self._objectives

    @objectives.setter
    def objectives(self, value):
        """Set objectives."""
        self._objectives = value

    @property
    def objective_keys(self):
        """Get objective_keys."""
        return self._objective_keys

    @objective_keys.setter
    def objective_keys(self, value):
        """Set objective_keys."""
        self._objective_keys = value if isinstance(value, list) else [value]

    @property
    def rewards(self):
        """Get reward_performance(ReadOnly)."""
        if not self.performance:
            return None
        if isinstance(self.performance, list):
            return self.performance
        if not self.objective_keys:
            self._objective_keys = list(self.performance.keys())
        res = []
        for obj in self.objective_keys:
            if isinstance(obj, int):
                obj = list(self.performance.keys())[obj]
            value = self.performance.get(obj)
            if value is None:
                raise ValueError("objective_keys in search_algorithm should be the same in trainer.metrics.")
            if self.objectives.get(obj) == 'MIN':
                value = -value
            res.append(value)
        return res[0] if len(res) == 1 else res

    @rewards.setter
    def rewards(self, value):
        """Get rewards, ReadOnly property."""
        self._rewards = value

    @property
    def runtime(self):
        """Get runtime."""
        return self._runtime

    @runtime.setter
    def runtime(self, value):
        """Set runtime."""
        self._runtime = value

    @classmethod
    def from_dict(cls, src_dic):
        """Create report class from dict."""
        src_cls = cls()
        if src_dic:
            for key, value in src_dic.items():
                setattr(src_cls, key, value)
        return src_cls

    def load_dict(self, src_dic):
        """Load values from dict."""
        if src_dic:
            for key, value in src_dic.items():
                setattr(self, key, value)
        return self

    def from_sample(self, sample, desc=None):
        """Load values from sample."""
        if isinstance(sample, tuple):
            sample = dict(worker_id=sample[0], desc=sample[1])
        self.load_dict(sample)
        if desc:
            self.desc = desc
        return self

    def serialize(self):
        """Serialize record class into a dict."""
        return ast.literal_eval(self.__repr__())
