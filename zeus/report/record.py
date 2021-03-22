# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Report."""

import json
from zeus.common.utils import remove_np_value


class ReportRecord(object):
    """Record Class to record all data in one search loop."""

    def __init__(self, step_name=None, worker_id=None, **kwargs):
        self._step_name = step_name
        self._worker_id = worker_id
        self._desc = None
        self._hps = None
        self._performance = None
        self._checkpoint_path = None
        self._model_path = None
        self._weights_file = None
        self._epoch = 0
        self._objectives = {}
        self._objective_keys = None
        self._rewards = None
        self._runtime = {}
        self._original_rewards = None
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __hash__(self):
        """Override hash code."""
        return hash(self.uid)

    @property
    def code(self):
        """Hash code of record."""
        return hash(self.__repr__())

    def __eq__(self, other):
        """Override eq func, step name and worker id is same."""
        return self.uid == other.uid

    def __repr__(self):
        """Override repr, output all record attrs."""
        return json.dumps(self.to_dict())

    def to_dict(self):
        """Convert to dictionary."""
        all_attr = {}
        for _name in dir(self):
            if _name.startswith("_"):
                continue
            if _name in self.__dict__:
                all_attr[_name] = self.__dict__[_name]
            elif "_" + _name in self.__dict__:
                all_attr[_name] = self.__dict__["_" + _name]
        all_attr["original_rewards"] = self._original_rewards
        all_attr = remove_np_value(all_attr)
        return all_attr

    def __gt__(self, other):
        """Override gt for sorted according to performance attr."""
        return self.rewards > other.rewards

    @property
    def uid(self):
        """Uid for record. ReadOnly."""
        return "{}_{}".format(self.step_name, self.worker_id)

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
            value = json.loads(value)
        value = remove_np_value(value)
        self._desc = value

    @property
    def hps(self):
        """Get hps."""
        return self._hps

    @hps.setter
    def hps(self, value):
        """Set hps and parse value into dict."""
        if isinstance(value, str):
            value = json.loads(value)
        value = remove_np_value(value)
        self._hps = value

    @property
    def performance(self):
        """Get performance."""
        return self._performance

    @performance.setter
    def performance(self, value):
        """Set performance and parse value into dict."""
        if isinstance(value, str):
            value = json.loads(value)
        value = remove_np_value(value)
        self._performance = value
        self._cal_rewards()

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
        return self._rewards

    def _cal_rewards(self):
        if not self.performance:
            return None
        if isinstance(self.performance, list):
            return self.performance
        if not self.objective_keys:
            self._objective_keys = list(self.performance.keys())
        res = []
        res_ori = []
        for obj in self.objective_keys:
            if isinstance(obj, int):
                obj = list(self.performance.keys())[obj]
            value = self.performance.get(obj)
            ori_value = value
            # if value is None:
            #     raise ValueError("objective_keys in search_algorithm should be the same in trainer.metrics.")
            if self.objectives.get(obj) == "MIN":
                value = -value
            res.append(value)
            res_ori.append(ori_value)
        self._original_rewards = res_ori[0] if len(res_ori) == 1 else res_ori
        self._rewards = res[0] if len(res) == 1 else res

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
                setattr(src_cls, key, remove_np_value(value))
        return src_cls

    def load_dict(self, src_dic):
        """Load values from dict."""
        if src_dic:
            for key, value in src_dic.items():
                setattr(self, key, remove_np_value(value))
        return self

    def init(self, step_name, worker_id, desc=None, hps=None, **kwargs):
        """Set reord initial values."""
        self.step_name = step_name
        self.worker_id = worker_id
        self.desc = desc
        self.hps = hps
        for key in kwargs:
            setattr(self, key, remove_np_value(kwargs[key]))
        return self

    def serialize(self):
        """Serialize record class into a dict."""
        return self.to_dict()
