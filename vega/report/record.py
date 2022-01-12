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

"""Report."""

import json
import logging
from datetime import datetime
from vega.common.utils import remove_np_value
from vega.common import Status, JsonEncoder, DatatimeFormatString

logger = logging.getLogger(__name__)


class ReportRecord(object):
    """Record Class to record all data in one search loop."""

    def __init__(self, step_name=None, worker_id=None, **kwargs):
        self.step_name = step_name
        self.worker_id = worker_id
        self._desc = None
        self._hps = None
        self._performance = {}
        self.checkpoint_path = None
        self.model_path = None
        self.weights_file = None
        self.num_epochs = 1
        self.current_epoch = 1
        self._objectives = {}
        self._objective_keys = None
        self._rewards = []
        self.runtime = None
        self._original_rewards = None
        self._start_time = datetime.now()
        self._end_time = None
        self._status = Status.running
        self.message = None
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
        if isinstance(other, ReportRecord):
            return self.uid == other.uid
        elif isinstance(other, dict):
            data = json.loads(json.dumps(self.to_dict(), cls=JsonEncoder))
            _other = json.loads(json.dumps(other, cls=JsonEncoder))
            for item in _other:
                if item not in data or data[item] != _other[item]:
                    if data[item] in [None, [None]] and _other[item] in [None, [None]]:
                        continue
                    if item in ["original_rewards"]:
                        continue
                    return False
            return True
        else:
            return False

    def __repr__(self):
        """Override repr, output all record attrs."""
        return json.dumps(self.to_dict(), cls=JsonEncoder)

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
    def start_time(self):
        """Start time."""
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        """Start time."""
        if isinstance(value, str):
            self._start_time = datetime.strptime(value, DatatimeFormatString)
        else:
            self._start_time = value

    @property
    def end_time(self):
        """End time."""
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        """End time."""
        if isinstance(value, str):
            self._end_time = datetime.strptime(value, DatatimeFormatString)
        else:
            self._end_time = value

    @property
    def status(self):
        """End time."""
        return self._status

    @status.setter
    def status(self, value):
        """End time."""
        if isinstance(value, str):
            self._status = Status(value)
        else:
            self._status = value

    @property
    def performance(self):
        """Get performance."""
        return self._performance

    @performance.setter
    def performance(self, value):
        """Set performance and parse value into dict."""
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            self._performance.update(value)
        elif value is not None:
            logger.warn(f"Invalid record performance value: {value}")

    @property
    def objective_keys(self):
        """Get objective_keys."""
        return self._objective_keys

    @objective_keys.setter
    def objective_keys(self, value):
        """Set objective_keys."""
        self._objective_keys = value if isinstance(value, list) else [value]

    @property
    def objectives(self):
        """Get objective."""
        return self._objectives

    @objectives.setter
    def objectives(self, value):
        """Set objective_keys."""
        if isinstance(value, dict):
            self._objectives.update(value)
        elif value is not None:
            logger.warn(f"Invalid record objectives value: {value}")

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
        for k in self.performance.keys():
            if k not in self.objectives and k in ['flops', 'params', 'latency']:
                self._objectives.update({k: 'MIN'})
        for obj in self.objective_keys:
            if isinstance(obj, int):
                obj = list(self.performance.keys())[obj]
            value = self.performance.get(obj)
            ori_value = value
            if self.objectives and self.objectives.get(obj) == "MIN":
                value = -value
            res.append(value)
            res_ori.append(ori_value)
        self._original_rewards = res_ori[0] if len(res_ori) == 1 else res_ori
        self._rewards = res[0] if len(res) == 1 else res

    @property
    def rewards_compeleted(self):
        """Get reward compeleted(ReadOnly)."""
        if isinstance(self._rewards, list):
            if len(self._rewards) == 0:
                return False
            for reward in self._rewards:
                if reward is None:
                    return False
            return True
        else:
            return self._rewards is not None

    def load_dict(self, src_dic):
        """Load values from dict."""
        if src_dic:
            for key, value in src_dic.items():
                if key in ["original_rewards", "rewards"]:
                    continue
                update_flag = isinstance(value, dict) and isinstance(getattr(self, key), dict)
                update_flag = update_flag and key not in ["desc"]
                if update_flag:
                    for value_key, value_value in value.items():
                        getattr(self, key)[value_key] = remove_np_value(value_value)
                else:
                    setattr(self, key, remove_np_value(value))
        self._cal_rewards()
        return self

    def serialize(self):
        """Serialize record class into a dict."""
        return self.to_dict()
