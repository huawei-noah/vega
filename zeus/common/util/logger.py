# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Utils for the information recording during the training process.

Use absl.logging with an default formatter.
"""
import json
import os
import threading
import platform
from collections import deque
from time import time

import numpy as np
import logging as normal_logging
from absl import logging
from .profile_stats import LoopTracker, SingleTracker
from .local_data import LocalDataWriter
from zeus.common.util.common import get_host_ip
LOG_DEFAULT_PATH = os.path.join(os.path.expanduser("~"), "xt_archive")


class HostnameFilter(normal_logging.Filter):
    """Host name filter."""

    hostname = platform.node()
    hostip = get_host_ip()

    def filter(self, record):
        """Filter."""
        record.hostname = HostnameFilter.hostname
        record.hostip = HostnameFilter.hostip
        return True


def set_logging_format():
    """Set logging format."""
    fmt = "%(levelname)s [%(hostname)s %(hostip)s] %(asctime)-8s: %(message)s"
    date_fmt = "%b %d %H:%M:%S"
    formatter = normal_logging.Formatter(fmt, date_fmt)
    handler = logging.get_absl_handler()
    handler.setFormatter(formatter)
    handler.addFilter(HostnameFilter())
    logging.use_absl_handler()


VERBOSITY_MAP = {
    "fatal": logging.FATAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

BOARD_GROUP_MAP = {
    "train_loss": "learner",
    "train_reward_avg": "learner",
    "step_per_second": "learner",
    "mean_wait_sample_ms": "learner",
    "mean_prepare_data_ms": "learner",
    "mean_train_time_ms": "learner",
    "mean_loop_time_ms": "learner",
    "mean_env_step_ms": "explorer",
    "mean_inference_ms": "explorer",
    "mean_explore_ms": "explorer",
    "mean_wait_model_ms": "explorer",
    "mean_explore_reward": "explorer",
    "mean_predictor_wait_ms": "predictor",
    "mean_predictor_infer_ms": "predictor",
    # "bm_rewards": "benchmark",
    # "eval_criteria": "benchmark",
}

# pylint: disable=C0330


class Logger(object):
    """Logger for record training's information."""

    def __init__(self, workspace):
        """Init with template records."""
        self.abs_start = time()
        self.loop_start = time()
        self.records = {
            "train_reward": [],
            "step": [],
            "train_count": [],
            "train_loss": [],
        }
        self._workspace = workspace
        self.train_timer = LoopTracker(20)
        self.wait_sample_timer = SingleTracker(20)
        self.prepare_data_timer = SingleTracker(20)

    @property
    def elapsed_time(self):
        """Elapsed time set as an property."""
        return time() - self.abs_start

    def update(self, **kwargs):
        """Update value could been rewrite."""
        self.records.update(kwargs)

    def record(self, **kwargs):
        """
        Record stuffs.

        Example: record(step=1, train_count=2, train_reward=1)
        """
        for _key, val in kwargs.items():
            if _key not in self.records.keys():
                self.records.update({_key: [val]})
            else:
                self.records[_key].append(val)

    def get_new_info(self):
        """To assemble newest records for display."""
        _info = {
            "train_count": np.nan,
            "step": np.nan,
            "mean_train_time_ms": self.train_timer.average("enter"),
            "mean_loop_time_ms": self.train_timer.average("loop"),
            "elapsed_time": self.elapsed_time,
        }
        for _key in (
            "train_loss",
            "train_count",
        ):
            if self.records[_key]:
                _info.update({_key: self.records[_key][-1]})

        if getattr(self, "wait_sample_timer"):
            _info.update({"mean_wait_sample_ms": self.wait_sample_timer.average()})
        if getattr(self, "prepare_data_timer"):
            _info.update({"mean_prepare_data_ms": self.prepare_data_timer.average()})

        if self.records["step"]:
            _cur_step = self.records["step"][-1]
            _info.update(
                {
                    "step": _cur_step,
                    "step_per_second": int(_cur_step / (time() - self.abs_start)),
                }
            )
        if self.records["train_reward"]:
            _info.update(
                {"train_reward_avg": np.mean(self.records["train_reward"][-100:])}
            )

        _extend_item = ("explore_won_rate", )
        for _item in _extend_item:
            if _item not in self.records.keys():
                continue
            _info.update({_item: self.records[_item]})

        return _info

    @property
    def train_reward_avg(self):
        """Train reward average could been property."""
        if not self.records["train_reward"]:
            return np.nan
        return np.mean(self.records["train_reward"][-100:])

    @property
    def train_reward(self):
        """Train reward."""
        if not self.records["train_reward"]:
            return np.nan
        return self.records["train_reward"][-1]

    def save_to_json(self, save_path=None, file_name="train_records.json"):
        """Save the records into json file, when experiment finished."""
        _save_path = save_path or self._workspace
        with open(os.path.join(_save_path, file_name), "w") as json_file:
            json.dump(self.records, json_file, cls=XtEncoder)


class XtEncoder(json.JSONEncoder):
    """Overwrite the json to support more data format."""

    def default(self, obj):
        """Set default value."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(XtEncoder, self).default(obj)


def time_to_str(sec):
    """Convert seconds to days, hours, minutes and seconds."""
    days, remainder = divmod(sec, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    _str = ""
    if days > 0:
        _str += "{:d} days, ".format(int(days))
    if hours > 0:
        _str += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        _str += "{:d} minutes, ".format(int(minutes))
    _str += "{:d} seconds".format(int(seconds))
    return _str


class StatsRecorder(threading.Thread):
    """
    StatsRecorder implemented with threading.Thread.

    Marge the status among predictor, trainer and explorer etc.
    And then, write into tensorboard with train workspace.
    """

    def __init__(
        self,
        msg_deliver,
        bm_args,
        workspace,
        bm_board=None,
        show_interval=10000,
        name="xt_stats",
        explore_deque_len=30,
    ):
        """Stats recorder inherit from threading.Thread, run with single thread."""
        threading.Thread.__init__(self, name=name)
        self.msg_deliver = msg_deliver
        self.bm_args = bm_args
        self.workspace = workspace
        self.bm_board = bm_board
        self._data = dict(train_count=np.nan, step=np.nan, elapsed_time=0)
        self._last_show_step = -9999
        self.show_interval = show_interval

        # stats from explorer
        self.explore_stats = {
            "mean_env_step_ms": deque(maxlen=explore_deque_len),
            "mean_inference_ms": deque(maxlen=explore_deque_len),
            "iters": deque(maxlen=explore_deque_len),
            "explore_ms": deque(maxlen=explore_deque_len),
            "wait_model_ms": deque(maxlen=explore_deque_len),
            "restore_model_ms": deque(maxlen=explore_deque_len),
            "mean_explore_reward": deque(maxlen=explore_deque_len)
        }

        self.local_data_writer = LocalDataWriter(
            os.path.join(self.workspace, "benchmark")
        )
        self.local_data_writer.add_new_train_event(self.bm_args)

    def update(self, **kwargs):
        """Update with new status received."""
        self._data.update(**kwargs)

    def record_explore_status(self, msg_data: dict):
        """Record message from explore."""
        for k, v in msg_data.items():
            if k not in self.explore_stats.keys():
                logging.debug("skip un-known status-{}: {}".format(k, v))
                continue
            self.explore_stats[k].append(v)

    def could_show_stats(self):
        """Check whether show or not."""
        if self._data.get("step", 0) - self._last_show_step >= self.show_interval:
            self._last_show_step = self._data.get("step", 0)
            return True
        return False

    @staticmethod
    def add_board_prefix(raw_key):
        """Add prefix for tensorboard visualization."""
        if raw_key in BOARD_GROUP_MAP.keys():
            key_in_group = "/".join([BOARD_GROUP_MAP.get(raw_key), raw_key])
        else:
            key_in_group = raw_key
        return key_in_group

    def assemble_records(self):
        """Assemble the data format for tensorboard."""
        record_list = list()
        for _key in BOARD_GROUP_MAP.keys():
            try:
                g_key = self.add_board_prefix(_key)
                if not self._data[_key]:
                    continue

                if np.nan is self._data[_key]:
                    continue

                record_list.append((g_key, self._data[_key], self._data["step"]))
            except KeyError:
                continue

        return record_list

    def show_recent_stats(self, row=4):
        """Display recent status."""
        # update explorer status firstly.
        for target_key in (
            "mean_env_step_ms",
            "mean_inference_ms",
            "explore_ms",
            "wait_model_ms",
            "restore_model_ms",
            "mean_explore_reward",
        ):
            if "mean_explore_reward" not in self.explore_stats:
                # extend explore reward
                continue

            display_name = (
                "mean_" + target_key
                if not target_key.startswith("mean")
                else target_key
            )
            if self.explore_stats[target_key]:
                self._data.update(
                    {display_name: np.nanmean(self.explore_stats[target_key])}
                )
                # sync mean_explore_reward to train reward, if need
                if target_key == "mean_explore_reward":
                    self._data.update({
                        "train_reward_avg": np.nanmean(self.explore_stats[target_key])})
                    self._data.pop(target_key)

        _info = self._data
        _train_count = _info["train_count"]
        _step = _info["step"]
        _elapsed_time = _info["elapsed_time"]

        _str = "Train_count:{:>10} | Steps:{:>10} | Elapsed time: {}\n".format(
            _train_count, _step, time_to_str(_elapsed_time)
        )
        _show_items = 0
        skip_items = ("train_count", "step", "elapsed_time")
        for _key, val in sorted(_info.items()):
            if _key in skip_items:
                continue
            _show_items += 1
            _str += "{:<24}{:>10}".format(_key + ":", round(float(val), 6))
            _str += "\n" if _show_items % row == 0 else "\t"

        show_str = _str + "\n" if not _str.endswith("\n") else _str
        logging.info(show_str)

    def run(self):
        """Overwrite threading.Thread.run() function with True."""
        bm_record_key = ("train_reward", "eval_reward", "custom_criteria", "battle_won")
        while True:
            _stats = self.msg_deliver.recv()
            if _stats.get("ctr_info"):  # msg from explore & broker
                if _stats.get("ctr_info").get("cmd") == "stats_msg":
                    logging.debug("recv explore status: {}.".format(_stats["data"]))
                    self.record_explore_status(_stats["data"])
            elif _stats.get("is_bm"):  # msg from benchmark
                # write benchmark result into local file
                self.local_data_writer.insert_records(_stats["data"])
                bm_data2board = list()
                from zeus.common.util.printer import print_immediately
                print_immediately(_stats)
                for item in _stats["data"]:
                    for k, val in item.items():
                        if k not in bm_record_key:
                            continue
                        to_str = "eval_won_rate" if k == "battle_won" else k
                        val = ("/".join(["benchmark", to_str]),
                               item[k], item["sample_step"])
                        bm_data2board.append(val)
                logging.debug("record bm stats: {}".format(bm_data2board))
                if self.bm_board and bm_data2board:
                    self.bm_board.insert_records(bm_data2board)

            else:  # msg from learner as default
                logging.debug("to update _stats: {}".format(_stats))
                self.update(**_stats)

            if not self.could_show_stats():
                continue

            self.show_recent_stats()

            # write into tensorboard
            if self.bm_board:
                records = self.assemble_records()
                # fetch benchmark information
                logging.debug("records: {} insert tensorboard".format(records))
                if records:
                    self.bm_board.insert_records(records)

    def __del__(self):
        """Delete."""
        if self.bm_board:
            self.bm_board.close()

        if self.local_data_writer:
            self.local_data_writer.close()
