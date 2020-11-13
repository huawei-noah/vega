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
"""Utils for profiling status."""

import pprint
from collections import deque
from time import time
from absl import logging
import numpy as np


class LoopTracker(object):
    """
    Timekeeping.

    contains:
        1) with `enter`-> `exit`;
        2) loop between current and next `exit`.
    """

    def __init__(self, length):
        """Initialize."""
        self.with_time_list = deque(maxlen=length)
        self.loop_time_list = deque(maxlen=length)
        self.loop_point = None

    def __enter__(self):
        """Enter."""
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit."""
        self.end = time()
        self.with_time_list.append(self.end - self.start)

        if not self.loop_point:
            self.loop_point = self.end
        else:
            self.loop_time_list.append(self.end - self.loop_point)
            self.loop_point = self.end

    def average(self, time_name):
        """Mean time of `with` interaction, and loop time as well."""
        if time_name == "enter":
            return np.nanmean(self.with_time_list) * 1000 if self.loop_time_list else np.nan
        elif time_name == "loop":
            return np.nanmean(self.loop_time_list) * 1000 if self.loop_time_list else np.nan
        else:
            return np.nan


class SingleTracker(object):
    """Single time tracker, only profiling the enter time used."""

    def __init__(self, length):
        """Initialize."""
        self.with_time_list = deque(maxlen=length)
        self.start = time()

    def __enter__(self):
        """Enter."""
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit."""
        self.with_time_list.append(time() - self.start)

    def average(self):
        """Mean time of `with` interaction."""
        if not self.with_time_list:
            return np.nan
        return np.nanmean(self.with_time_list) * 1000


class PredictStats(object):
    """
    Predictor status records.

    handle the wait and inference time of predictor.
    """

    def __init__(self):
        """Init with default value."""
        self.obs_wait_time = 0.0
        self.inference_time = 0.0
        self.iters = 0.0

    def get(self):
        """Get agent status and clear the buffer."""
        ret = {
            "mean_predictor_wait_ms": self.obs_wait_time * 1000 / self.iters,
            "mean_predictor_infer_ms": self.inference_time * 1000 / self.iters,
        }
        self.reset()
        return ret

    def reset(self):
        """Reset buffer."""
        self.obs_wait_time = 0.0
        self.inference_time = 0.0
        self.iters = 0.0


class AgentStats(object):
    """
    Agent status records.

    handle the env.step and inference time of Agent
    """

    def __init__(self):
        """Init with default value."""
        self.env_step_time = 0.0
        self.inference_time = 0.0
        self.iters = 0.0

    def get(self):
        """Get agent status and clear the buffer."""
        ret = {
            "mean_env_step_time_ms": self.env_step_time * 1000 / self.iters,
            "mean_inference_time_ms": self.inference_time * 1000 / self.iters,
            "iters": self.iters,
        }

        self.reset()
        return ret

    def reset(self):
        """Reset buffer."""
        self.env_step_time = 0.0
        self.inference_time = 0.0
        self.iters = 0


class AgentGroupStats(object):
    """
    AgentGroup status records.

    handle the env.step and inference time of AgentGroup
    the status could been make sence within once explore

    There should been gather by logger or others.
    """

    def __init__(self, n_agents, env_type):
        """Init with default value."""
        self.env_step_time = 0.0
        self.inference_time = 0.0
        self.iters = 0
        self.explore_time_in_epi = 0.0
        self.wait_model_time = 0.0
        self.restore_model_time = 0.0

        self.n_agents = n_agents
        self.env_api_type = env_type
        self._stats = dict()
        self.ext_attr = "mean_explore_reward"

    def update_with_agent_stats(self, agent_stats: list):
        """Update agent status to agent group."""
        _steps = [sta["mean_env_step_time_ms"] for sta in agent_stats]
        _infers = [sta["mean_inference_time_ms"] for sta in agent_stats]
        _iters = [sta["iters"] for sta in agent_stats]
        self._stats.update(
            {
                "mean_env_step_ms": np.nanmean(_steps),
                "mean_inference_ms": np.nanmean(_infers),
                "iters": np.max(_iters),  # multi-agent use max steps in group.
            }
        )

        if self.ext_attr in agent_stats[0] and agent_stats[0][self.ext_attr] is not np.nan:
            self._stats.update(
                {self.ext_attr: np.nanmean([sta[self.ext_attr] for sta in agent_stats])})

    def get(self):
        """Get the newest one-explore-status of agent group."""
        self._stats.update(
            {
                "explore_ms": self.explore_time_in_epi * 1000,
                "wait_model_ms": self.wait_model_time * 1000,
                "restore_model_ms": self.restore_model_time * 1000,
            }
        )
        # use unified api, agent group will record the interaction times.
        if self.iters > 0:
            self._stats.update(
                {
                    "mean_env_step_time_ms": self.env_step_time * 1000 / self.iters,
                    "mean_inference_time_ms": self.inference_time * 1000 / self.iters,
                    "iters": self.iters,
                }
            )

        self.reset()
        return self._stats

    def reset(self):
        """Reset buffer."""
        self.env_step_time = 0.0
        self.inference_time = 0.0
        self.iters = 0
        self.explore_time_in_epi = 0.0
        self.wait_model_time = 0.0
        self.restore_model_time = 0.0


class TimerRecorder(object):
    """Recorder for time used."""

    def __init__(self, style, maxlen=50, fields=("send", "recv")):
        self.style = style
        self.fields = fields
        self.track_stub = {item: deque(maxlen=maxlen) for item in fields}

        self.report_interval = 30  # s
        self.last_report_time = 0  # -self.report_interval

    def append(self, **kwargs):
        """Update record items."""
        for _k, _val in kwargs.items():
            if _k in self.track_stub:
                self.track_stub[_k].append(_val)

    def get_metric(self, fields):
        """Fetch the newest time record."""
        ret = dict()
        for _task in fields:
            if not self.track_stub[_task]:
                continue

            ret.update({
                "{}_{}_mean_ms".format(self.style, _task):
                    1000 * np.nanmean(self.track_stub[_task]),
                "{}_{}_max_ms".format(self.style, _task):
                    1000 * np.max(self.track_stub[_task]),
                "{}_{}_min_ms".format(self.style, _task):
                    1000 * np.min(self.track_stub[_task]),
            })

        return ret

    def report_if_need(self, field_sets=None, **kwargs):
        """Rreport the time metric if need."""
        if time() - self.last_report_time >= self.report_interval:
            to_log = self.get_metric(field_sets or self.fields)
            if kwargs:
                to_log.update(kwargs)
            to_log_format = pprint.pformat(to_log, indent=0, width=1)

            logging.debug("\n{}\n".format(to_log_format))
            self.last_report_time = time()
