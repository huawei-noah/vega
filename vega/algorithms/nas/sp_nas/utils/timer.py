# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""A progress bar which can print the progress."""
from mmcv.utils import Timer as Timer_, TimerError
from time import time
import torch


class Time:
    """Get Time class.."""

    def __init__(self, cuda_mode=False):
        self._cuda_mode = cuda_mode
        if cuda_mode:
            self._time = torch.cuda.Event(enable_timing=True)
            self._time.record()
        else:
            self._time = time()

    def elapsed_time(self, other):
        """Get elapsed time.

        :return: Time in seconds.
        :rtype: float
        """
        if not isinstance(other, Time):
            raise TypeError('{} should be an object of {}'.format(other, self))
        if self._cuda_mode != other._cuda_mode:
            raise RuntimeError('The cuda mode must be same')
        if self._cuda_mode:
            torch.cuda.synchronize()
            t = self._time.elapsed_time(other._time) / 1000  # convert to ms
        else:
            t = other._time - self._time
        return t


class Timer(Timer_):
    """Get Timer class."""

    def __init__(self, start=True, print_tmpl=None, cuda_mode=False):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        self._cuda_mode = cuda_mode
        if start:
            self.start()

    def start(self):
        """Record start time."""
        if not self._is_running:
            self._t_start = Time(self._cuda_mode)
            self._is_running = True
        self._t_last = Time(self._cuda_mode)

    def since_start(self):
        """Compute total time since the timer is started.

        :return: Time in seconds.
        :rtype: float
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = Time(self._cuda_mode)
        return self._t_start.elapsed_time(self._t_last)

    def since_last_check(self):
        """Get time since the last checking. Either :func:`since_start` or :func:`since_last_check` is a checking operation.

        :return: Time in seconds.
        :rtype: float
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = self._t_last.elapsed_time(Time(self._cuda_mode))
        self._t_last = Time(self._cuda_mode)
        return dur
