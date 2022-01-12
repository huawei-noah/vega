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

import sys
import re
import os
import time
import inspect
import importlib
import hashlib
from functools import partial
from typing import Callable, Dict, List, Optional, Union, Any
import numpy as np
from modnas.version import __version__
from modnas import backend as be
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None
from .logging import get_logger


logger = get_logger('utils')


def import_file(path, name=None):
    """Import modules from file."""
    spec = importlib.util.spec_from_file_location(name or '', path)
    module = importlib.util.module_from_spec(spec)
    if name:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_value(value, pattern):
    """Check value."""
    if isinstance(value, str) and len(re.compile(pattern).findall(value)) > 0:
        raise ValueError(f"{value} contains invalid characters.")


def import_modules(modules: List[str]) -> None:
    """Import modules by name."""
    if modules is None:
        return
    if isinstance(modules, str):
        modules = [modules]
    for m in modules:
        name = None
        if isinstance(m, str):
            path = m
        elif isinstance(m, (list, tuple)):
            name, path = m
        else:
            raise ValueError('Invalid import spec')
        if path.endswith('.py'):
            mod = import_file(path)
        else:
            mod = importlib.import_module(path)
        if name:
            sys.modules[name] = mod


def get_exp_name(config):
    """Return experiment name."""
    return '{}.{}'.format(time.strftime('%Y%m%d', time.localtime()),
                          hashlib.sha256(str(config).encode()).hexdigest()[:4])


def env_info() -> str:
    """Return environment info."""
    info = {
        'platform': sys.platform,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'modnas': __version__,
        'backend': None if be.backend() is None else '{{{}}}'.format(getattr(be, 'version', lambda: None)()),
    }
    return 'env info: {}'.format(', '.join(['{k}={{{k}}}'.format(k=k) for k in info])).format(**info)


def check_config(config: Dict, defaults: Optional[Any] = None) -> None:
    """Check config and set default values."""
    def check_field(config, field, default):
        cur_key = ''
        idx = -1
        keys = field.split('.')
        cur_dict = config
        try:
            for idx in range(len(keys)):
                cur_key = keys[idx]
                if cur_key == '*':
                    if isinstance(cur_dict, list):
                        key_list = ['#{}'.format(i) for i in range(len(cur_dict))]
                    else:
                        key_list = cur_dict.keys()
                    for k in key_list:
                        keys[idx] = k
                        nfield = '.'.join(keys)
                        if check_field(config, nfield, default):
                            return True
                    return False
                if cur_key.startswith('#'):
                    cur_key = int(cur_key[1:])
                cur_dict = cur_dict[cur_key]
        except KeyError:
            if idx != len(keys) - 1:
                logger.warning('check_config: key \'{}\' in field \'{}\' missing'.format(cur_key, field))
            else:
                logger.warning('check_config: setting field \'{}\' to default: {}'.format(field, default))
                cur_dict[cur_key] = default
        return False

    default_config = {
        'backend': 'torch',
        'device_ids': 'all',
        'estim.*.arch_update_epoch_start': 0,
        'estim.*.arch_update_epoch_intv': 1,
        'estim.*.arch_update_intv': -1,
        'estim.*.arch_update_batch': 1,
        'estim.*.metrics': 'ValidateMetrics',
    }
    default_config.update(defaults or {})

    for key, val in default_config.items():
        check_field(config, key, val)


class DummyWriter():
    """A no-op writer."""

    def __getattr__(self, item: str) -> Callable:
        """Return no-op."""
        def noop(*args, **kwargs):
            pass

        return noop


def get_writer(log_dir: str, enabled: bool = False) -> DummyWriter:
    """Return a new writer."""
    if enabled:
        if SummaryWriter is None:
            raise ValueError('module SummaryWriter is not found')
        writer = SummaryWriter(log_dir)
    else:
        writer = DummyWriter()
    return writer


def copy_members(
    dest: Any, src: Any, includes: Optional[List[str]] = None, excepts: Optional[List[str]] = None,
    skip_private: bool = True, method: bool = True
) -> None:
    """Copy member methods from src to dest."""
    for attr, mem in inspect.getmembers(src):
        if includes is not None and attr not in includes:
            continue
        if excepts is not None and attr in excepts:
            continue
        if skip_private and attr.startswith('_'):
            continue
        if method and not inspect.ismethod(mem):
            continue
        setattr(dest, attr, mem)


def get_same_padding(kernel_size: int) -> int:
    """Return SAME padding size for convolutions."""
    if isinstance(kernel_size, tuple):
        if len(kernel_size) != 2:
            raise ValueError('invalid kernel size: %s' % kernel_size)
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    if isinstance(kernel_size, int):
        if kernel_size % 2 > 0:
            return kernel_size // 2
        else:
            raise ValueError('kernel size should be odd number')
    else:
        raise ValueError('kernel size should be either `int` or `tuple`')


class AverageMeter():
    """Compute and store the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update statistics."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(sec: float) -> str:
    """Return formatted time in seconds."""
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d h %d m %d s" % (h, m, s)


def format_key(key: str, title: bool = True) -> str:
    """Return formatted key."""
    key = ' '.join(key.split('_'))
    return key.title() if title and key.islower() else key


def format_value(
    value: Union[str, float, int], binary: bool = False, div: Optional[int] = None,
    factor: Optional[int] = None, prec: int = 2, unit: bool = True, to_str: bool = False
) -> Union[str, float]:
    """Return formatted value."""
    if value is None:
        return None
    if not hasattr(value, '__truediv__'):
        return value
    f_value = float(value)
    units = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    _div = (1024 if binary else 1000) if div is None else div
    _factor = factor or 0
    if _factor:
        tot_div = _div ** _factor
    else:
        tot_div = 1
        while f_value > tot_div * _div:
            _factor += 1
            tot_div *= _div
    f_value = round(f_value / tot_div, prec)
    if not to_str and not unit:
        return f_value
    return '{{:.{}f}}'.format(prec).format(f_value) + (units[_factor] if unit else '')


def format_dict(
    dct: Dict[str, Union[float, str]], sep: str = ' | ', kv_sep: str = ': ',
    fmt_key: Optional[Callable] = None, fmt_val: Optional[Callable] = None
) -> str:
    """Return formatted dict."""
    fmt_vals = None if fmt_val is False else (fmt_val if isinstance(fmt_val, dict) else {})
    _fmt_val = fmt_val if callable(fmt_val) else partial(format_value, unit=False, factor=0, prec=4, to_str=True)
    _fmt_key = fmt_key if callable(fmt_key) else None if fmt_key is False else format_key
    val_dct = {k: v if fmt_vals is None else fmt_vals.get(k, _fmt_val)(v) for k, v in dct.items()}
    return sep.join(['{}{}{{{}}}'.format(_fmt_key(k) if _fmt_key else k, kv_sep, k) for k in dct]).format(**val_dct)


class ETAMeter():
    """ETA Meter."""

    def __init__(self, total_steps: int, cur_steps: int = -1, time_fn: Optional[Callable] = None) -> None:
        self.time_fn = time_fn or time.perf_counter
        self.total_steps = total_steps
        self.last_step = cur_steps
        self.last_time = self.time_fn()
        self.speed = 0.

    def start(self) -> None:
        """Start timing."""
        self.last_time = self.time_fn()

    def set_step(self, step):
        """Set current step."""
        self.speed = (step - self.last_step) / (self.time_fn() - self.last_time + 1e-7)
        self.last_step = step
        self.last_time = self.time_fn()

    def step(self, n: int = 1) -> None:
        """Increment current step."""
        self.speed = n / (self.time_fn() - self.last_time + 1e-7)
        self.last_step += n
        self.last_time = self.time_fn()

    def eta(self) -> float:
        """Return ETA in seconds."""
        if self.speed < 1e-7:
            return 0
        return (self.total_steps - self.last_step) / (self.speed + 1e-7)

    def eta_fmt(self) -> str:
        """Return formatted ETA."""
        return format_time(self.eta())
