# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import sys
import time
import inspect
import importlib
import numpy as np
import hashlib
from functools import partial
from modnas.version import __version__
from .logging import get_logger
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


logger = get_logger('utils')


def import_file(path, name=None):
    """Import modules from file."""
    spec = importlib.util.spec_from_file_location(name or '', path)
    module = importlib.util.module_from_spec(spec)
    if name:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exec_file(path):
    """Execute file and return globals."""
    with open(path, 'rb') as fp:
        code = compile(fp.read(), path, 'exec')
    globs = {
        '__file__': path,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }
    exec(code, globs, None)
    return globs


def import_modules(modules):
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
    return '{}.{}'.format(time.strftime('%Y%m%d', time.localtime()), hashlib.sha1(str(config).encode()).hexdigest()[:4])


def merge_config(src, dest, extend_list=True, overwrite=True):
    """Return merged config."""
    if isinstance(src, dict) and isinstance(dest, dict):
        for k, v in dest.items():
            if k not in src:
                src[k] = v
                logger.debug('merge_config: add key %s' % k)
            else:
                src[k] = merge_config(src[k], v, extend_list, overwrite)
    elif isinstance(src, list) and isinstance(dest, list) and extend_list:
        logger.debug('merge_config: extend list: %s + %s' % (src, dest))
        src.extend(dest)
    elif overwrite:
        logger.debug('merge_config: overwrite: %s -> %s' % (src, dest))
        src = dest
    return src


def env_info():
    """Return environment info."""
    info = {
        'platform': sys.platform,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'modnas': __version__,
    }
    return 'env info: {}'.format(', '.join(['{k}={{{k}}}'.format(k=k) for k in info])).format(**info)


def check_config(config, defaults=None):
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

    def __getattr__(self, item):
        """Return no-op."""
        def noop(*args, **kwargs):
            pass

        return noop


def get_writer(log_dir, enabled=False):
    """Return a new writer."""
    if enabled:
        if SummaryWriter is None:
            raise ValueError('module SummaryWriter is not found')
        writer = SummaryWriter(log_dir)
    else:
        writer = DummyWriter()
    return writer


def copy_members(dest, src, excepts=None, skip_private=True, method=True):
    """Copy member methods from src to dest."""
    for attr, mem in inspect.getmembers(src):
        if excepts is not None and attr in excepts:
            continue
        if skip_private and attr.startswith('_'):
            continue
        if method and not inspect.ismethod(mem):
            continue
        setattr(dest, attr, mem)


def get_same_padding(kernel_size):
    """Return SAME padding size for convolutions."""
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


class AverageMeter():
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(sec):
    """Return formatted time in seconds."""
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d h %d m %d s" % (h, m, s)


def format_key(key, title=True):
    """Return formatted key."""
    key = ' '.join(key.split('_'))
    return key.title() if title and key.islower() else key


def format_value(value, binary=False, div=None, factor=None, prec=2, unit=True, to_str=False):
    """Return formatted value."""
    if value is None:
        return None
    if not hasattr(value, '__truediv__'):
        return value
    units = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    div = (1024. if binary else 1000.) if div is None else div
    if factor is None:
        factor = 0
        tot_div = 1
        while value > tot_div:
            factor += 1
            tot_div *= div
    else:
        tot_div = div ** factor
    value = round(value / tot_div, prec)
    if not to_str and not unit:
        return value
    return '{{:.{}f}}'.format(prec).format(value) + (units[factor] if unit else '')


def format_dict(dct, sep=None, kv_sep=None, fmt_key=None, fmt_val=None):
    """Return formatted dict."""
    sep = sep or ' | '
    kv_sep = kv_sep or ':'
    fmt_vals = None if fmt_val is False else (fmt_val if isinstance(fmt_val, dict) else {})
    fmt_val = fmt_val if callable(fmt_val) else partial(format_value, unit=False, factor=0, prec=4, to_str=True)
    fmt_key = fmt_key if callable(fmt_key) else None if fmt_key is False else format_key
    val_dct = {k: v if fmt_vals is None else fmt_vals.get(k, fmt_val)(v) for k, v in dct.items()}
    return sep.join(['{}{} {{{}}}'.format(fmt_key(k) if fmt_key else k, kv_sep, k) for k in dct]).format(**val_dct)


class ETAMeter():
    """ETA Meter."""

    def __init__(self, total_steps, cur_steps=-1, time_fn=None):
        self.time_fn = time_fn or time.perf_counter
        self.total_steps = total_steps
        self.last_step = cur_steps
        self.last_time = self.time_fn()
        self.speed = None

    def start(self):
        """Start timing."""
        self.last_time = self.time_fn()

    def set_step(self, step):
        """Set current step."""
        self.speed = (step - self.last_step) / (self.time_fn() - self.last_time + 1e-7)
        self.last_step = step
        self.last_time = self.time_fn()

    def step(self, n=1):
        """Increment current step."""
        self.speed = n / (self.time_fn() - self.last_time + 1e-7)
        self.last_step += n
        self.last_time = self.time_fn()

    def eta(self):
        """Return ETA in seconds."""
        if self.speed is None:
            return 0
        return (self.total_steps - self.last_step) / (self.speed + 1e-7)

    def eta_fmt(self):
        """Return formatted ETA."""
        return format_time(self.eta())
