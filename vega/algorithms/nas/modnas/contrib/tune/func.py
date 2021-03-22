# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Wrappers that run hyperparameter tuning on functions."""

import copy
from functools import partial
from modnas.utils.wrapper import run_hptune
from modnas.utils.logging import get_logger


logger = get_logger()


_default_hptune_config = {
    'optim': {
        'type': 'RandomSearchOptim'
    },
    'estimator': {
        'tune': {
            'type': 'HPTuneEstim',
            'epochs': -1,
        }
    }
}


def tune(func, *args, tune_config=None, tune_options=None, tuned_args=None, tuned=False, **kwargs):
    """Return tuned hyperparameters for given function."""
    tuned_args = tuned_args or {}

    def parse_hp(hp):
        fn_kwargs = copy.deepcopy(kwargs)
        hp_kwargs = {k: v for k, v in hp.items() if not k.startswith('#')}
        fn_kwargs.update(hp_kwargs)
        fn_args = [hp.get('#{}'.format(i), v) for i, v in enumerate(args)]
        return fn_args, fn_kwargs

    def measure_fn(hp):
        fn_args, fn_kwargs = parse_hp(hp)
        return func(*fn_args, **fn_kwargs)

    opts = {
        'name': func.__name__,
        'config': [_default_hptune_config.copy()],
    }
    opts['config'][0]['hp_space'] = tuned_args
    opts['config'].extend(tune_config or [])
    opts['config_override'] = tune_options
    tune_res = run_hptune(measure_fn=measure_fn, **opts)
    best_hparams = list(tune_res.values())[0]['best_hparams']
    logger.info('tune: best hparams: {}'.format(dict(best_hparams)))
    ret = parse_hp(best_hparams)
    if tuned:
        fn_args, fn_kwargs = ret
        return func(*fn_args, **fn_kwargs)
    return ret


def tune_template(*args, **kwargs):
    """Run hyperparameter tuning on given function."""
    tuned_args = {}
    tuned_args.update(kwargs)
    tuned_args.update({'#{}'.format(i): v for i, v in enumerate(args) if v is not None})
    return lambda func: partial(func, tuned_args=tuned_args)


def tunes(*args, **kwargs):
    """Return hyperparameter tuner decorator."""
    def wrapper(func):
        return partial(tune, func, *args, **kwargs)
    return wrapper


tuned = partial(tune, tuned=True)
