# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run hyperparameter tuning on python programs."""
import sys
import yaml
import argparse
import importlib
from .func import tune
from modnas.utils import exec_file


def tune_prog(progname=None, funcname=None, config=None, options=None, hparams=None, prog_args=None):
    """Run hyperparameter tuning on python programs."""
    prog_args = [] if prog_args is None else prog_args
    if isinstance(hparams, str):
        hparams = yaml.load(hparams, Loader=yaml.FullLoader)
    hp_dict = hparams if hparams is not None else {}
    sys.argv[:] = prog_args[:]
    if progname is None:
        prog_spec, prog_args = prog_args[0], prog_args[1:]
    else:
        prog_spec = progname
    prog_spec = prog_spec.split(':')
    exec_name = prog_spec[0]
    funcname = funcname or (None if len(prog_spec) == 1 else prog_spec[1])
    if exec_name.endswith('.py'):
        mod = exec_file(exec_name)
    else:
        mod = importlib.import_module(exec_name)
    if funcname is None:
        for k in mod.keys():
            if not k.startswith('_'):
                funcname = k
                break
    func = mod.__dict__.get(funcname)
    entry_name = '{}:{}'.format(exec_name, funcname)
    if func is None:
        raise ValueError('entrypoint {} not exist'.format(entry_name))
    options = options or []
    options.append({'defaults': {'name': entry_name}})
    return tune(func, *prog_args, tune_config=config, tune_options=options, tuned_args=hp_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune program hyperaparameters')
    parser.add_argument('-c', '--config', default=None, help="yaml config file")
    parser.add_argument('-f', '--funcname', default=None, help="name of the tuned function")
    parser.add_argument('-p', '--hparams', default=None, help="hyperaparameters config")
    parser.add_argument('-o', '--options', default=None, help="config override options")
    (opts, args) = parser.parse_known_args()
    tune_prog(**vars(opts), prog_args=args)
