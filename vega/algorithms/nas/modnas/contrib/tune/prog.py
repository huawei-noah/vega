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

"""Run hyperparameter tuning on python programs."""
import sys
import yaml
import argparse
import importlib
from .func import tune


def tune_prog(progname=None, funcname=None, config=None, options=None, hparams=None, prog_args=None):
    """Run hyperparameter tuning on python programs."""
    prog_args = [] if prog_args is None else prog_args
    if isinstance(hparams, str):
        hparams = yaml.safe_load(hparams)
    hp_dict = hparams if hparams is not None else {}
    sys.argv[:] = prog_args[:]
    if progname is None:
        prog_spec, prog_args = prog_args[0], prog_args[1:]
    else:
        prog_spec = progname
    prog_spec = prog_spec.split(':')
    exec_name = prog_spec[0]
    funcname = funcname or (None if len(prog_spec) == 1 else prog_spec[1])
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
