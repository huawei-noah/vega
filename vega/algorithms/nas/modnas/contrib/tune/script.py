# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run hyperparameter tuning on python scripts."""

import os
import sys
import yaml
import optparse
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


def tune_script():
    """Run hyperparameter tuning on python scripts."""
    usage = "tune_script.py [options] script [args] ..."
    parser = optparse.OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option('-n', '--name', default=None, help="name of the experiment")
    parser.add_option('-f', '--func', default=None, help="name of the tuned function")
    parser.add_option('-c', '--config', default=None, help="yaml config file")
    parser.add_option('-p', '--hparam', default=None, help="hparam")
    parser.add_option('-o', '--config_override', action="append", type=str, default=None, help="override config")

    (options, args) = parser.parse_args()

    if not sys.argv[1:] or len(args) == 0:
        parser.print_usage()
        sys.exit(2)

    progname = args[0]
    options = vars(options)
    funcname = options.pop('func', None)
    opts = {
        'name': options.pop('name') or os.path.basename(progname),
        'config': [_default_hptune_config.copy()],
    }
    hparam = options.pop('hparam')
    if hparam is None and 'hp_space' not in opts['config']:
        raise RuntimeError('Argument -p is required')
    hp_dict = yaml.load(hparam, Loader=yaml.FullLoader)
    opts['config'][0]['hp_space'] = hp_dict
    opts['config'].extend(options.pop('config') or [])
    overrides = options.pop('config_override')
    if overrides is None:
        overrides = []
    opts.update(options)

    sys.argv[:] = args[:]
    sys.path.insert(0, os.path.dirname(progname))
    with open(progname, 'rb') as fp:
        code = compile(fp.read(), progname, 'exec')
    globs = {
        '__file__': progname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }
    exec(code, globs, None)
    if funcname is None:
        for k in globs.keys():
            if not k.startswith('_'):
                funcname = k
                break
    func = globs.get(funcname, None)
    if func is None:
        raise ValueError('function {} not exist'.format(funcname))

    tune_res = run_hptune(**opts, measure_fn=lambda hp: func(**hp))
    best_hparams = list(tune_res.values())[0]['best_hparams']
    logger.info('tune_script: best hparams: {}'.format(dict(best_hparams)))


if __name__ == '__main__':
    tune_script()
