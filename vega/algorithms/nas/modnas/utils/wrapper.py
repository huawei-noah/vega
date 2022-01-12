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

"""Wrapper for routine initialization and execution."""
import argparse
from collections import OrderedDict
from functools import partial
from modnas.core.event import EventManager
from modnas.core.param_space import ParamSpace
from modnas.registry.construct import build as build_con
from modnas.registry.callback import build as build_callback
from modnas.registry.export import build as build_exp
from modnas.registry.optim import build as build_optim
from modnas.registry.estim import build as build_estim
from modnas.registry.trainer import build as build_trainer
from modnas.registry import parse_spec, to_spec
from modnas.utils.config import merge_config
from modnas import utils
from modnas import backend as be
from .exp_manager import ExpManager
from .config import Config
from .logging import configure_logging, get_logger


logger = get_logger()


_default_arg_specs = [
    {
        'flags': ['-c', '--config'],
        'type': str,
        'action': 'append',
        'help': 'yaml config file'
    },
    {
        'flags': ['-n', '--name'],
        'type': str,
        'default': None,
        'help': 'name of the job to run'
    },
    {
        'flags': ['-r', '--routine'],
        'type': str,
        'default': None,
        'help': 'routine type'
    },
    {
        'flags': ['-b', '--backend'],
        'type': str,
        'default': None,
        'help': 'backend type'
    },
    {
        'flags': ['-p', '--chkpt'],
        'type': str,
        'default': None,
        'help': 'checkpoint file'
    },
    {
        'flags': ['-d', '--device_ids'],
        'type': str,
        'default': None,
        'help': 'override device ids'
    },
    {
        'flags': ['-g', '--arch_desc'],
        'type': str,
        'default': None,
        'help': 'override arch_desc file'
    },
    {
        'flags': ['-o', '--override'],
        'type': str,
        'default': None,
        'help': 'override config',
        'action': 'append'
    },
]


DEFAULT_CALLBACK_CONF = {
    'eta': 'ETAReporter',
    'estim': 'EstimReporter',
    'estim_export': 'EstimResultsExporter',
    'trainer': 'TrainerReporter',
    'opt': 'OptimumReporter',
}


def parse_routine_args(name='default', arg_specs=None):
    """Return default arguments."""
    parser = argparse.ArgumentParser(prog='modnas', description='ModularNAS {} routine'.format(name))
    arg_specs = arg_specs or _default_arg_specs.copy()
    for spec in arg_specs:
        parser.add_argument(*spec.pop('flags'), **spec)
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}
    return args


def load_config(conf):
    """Load configurations."""
    if not isinstance(conf, list):
        conf = [conf]
    config = None
    for cfg in conf:
        loaded_cfg = Config.load(cfg)
        config = loaded_cfg if config is None else merge_config(config, loaded_cfg)
    return config


def get_data_provider_config(config):
    """Return data provider config."""
    keys = ['data', 'train_data', 'valid_data', 'data_loader', 'data_provider']
    return {k: config[k] for k in keys if k in config}


def get_init_constructor(config, device):
    """Return default init constructor."""
    if be.is_backend('torch'):
        default_conf = {'type': 'TorchInitConstructor', 'args': {'device': device}}
    elif be.is_backend(None):
        default_conf = {'type': 'DefaultInitConstructor'}
    else:
        raise NotImplementedError
    return merge_config(default_conf, config)


def get_model_constructor(config):
    """Return default model constructor."""
    default_type = 'DefaultModelConstructor'
    default_args = {}
    default_args['model_type'] = config['type']
    if 'args' in config:
        default_args['args'] = config['args']
    return {'type': default_type, 'args': default_args}


def get_chkpt_constructor(path):
    """Return default checkpoint loader."""
    if be.is_backend('torch'):
        return {'type': 'TorchCheckpointLoader', 'args': {'path': path}}


def get_mixed_op_constructor(config):
    """Return default mixed operation constructor."""
    default_type = 'DefaultMixedOpConstructor'
    default_args = {}
    if 'candidates' in config:
        default_args['candidates'] = config.pop('candidates')
    default_args['mixed_op'] = config
    return {'type': default_type, 'args': default_args}


def get_arch_desc_constructor(arch_desc):
    """Return default archdesc constructor."""
    default_con = {'type': 'DefaultSlotArchDescConstructor', 'args': {}}
    default_con['args']['arch_desc'] = arch_desc
    return default_con


def build_constructor_all(config):
    """Build and return all constructors."""
    return OrderedDict([(k, build_con(conf)) for k, conf in config.items()])


def build_exporter_all(config):
    """Build and return all exporters."""
    if config is None:
        return None
    if len(config) == 0:
        if be.is_backend(None):
            config = {'default': {'type': 'DefaultParamsExporter'}}
        else:
            config = {'default': {'type': 'DefaultSlotTraversalExporter'}}
    if len(config) > 1:
        return build_exp('MergeExporter', config)
    if len(config) == 1:
        conf = list(config.values())[0]
        return build_exp(conf)


def build_trainer_all(trainer_config, trainer_comp=None):
    """Build and return all trainers."""
    trners = {}
    for trner_name, trner_conf in trainer_config.items():
        trner = build_trainer(trner_conf, **(trainer_comp or {}))
        trners[trner_name] = trner
    return trners


def build_estim_all(estim_config, estim_comp=None):
    """Build and return all estimators."""
    estims = {}
    if isinstance(estim_config, list):
        estim_config = OrderedDict([(c.get('name', str(i)), c) for i, c in enumerate(estim_config)])
    for estim_name, estim_conf in estim_config.items():
        estim = build_estim(estim_conf, name=estim_name, config=estim_conf, **(estim_comp or {}))
        estim.load(estim_conf.get('chkpt', None))
        estims[estim_name] = estim
    return estims


def bind_trainer(estims, trners):
    """Bind estimators with trainers."""
    for estim in estims.values():
        estim.set_trainer(trners.get(estim.config.get('trainer', estim.name), trners.get('default')))


def reset_all():
    """Reset all framework states."""
    ParamSpace().reset()
    EventManager().reset()


def estims_routine(optim, estims):
    """Run a chain of estimator routines."""
    results, ret = {}, None
    for estim_name, estim in estims.items():
        logger.info('Running estim: {} type: {}'.format(estim_name, estim.__class__.__name__))
        ret = estim.run(optim)
        results[estim_name] = ret
    logger.info('All results: {{\n{}\n}}'.format('\n'.join(['{}: {}'.format(k, v) for k, v in results.items()])))
    reset_all()
    return results


def default_constructor(model, construct_config=None, arch_desc=None):
    """Apply all constructors on model."""
    if arch_desc:
        reg_id, args = parse_spec(construct_config['arch_desc'])
        args['arch_desc'] = arch_desc
        construct_config['arch_desc'] = to_spec(reg_id, args)
    construct_fn = build_constructor_all(construct_config or {})
    for name, con_fn in construct_fn.items():
        logger.info('Running constructor: {} type: {}'.format(name, con_fn.__class__.__name__))
        model = con_fn(model)
    return model


def get_default_constructors(config):
    """Return default constructors from config."""
    con_config = OrderedDict()
    device_conf = config.get('device', {})
    device_ids = config.get('device_ids', None)
    arch_desc = config.get('arch_desc', None)
    if device_ids is not None:
        device_conf['device'] = device_ids
    else:
        device_ids = device_conf.get('device', device_ids)
    con_config['init'] = get_init_constructor(config.get('init', {}), device_ids)
    con_user_config = config.get('construct', {})
    if 'ops' in config:
        con_config['init']['args']['ops_conf'] = config['ops']
    if 'model' in config:
        con_config['model'] = get_model_constructor(config['model'])
    if 'mixed_op' in config:
        con_config['mixed_op'] = get_mixed_op_constructor(config['mixed_op'])
    if arch_desc is not None and 'arch_desc' not in con_user_config:
        con_config['arch_desc'] = get_arch_desc_constructor(arch_desc)
    con_config = merge_config(con_config, con_user_config)
    if be.is_backend('torch'):
        con_config['device'] = {'type': 'TorchToDevice', 'args': device_conf}
    if config.get('chkpt'):
        con_config['chkpt'] = get_chkpt_constructor(config['chkpt'])
    constructor = partial(default_constructor, construct_config=con_config, arch_desc=arch_desc)
    return constructor


def default_apply_config(config):
    """Apply routine config by default."""
    Config.apply(config, config.pop(config['routine'], {}))


def apply_hptune_config(config):
    """Apply hptune routine config."""
    Config.apply(config, config.pop('hptune', {}))
    # hpspace
    config['export'] = None
    if not config.get('construct', {}):
        config['construct'] = {
            'hparams': {
                'type': 'DefaultHParamSpaceConstructor',
                'args': {
                    'params': config.get('hp_space', {})
                }
            }
        }
    hptune_config = list(config.estim.values())[0]
    hptune_args = hptune_config.get('args', {})
    hptune_args['measure_fn'] = config.pop('measure_fn')
    hptune_config['args'] = hptune_args


def apply_pipeline_config(config):
    """Apply pipeline routine config."""
    override = {'estim': {'pipeline': {'type': 'PipelineEstim', 'pipeline': config.pop('pipeline', {})}}}
    Config.apply(config, override)


_default_apply_config_fn = {
    'pipeline': apply_pipeline_config,
    'hptune': apply_hptune_config,
}


def init_all(**kwargs):
    """Initialize all components from config."""
    config = load_config(kwargs.pop('config', {}))
    Config.apply(config, kwargs)
    Config.apply(config, config.get('override') or {})
    routine = config.get('routine')
    if routine:
        apply_config_fn = config.pop('apply_config_fn', None)
        if apply_config_fn is None:
            apply_config_fn = _default_apply_config_fn.get(routine, default_apply_config)
        apply_config_fn(config)
    utils.check_config(config, config.get('defaults'))
    # dir
    name = config.get('name') or utils.get_exp_name(config)
    expman = ExpManager(name, **config.get('expman', {}))
    configure_logging(config=config.get('logging', None), log_dir=expman.subdir('logs'))
    writer = utils.get_writer(expman.subdir('writer'), **config.get('writer', {}))
    logger.info('Name: {} Routine: {} Config:\n{}'.format(name, routine, config))
    # imports
    imports = config.get('import', [])
    if not isinstance(imports, list):
        imports = [imports]
    if config.get('predefined') is not False:
        imports.insert(0, 'modnas.utils.predefined')
    utils.import_modules(imports)
    be.use(config.get('backend'))
    logger.info(utils.env_info())
    # data
    data_provider_conf = get_data_provider_config(config)
    # construct
    construct_fn = config.pop('construct_fn', None)
    model = config.pop('base_model', None)
    if construct_fn is not False:
        constructor = construct_fn or get_default_constructors(config)
        model = constructor(model)
    else:
        constructor = None
    # export
    exporter = build_exporter_all(config.get('export', {}))
    # callback
    cb_config = DEFAULT_CALLBACK_CONF.copy()
    cb_user_conf = config.get('callback', {})
    if isinstance(cb_user_conf, list):
        cb_user_conf = {i: v for i, v in enumerate(cb_user_conf)}
    if cb_user_conf.pop('default', None) is False:
        cb_user_conf.update({k: None for k in cb_config})
    cb_config.update(cb_user_conf)
    for cb in cb_config.values():
        if cb is not None:
            build_callback(cb)
    # optim
    optim = None
    if 'optim' in config:
        optim = build_optim(config.optim)
    # trainer
    trner_comp = {
        'data_provider': data_provider_conf,
        'writer': writer,
    }
    trners = build_trainer_all(config.get('trainer', {}), trner_comp)
    # estim
    estim_comp = {
        'expman': expman,
        'constructor': constructor,
        'exporter': exporter,
        'model': model,
        'writer': writer,
    }
    estims = build_estim_all(config.get('estim', {}), estim_comp)
    bind_trainer(estims, trners)
    return {'optim': optim, 'estims': estims}


def run_default(*args, **kwargs):
    """Run default routines."""
    return estims_routine(**init_all(*args, **kwargs))


def run_search(*args, **kwargs):
    """Run search routines."""
    return estims_routine(**init_all(*args, routine='search', **kwargs))


def run_augment(*args, **kwargs):
    """Run augment routines."""
    return estims_routine(**init_all(*args, routine='augment', **kwargs))


def run_hptune(*args, **kwargs):
    """Run hptune routines."""
    return estims_routine(**init_all(*args, routine='hptune', **kwargs))


def run_pipeline(*args, **kwargs):
    """Run pipeline routines."""
    return estims_routine(**init_all(*args, routine='pipeline', **kwargs))


def run(*args, parse=False, **kwargs):
    """Run routine."""
    if parse or (not args and not kwargs):
        parsed_kwargs = parse_routine_args()
        parsed_kwargs = merge_config(parsed_kwargs, kwargs)
    else:
        parsed_kwargs = kwargs
    return run_default(*args, **parsed_kwargs)
