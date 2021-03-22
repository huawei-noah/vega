# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Pipeline Estimator."""

import traceback
import queue
from multiprocessing import Process, Pipe
from ..base import EstimBase
from modnas.registry.runner import build as build_runner
from modnas.registry import parse_spec
from modnas.registry.estim import register


def _mp_step_runner(conn, step_conf):
    ret = build_runner(step_conf)
    conn.send(ret)


def _mp_runner(step_conf):
    p_con, c_con = Pipe()
    proc = Process(target=_mp_step_runner, args=(c_con, step_conf))
    proc.start()
    proc.join()
    if not p_con.poll(0):
        return None
    return p_con.recv()


def _default_runner(step_conf):
    return build_runner(step_conf)


@register
class PipelineEstim(EstimBase):
    """Pipeline Estimator class."""

    def __init__(self, *args, use_multiprocessing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.runner = _mp_runner if use_multiprocessing else _default_runner

    def step(self, step_conf):
        """Return results from single pipeline process."""
        try:
            return self.runner(step_conf)
        except RuntimeError:
            self.logger.info('pipeline step failed with error: {}'.format(traceback.format_exc()))
        return None

    def run(self, optim):
        """Run Estimator routine."""
        del optim
        logger = self.logger
        config = self.config
        pipeconf = config.pipeline
        pending = queue.Queue()
        for pn in pipeconf.keys():
            pending.put(pn)
        finished = set()
        ret_values, ret = dict(), None
        while not pending.empty():
            pname = pending.get()
            pconf = pipeconf.get(pname)
            dep_sat = True
            for dep in pconf.get('depends', []):
                if dep not in finished:
                    dep_sat = False
                    break
            if not dep_sat:
                pending.put(pname)
                continue
            ptype, pargs = parse_spec(pconf)
            pargs['name'] = pargs.get('name', pname)
            for inp_kw, inp_idx in pconf.get('inputs', {}).items():
                keys = inp_idx.split('.')
                inp_val = ret_values
                for k in keys:
                    if not inp_val or k not in inp_val:
                        raise RuntimeError('input key {} not found in return {}'.format(inp_idx, ret_values))
                    inp_val = inp_val[k]
                pargs[inp_kw] = inp_val
            logger.info('pipeline: running {}, type={}'.format(pname, ptype))
            ret = self.step(pconf)
            ret_values[pname] = ret
            logger.info('pipeline: finished {}, results={}'.format(pname, ret))
            finished.add(pname)
        ret_values['final'] = ret
        logger.info('pipeline: all finished')
        return ret_values
