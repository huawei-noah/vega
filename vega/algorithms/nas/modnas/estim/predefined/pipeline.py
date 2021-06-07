# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Pipeline Estimator."""
import time
import yaml
import traceback
import queue
import threading
import multiprocessing as mp
from ..base import EstimBase
from modnas.registry.estim import register
from modnas.utils.wrapper import run


def _mp_step_runner(conn, step_conf):
    ret = run(**(yaml.load(step_conf, Loader=yaml.SafeLoader) or {}))
    conn.send(ret)


def _mp_runner(step_conf):
    ctx = mp.get_context('spawn')
    p_con, c_con = ctx.Pipe()
    proc = ctx.Process(target=_mp_step_runner, args=(c_con, yaml.dump(step_conf)))
    time.sleep(step_conf.get('delay', 0))
    proc.start()
    proc.join()
    if not p_con.poll(0):
        raise RuntimeError('step process failed')
    return p_con.recv()


def _default_runner(step_conf):
    time.sleep(step_conf.get('delay', 0))
    return run(**(yaml.load(step_conf, Loader=yaml.SafeLoader) or {}))


@register
class PipelineEstim(EstimBase):
    """Pipeline Estimator class."""

    def __init__(self, *args, use_multiprocessing=True, return_res=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_res = return_res
        self.runner = _mp_runner if use_multiprocessing else _default_runner
        self.step_results = dict()
        self.pending = queue.Queue()
        self.finished = set()
        self.cond_all_finished = threading.Lock()

    def exec_runner(self, pname):
        """Execute runner in a thread."""
        try:
            ret = self.runner(self.config.pipeline[pname])
        except RuntimeError:
            self.logger.info('pipeline step failed with error: {}'.format(traceback.format_exc()))
            ret = None
        self.step_done(pname, ret, None)

    def step(self, pname):
        """Run a single pipeline step."""
        pconf = self.config.pipeline[pname]
        pconf['name'] = pconf.get('name', pname)
        for inp_kw, inp_idx in pconf.get('inputs', {}).items():
            keys = inp_idx.split('.')
            inp_val = self.step_results
            for k in keys:
                if not inp_val or k not in inp_val:
                    raise RuntimeError('input key {} not found in return {}'.format(inp_idx, self.step_results))
                inp_val = inp_val[k]
            pconf[inp_kw] = inp_val
        self.logger.info('pipeline: running {}'.format(pname))
        self.th_step = threading.Thread(target=self.exec_runner, args=(pname, ))
        self.th_step.start()

    def step_done(self, params, value, arch_desc=None):
        """Store evaluation results of a pipeline step."""
        super().step_done(False, value, arch_desc)
        pname = params
        self.logger.info('pipeline: finished {}, results={}'.format(pname, value))
        self.step_results[pname] = value
        self.finished.add(pname)
        self._schedule()
        return {'no_opt': True}

    def _schedule(self):
        """Scheduler available jobs."""
        if len(self.finished) == len(self.config.pipeline):
            self.cond_all_finished.release()
            return
        while not self.pending.empty():
            pname = self.pending.get()
            pconf = self.config.pipeline.get(pname)
            dep_sat = True
            deps = pconf.get('depends', []) + list(set([v.split('.')[0] for v in pconf.get('inputs', {}).values()]))
            for dep in deps:
                if dep not in self.finished:
                    dep_sat = False
                    break
            if not dep_sat:
                self.pending.put(pname)
                continue
            self.stepped(pname)

    def run(self, optim):
        """Run Estimator routine."""
        del optim
        logger = self.logger
        config = self.config
        pipeconf = config.pipeline
        for pn in pipeconf.keys():
            self.pending.put(pn)
        self.cond_all_finished.acquire()
        self._schedule()
        self.cond_all_finished.acquire()
        self.cond_all_finished.release()
        logger.info('pipeline: all finished: {}'.format(self.step_results))
        if self.return_res:
            return {'step_results': self.step_results}
