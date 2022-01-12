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

"""Pipeline Estimator."""
import time
import traceback
import threading
import multiprocessing as mp
import yaml
from modnas.registry.estim import register
from modnas.utils.wrapper import run
from ..base import EstimBase


def _mp_step_runner(conn, step_conf):
    ret = run(**(yaml.load(step_conf, Loader=yaml.SafeLoader) or {}))
    conn.send(ret)


def _mp_runner(step_conf):
    ctx = mp.get_context(step_conf.get('mp_context', 'spawn'))
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
        self.pending = []
        self.finished = set()
        self.failed = set()
        self.cond_all_finished = threading.Lock()
        self.lock_schedule = threading.Lock()

    def exec_runner(self, pname):
        """Execute runner in a thread."""
        try:
            ret = self.runner(self.config.pipeline[pname])
        except RuntimeError as e:
            self.logger.debug(traceback.format_exc())
            self.logger.info(f'pipeline step failed with error, message: {e}')
            self.failed.add(pname)
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
                    self.logger.error('input key {} not found in return {}'.format(inp_idx, self.step_results))
                    self.failed.add(pname)
                    self.step_done(pname, None, None)
                    return
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
        if len(self.finished) == len(self.config.pipeline):
            self.cond_all_finished.release()
        else:
            self._schedule()
        return {'no_opt': True}

    def _schedule(self):
        """Scheduler available jobs."""
        self.lock_schedule.acquire()
        new_pending = []
        for pname in self.pending:
            pconf = self.config.pipeline.get(pname)
            dep_sat = True
            failed = False
            deps = pconf.get('depends', []) + list(set([v.split('.')[0] for v in pconf.get('inputs', {}).values()]))
            for dep in deps:
                if dep in self.failed:
                    failed = True
                    self.failed.add(pname)
                    self.finished.add(pname)
                    break
                if dep not in self.finished:
                    dep_sat = False
                    break
            if failed:
                continue
            if not dep_sat:
                new_pending.append(pname)
                continue
            self.stepped(pname)
        self.pending = new_pending
        self.lock_schedule.release()
        if len(self.finished) == len(self.config.pipeline):
            self.cond_all_finished.release()

    def run(self, optim):
        """Run Estimator routine."""
        del optim
        logger = self.logger
        config = self.config
        pipeconf = config.pipeline
        for pn in pipeconf.keys():
            self.pending.append(pn)
        self.cond_all_finished.acquire()
        self._schedule()
        self.cond_all_finished.acquire()
        self.cond_all_finished.release()
        if self.failed:
            if len(self.failed) == len(self.finished):
                raise RuntimeError('pipeline: all failed')
        logger.info('pipeline: all finished: {}'.format(self.step_results))
        if self.return_res:
            return {'step_results': self.step_results}
