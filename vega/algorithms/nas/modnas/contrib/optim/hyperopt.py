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

"""Optim wrapper for Hyperopt."""
from collections import OrderedDict
import numpy as np
from modnas.registry.optim import register
from modnas.optim.base import OptimBase
from modnas.core.params import Categorical, Numeric
try:
    import hyperopt
    from hyperopt import tpe, hp
    import hyperopt.utils
    import hyperopt.base
except ImportError:
    hyperopt = None


@register
class HyperoptOptim(OptimBase):
    """Hyperopt Optimizer class."""

    MAX_RANDINT = 2 ** 31 - 1

    def __init__(self, hyperopt_args=None, space=None):
        super().__init__(space)
        if hyperopt is None:
            raise ValueError('Hyperopt is not installed')
        hyperopt_dims = {}
        for n, p in self.space.named_params():
            if isinstance(p, Numeric):
                args = {
                    'low': p.bound[0],
                    'high': p.bound[1],
                }
                if p.is_int():
                    sd = hp.uniformint(n, **args)
                else:
                    sd = hp.uniform(n, **args)
            elif isinstance(p, Categorical):
                sd = hp.choice(n, p.choices)
            else:
                continue
            hyperopt_dims[n] = sd
        hyperopt_args = hyperopt_args or {}
        self.algo = tpe.suggest
        self.trials = hyperopt.base.Trials()
        self.domain = hyperopt.base.Domain(self.hp_eval, hyperopt_dims)
        self.rstate = np.random.RandomState()
        self.next_pts = []

    def hp_eval(self, args):
        """Hyperopt objective wrapper."""
        self.next_pts.append(args)
        return 0

    def has_next(self):
        """Return True if Optimizer has the next set of parameters."""
        return True

    def convert_param(self, p):
        """Return value converted from Hyperopt space."""
        if isinstance(p, (np.float, np.float64)):
            return float(p)
        if isinstance(p, (np.int, np.int64)):
            return int(p)
        return p

    def next(self, batch_size):
        """Return the next batch of parameter sets."""
        n_to_enqueue = batch_size
        new_ids = self.trials.new_trial_ids(n_to_enqueue)
        self.trials.refresh()
        seed = self.rstate.randint(self.MAX_RANDINT)
        new_trials = self.algo(new_ids, self.domain, self.trials, seed)
        self.trials.insert_trial_docs(new_trials)
        self.trials.refresh()
        for trial in self.trials._dynamic_trials:
            if trial['state'] == hyperopt.base.JOB_STATE_NEW:
                trial['state'] = hyperopt.base.JOB_STATE_RUNNING
                now = hyperopt.utils.coarse_utcnow()
                trial['book_time'] = now
                trial['refresh_time'] = now
                spec = hyperopt.base.spec_from_misc(trial['misc'])
                ctrl = hyperopt.base.Ctrl(self.trials, current_trial=trial)
                self.domain.evaluate(spec, ctrl)
        self.trials.refresh()

        next_params = []
        for pt in self.next_pts:
            params = OrderedDict()
            for n, p in pt.items():
                params[n] = self.convert_param(p)
            next_params.append(params)
        self.next_pts.clear()
        return next_params

    def step(self, estim):
        """Update Optimizer states using Estimator evaluation results."""
        def to_metrics(res):
            if res is None:
                return None
            if isinstance(res, dict):
                v = list(res.values())[0]
            if isinstance(res, (tuple, list)):
                v = res[0]
            else:
                v = res
            return v

        _, results = estim.get_last_results()
        skresults = [to_metrics(r) for r in results]
        trials = filter(lambda x: x['state'] == hyperopt.base.JOB_STATE_RUNNING, self.trials._dynamic_trials)
        for trial, result in zip(trials, skresults):
            now = hyperopt.utils.coarse_utcnow()
            if result is None:
                trial['state'] = hyperopt.base.JOB_STATE_ERROR
                trial['refresh_time'] = now
            else:
                trial['state'] = hyperopt.base.JOB_STATE_DONE
                trial['result'] = {
                    'loss': -result,
                    'status': hyperopt.base.STATUS_OK
                }
                trial['refresh_time'] = now
        self.trials.refresh()
