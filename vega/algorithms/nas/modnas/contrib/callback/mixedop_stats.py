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

"""Mixed operator statistics reporter."""
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from typing import Dict, Optional, Any
from modnas.registry.callback import register
from modnas.arch_space.mixed_ops import MixedOp
from modnas.callback.base import CallbackBase
from modnas.estim.base import EstimBase
from modnas.optim.base import OptimBase
from vega.common import FileOps

plt.switch_backend('Agg')


@register
class MixedOpStatsReporter(CallbackBase):
    """Mixed operator statistics reporter class."""

    def __init__(self) -> None:
        super().__init__({
            'before:EstimBase.run_epoch': self.record_probs,
            'after:EstimBase.run': self.save_stats,
        })
        self.probs = []

    def record_probs(
        self, estim: EstimBase, optim: Optional[OptimBase], epoch: Optional[int], tot_epochs: Optional[int]
    ) -> None:
        """Record mixed operator probabilities on each epoch."""
        self.probs.append([F.softmax(m.alpha().detach(), dim=-1).cpu().numpy() for m in MixedOp.gen(estim.model)])

    def save_stats(self, ret: Dict[str, Any], estim: EstimBase, optim: OptimBase) -> Dict[str, Any]:
        """Save statistics on search end."""
        self.record_probs(estim, None, None, None)
        probs = self.probs
        n_epochs, n_alphas = len(probs), len(probs[0])
        self.logger.info('mixed op stats: epochs: {} alphas: {}'.format(n_epochs, n_alphas))
        epochs = list(range(n_epochs))
        save_probs = []
        for i, m in enumerate(MixedOp.gen(estim.model)):
            plt.figure(i)
            plt.title('alpha: {}'.format(i))
            prob = np.array([p[i] for p in probs])
            alpha_dim = prob.shape[1]
            for a in range(alpha_dim):
                plt.plot(epochs, prob[:, a])
            legends = m.candidate_names()
            plt.legend(legends)
            plt.savefig(estim.expman.join('plot', 'prob_{}.png'.format(i)))
            save_probs.append(prob)
        probs_path = estim.expman.join('output', 'probs.pkl')
        FileOps.dump_pickle(save_probs, probs_path)
        self.logger.info('mixed op probs saved to {}'.format(probs_path))
        self.probs = []
        return ret
