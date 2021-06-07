# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Mixed operator statistics reporter."""
import numpy as np
import pickle
import torch.nn.functional as F
from modnas.registry.callback import register
from modnas.arch_space.mixed_ops import MixedOp
from modnas.callback.base import CallbackBase
from matplotlib import pyplot as plt
plt.switch_backend('Agg')


@register
class MixedOpStatsReporter(CallbackBase):
    """Mixed operator statistics reporter class."""

    def __init__(self):
        super().__init__({
            'before:EstimBase.run_epoch': self.record_probs,
            'after:EstimBase.run': self.save_stats,
        })
        self.probs = []

    def record_probs(self, estim, optim, epoch, tot_epochs):
        """Record mixed operator probabilities on each epoch."""
        self.probs.append([F.softmax(m.alpha().detach(), dim=-1).cpu().numpy() for m in MixedOp.gen(estim.model)])

    def save_stats(self, ret, estim, optim):
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
        with open(probs_path, 'wb') as f:
            pickle.dump(save_probs, f)
            self.logger.info('mixed op probs saved to {}'.format(probs_path))
        self.probs = []
