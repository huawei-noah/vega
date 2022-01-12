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

"""Fake data estimator."""
import numpy as np
from typing import Dict, List, Union
from modnas.core.param_space import ParamSpace
from modnas.core.params import Categorical
from modnas.estim.predefined.regression import RegressionEstim
from modnas.registry.construct import register as register_constructor
from modnas.registry.estim import register as register_estim
from modnas.optim.base import OptimBase


@register_constructor
class FakeDataSpaceConstructor():
    """Fake data space constructor class."""

    def __init__(self, n_params: int = 2**5, dim: int = 2**1) -> None:
        self.n_params = n_params
        self.dim = dim

    def __call__(self, model: None) -> None:
        """Construct search space."""
        del model
        _ = [Categorical(list(range(self.dim))) for _ in range(self.n_params)]


class FakeDataPredictor():
    """Fake data regression predictor class."""

    def __init__(
        self, score_dim: int = 1, seed: int = 11235, random_score: bool = False, noise_scale: float = 0.01
    ) -> None:
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.score_dim = score_dim
        self.random_score = random_score
        self.noise_scale = noise_scale
        self.scores = {'dim_{}'.format(i): {} for i in range(score_dim)}

    def get_score(self, params: Dict[str, int], scores: Dict[str, Union[List[float]]]) -> float:
        """Return score of given parameters."""
        score = 0
        for pn, v in params.items():
            p = ParamSpace().get_param(pn)
            idx = p.get_index(v)
            dim = len(p)
            if pn not in scores:
                if self.random_score:
                    p_score = self.rng.rand(dim)
                    p_score = (p_score / np.max(p_score)).tolist()
                else:
                    p_score = list(range(dim))
                scores[pn] = p_score
            score += scores[pn][idx]
        score /= len(params)
        score += 0 if self.noise_scale is None else self.rng.normal(loc=0, scale=self.noise_scale)
        return score

    def predict(self, params: Dict[str, int]) -> Union[float, Dict[str, float]]:
        """Return predicted evaluation results."""
        scores = {k: self.get_score(params, v) for k, v in self.scores.items()}
        if len(scores) == 1:
            return list(scores.values())[0]
        return scores


@register_estim
class FakeDataEstim(RegressionEstim):
    """Fake data regression estimator class."""

    def __init__(self, *args, pred_conf=None, **kwargs) -> None:
        super().__init__(*args, predictor=FakeDataPredictor(**(pred_conf or {})), **kwargs)

    def run(self, optim: OptimBase) -> None:
        """Run Estimator routine."""
        ret = super().run(optim)
        scores = self.predictor.scores
        if scores:
            for k, score in scores.items():
                score = np.array(list(score.values()))
                if len(score) <= 0:
                    continue
                opt_score = np.sum(np.max(score, 1)) / score.shape[0]
                self.logger.info('global opt. dim: {} param: {}, score: {}'.format(k, score.argmax(1), opt_score))
        return ret
