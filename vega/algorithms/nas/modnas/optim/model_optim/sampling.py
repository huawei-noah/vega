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

"""Random sampling model optimum finder."""
from modnas.registry.model_optim import register
from .base import ModelOptim


@register
class RandomSamplingModelOptim(ModelOptim):
    """Random sampling model optimum finder class."""

    def __init__(self, space, n_iter=1000):
        super().__init__(space)
        self.n_iter = n_iter

    def get_optimums(self, model, size, excludes):
        """Return optimums in score model."""
        smpl_pts = [self.get_random_index(excludes) for _ in range(self.n_iter)]
        smpl_val = model.predict([self.space.get_categorical_params(i) for i in smpl_pts])
        topk_idx = smpl_val.argsort()[::-1][:size]
        return [smpl_pts[i] for i in topk_idx]
