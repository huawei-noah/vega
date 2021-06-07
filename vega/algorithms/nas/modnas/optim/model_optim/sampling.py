# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Random sampling model optimum finder."""
from .base import ModelOptim
from modnas.registry.model_optim import register


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
