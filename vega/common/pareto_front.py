# -*- coding: utf-8 -*-

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

"""Pareto front."""

import numpy as np


def get_pareto(scores, index=False, max_nums=-1, choice_column=0, choice='normal', seed=None):
    """Get pareto front."""
    # TODO Get a specified number of samples
    data = scores
    if index:
        data = scores[:, 1:]
    pareto_indexes = get_pareto_index(data)
    res = scores[pareto_indexes]
    if max_nums == -1 or len(res) <= max_nums:
        return res
    if choice == 'normal':
        return normal_selection(res, max_nums, choice_column, seed)


def get_pareto_index(scores):
    """Get pareto front."""
    _size = scores.shape[0]
    pareto_indexes = np.ones(_size, dtype=bool)
    for i in range(_size):
        for j in range(_size):
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                pareto_indexes[i] = False
                break
    return pareto_indexes


def normal_selection(outs, max_nums, choice_column=0, seed=None):
    """Select one record."""
    if seed:
        np.random.seed(seed)
    data = outs[:, choice_column].tolist()
    prob = [round(np.log(i + 1e-2), 2) for i in range(1, len(data) + 1)]
    prob_temp = prob
    for _, out in enumerate(data):
        sorted_ind = np.argsort(out)
        for idx, ind in enumerate(sorted_ind):
            prob[ind] += prob_temp[idx]
    normalization = [float(i) / float(sum(prob)) for i in prob]
    idx = [np.random.choice(len(data), max_nums, replace=False, p=normalization)]
    return outs[idx]
