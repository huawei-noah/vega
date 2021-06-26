# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Pareto front."""

import numpy as np


def get_pareto(scores, index=False):
    """Get pareto front."""
    # TODO Get a specified number of samples
    data = scores
    if index:
        data = scores[:, 1:]
    pareto_indexes = get_pareto_index(data)
    return scores[pareto_indexes]


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
