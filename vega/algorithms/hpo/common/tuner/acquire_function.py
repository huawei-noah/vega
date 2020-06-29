# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Acquire functions."""
import numpy as np
import random
from scipy.stats import norm, multivariate_normal


def expected_improvement(predictions, best_score):
    """Rewrite the acquire function, here we use the Expected Improvement.

    The application of Bayesian methods for seeking the extremum.
    Towards Global Optimization, (117-129), 1978.

    :param predictions:
    :return
    """
    if predictions.shape[1] == 1:
        return np.argmax(predictions)
    p_cdf = norm.cdf
    n_pdf = norm.pdf

    label, std = predictions.T
    if best_score is None:
        best_score = np.max(label)

    if np.any(std == 0.0):
        std[std == 0.0] = np.inf
    z_std = (label - best_score) / std
    e_i = std * (z_std * p_cdf(z_std) + n_pdf(z_std))
    e_i = e_i.round(3)

    rnd = np.random.RandomState(np.random.randint(10000)).rand(len(e_i))
    ind = np.lexsort((rnd.flatten(), e_i.flatten()))
    return ind[-1]


def thompson_sampling(feature, predictions):
    """Define ac_func use the sequence thompson sampling.

    get the sample count sample_num=10 * d^2 * j
    where d is the dimension of hp(feature),
    j is the current number of evaluations(feature)

    :param feature:
    :param predictions:
    :return
    """
    sample_num = 10

    if feature.size != 0:
        sample_num = 10 * feature.shape[0] * feature.shape[1] * feature.shape[1]

    if predictions.shape[0] < sample_num:
        sample_num = predictions.shape[0]

    y_sample = predictions.T[0]
    indices = range(y_sample.size)
    idx_sample = np.random.choice(indices, sample_num, replace=False)
    sub_y_sample = np.take(y_sample, idx_sample)
    max_idx = np.argmax(sub_y_sample)
    return idx_sample[max_idx]


def minimize_pdf(parameters, mean, cov):
    """Minimize two Probability density function according to TPE."""
    gx = multivariate_normal.pdf(parameters, mean=mean[0], cov=cov[0])
    lx = multivariate_normal.pdf(parameters, mean=mean[1], cov=cov[1])
    gx_min = gx[np.nonzero(gx)]
    lx_max = lx[np.nonzero(lx)]
    if any(gx_min) and any(lx_max):
        np.seterr(divide='ignore', invalid='ignore')
        theanp = np.true_divide(gx, lx)
        where_are_nan = np.isnan(theanp)
        theanp[where_are_nan] = float("inf")
        index = np.argmin(np.where(theanp == 0.0, float("inf"), theanp))
    elif any(gx_min) and not any(lx_max):
        index = np.argmin(np.where(gx == 0.0, float("inf"), gx))
    elif not any(gx_min) and any(lx_max):
        index = np.argmax(np.where(lx == 0.0, float("-inf"), lx))
    else:
        index = random.randint(0, parameters.shape[0])
    return index
