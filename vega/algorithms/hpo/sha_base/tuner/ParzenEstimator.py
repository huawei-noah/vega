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

"""Class of ParzenEstimator."""
import numpy as np
from scipy.special import erf


EPS = 1e-12


class ParzenEstimator:
    """ParzenEstimator.

    :param hyper_param_list: List[HyperParameter]
    :type hyper_param_list: list
    :param gamma: gamma.
    :type gamma: int
    :param left_forget: left_forget
    :type left_forget: int
    """

    def __init__(self, hyper_param_list, gamma=0.25, left_forget=25):
        self.kdes = {"upper": [], "lower": []}
        self.range_list = [
            hyper_param.range for hyper_param in hyper_param_list
        ]
        self.gamma = gamma
        self.left_forget = left_forget

    def fit(self, X, y):
        """Fit a model."""
        if len(X) <= 1:
            return self
        num = X.shape[0]
        dim = X.shape[1]

        if dim != len(self.range_list):
            raise ValueError(
                "Variable `X's` column numbers should be consistent with the "
                "number of hyper-parameters: {}!".format(len(self.range_list))
            )

        order = np.argsort(-y, kind="stable")

        segmentation = int(np.ceil(self.gamma * np.sqrt(num)))

        points_upper, points_lower = [], []
        idxs_upper = set(order[:segmentation])
        for idx in range(len(order)):
            if idx in idxs_upper:
                points_upper.append(X[idx])
            else:
                points_lower.append(X[idx])

        points_upper, points_lower = np.asarray(points_upper), np.asarray(points_lower)

        for index in range(dim):
            range_ = self.range_list[index]
            ori_mu, ori_sigma = sum(range_) / 2.0, abs(range_[0] - range_[1])

            weights, mus, sigmas = self._generate_dim_info(
                points_upper[:, index], ori_mu, ori_sigma
            )
            self._generate_kde(
                "upper", range_, weights, mus, sigmas
            )
            weights, mus, sigmas = self._generate_dim_info(
                points_lower[:, index], ori_mu, ori_sigma
            )
            self._generate_kde(
                "lower", range_, weights, mus, sigmas
            )
        return self

    def predict(self, X):
        """Predict a mean and std for input X.

        :param X:
        :return:
        """
        output = np.zeros_like(X)
        dim = output.shape[1]
        for index in range(dim):
            lx = self.kdes["upper"][index].score_samples(X[:, index:index + 1])
            gx = self.kdes["lower"][index].score_samples(X[:, index:index + 1])
            output[:, index] = lx - gx
        mean = output.mean(axis=1)
        std = output.std(axis=1)
        return mean, std

    def _generate_dim_info(self, points, ori_mu, ori_sigma):
        """Generate dim info."""
        mus, sigmas, ori_pos, srt_order = self._generate_ordered_points(points, ori_mu, ori_sigma)
        weights = self._generate_weights(len(points), ori_pos, srt_order)

        upper_sigma = ori_sigma
        lower_sigma = ori_sigma / (min(100.0, (1.0 + len(mus))))

        sigmas = np.clip(sigmas, lower_sigma, upper_sigma)
        sigmas[ori_pos] = ori_sigma

        weights /= weights.sum()

        return weights, mus, sigmas

    def _generate_ordered_points(self, points, ori_mu, ori_sigma):
        """Generate ordered point."""
        if len(points) >= 2:
            srt_order = np.argsort(points)
            srt_points = points[srt_order]
            ori_pos = np.searchsorted(srt_points, ori_mu)

            mus = np.hstack((srt_points[:ori_pos], ori_mu, srt_points[ori_pos:]))

            points_diff = np.diff(mus)
            sigmas = np.hstack((
                mus[1] - mus[0],
                np.maximum(points_diff[:-1], points_diff[1:]),
                mus[-1] - mus[-2],
            ))
        else:
            srt_order = None
            mus = np.asarray([ori_mu])
            sigmas = np.asarray([ori_sigma])
            if len(points) == 1:
                if ori_mu > points[0]:
                    ori_pos = 1
                else:
                    ori_pos = 0
                mus = np.insert(mus, 1 - ori_pos, points[0])
                sigmas = np.insert(sigmas, 1 - ori_pos, 0.5 * ori_sigma)
            else:
                ori_pos = 0
        return mus, sigmas, ori_pos, srt_order

    def _generate_weights(self, num, ori_pos=None, srt_order=None):
        if num <= self.left_forget:
            weights = np.ones((num + 1,))
        else:
            lf_weights = np.hstack(
                (np.linspace(1.0 / num, 1.0, num - self.left_forget),
                 np.ones((self.left_forget,)))
            )[srt_order]
            weights = np.hstack(
                (lf_weights[:ori_pos], 1.0, lf_weights[ori_pos:])
            )
        return weights

    def _generate_kde(self, label, range_, weights, mus, sigmas):
        kde = _KernelDensity(range_).fit(weights, mus, sigmas)
        self.kdes[label].append(kde)


class _KernelDensity:
    """KernelDensity."""

    def __init__(self, range_):
        self.lb, self.ub = min(range_), max(range_)

    def fit(self, weights, mus, sigmas):
        """Fit a model."""
        self.weights = weights
        self.mus = mus
        self.sigmas = sigmas
        return self

    def score_samples(self, X):
        """Sample score."""
        logpdf = _weighted_truncated_norm_lpdf(
            X, self.lb, self.ub, self.mus, self.sigmas, self.weights
        )

        max_logpdf = np.max(logpdf, axis=1)
        return \
            np.log(
                np.sum(np.exp(logpdf - max_logpdf[:, None]), axis=1)
            ) + max_logpdf


def _weighted_truncated_norm_lpdf(X, lower_bound, upper_bound, mus, sigmas, weights):
    """Get pdf according to lower_bound and upper_bound."""
    sigmas = np.maximum(sigmas, EPS)

    p_inside = (weights * (
        _gauss_cdf(upper_bound, mus, sigmas) - _gauss_cdf(lower_bound, mus, sigmas)
    )).sum()

    standard_X = (X - mus) / sigmas
    partition = np.sqrt(2 * np.pi) * sigmas

    log_probs = -(standard_X)**2 / 2.0 + np.log(weights / p_inside / partition)
    return log_probs


def _gauss_cdf(x, mu, sigma):
    """Get cdf."""
    standard_x = (x - mu) / sigma
    cdf = (1 + erf(standard_x / np.sqrt(2))) / 2.0
    return cdf
