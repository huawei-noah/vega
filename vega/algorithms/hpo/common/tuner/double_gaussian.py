# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Class of DoubleMultiGaussian."""
from sklearn.mixture import GaussianMixture
import numpy as np


class DoubleMultiGaussian(object):
    """Gaussian Process.

    :param gamma: gamma.
    :type gamma: int
    """

    def __init__(self, gamma=0.25):
        """Init TunerModel."""
        self.gamma = gamma
        self.means_ = None
        self.covariances_ = None

    def fit(self, X, y):
        """Divide X according to y and get two Gaussian model."""
        X_sorted = X[np.argsort(-y)]
        if X.shape[0] < 4:
            gaussian_high = GaussianMixture().fit(X_sorted)
            gaussian_low = gaussian_high
        else:
            point_segmentation = max(2, int(self.gamma * X.shape[0]))
            gaussian_high = GaussianMixture().fit(X_sorted[:point_segmentation])
            gaussian_low = GaussianMixture().fit(X_sorted[point_segmentation:])

        self.means_ = [gaussian_high.means_[0], gaussian_low.means_[0]]
        self.covariances_ = [gaussian_high.covariances_[0], gaussian_low.covariances_[0]]
