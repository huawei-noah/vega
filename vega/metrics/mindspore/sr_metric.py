# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of super resolution task."""
from mindspore.nn.metrics import Metric
from vega.common import ClassFactory, ClassType
import numpy as np
import math


@ClassFactory.register(ClassType.METRIC)
class PSNR(Metric):
    """Calculate IoU between output and target."""

    __metric_name__ = 'PSNR'

    def __init__(self, to_y=True, scale=2, max_rgb=1):
        self.to_y = to_y
        self.scale = scale
        self.max_rgb = max_rgb
        self.data_num = 0
        self.sum = 0

    def update(self, *inputs):
        """Update the metric."""
        if len(inputs) != 2:
            raise ValueError('PSNR need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        res = self.compute_sr_metric(y_pred, y)
        n = y_pred.shape[0]
        self.data_num += n
        self.sum = self.sum + res * n
        self.pfm = self.sum / self.data_num

    def eval(self):
        """Get the metric."""
        return self.pfm

    def clear(self):
        """Reset the metric."""
        self.data_num = 0
        self.sum = 0
        self.pfm = 0

    def compute_sr_metric(self, img_sr, img_hr):
        """Compute sr metric."""
        if len(img_sr.shape) == 5:
            result = 0.
            for ind in range(img_sr.size(4)):
                result += self.compute_metric(img_sr[:, :, :, :, ind],
                                              img_hr[:, :, :, :, ind])
            return result / img_sr.size(4)
        else:
            return self.compute_metric(img_sr, img_hr)

    def compute_metric(self, img_sr, img_hr):
        """Compute super solution metric according metric type.

        :param img_sr: predicted tensor (4D)
        :type img_sr: numpy.array
        :param img_hr: target tensor (4D)
        :type img_hr: numpy.array
        :return: Average metric of the batch
        :rtype: float
        """
        if self.scale != 0:
            img_sr = img_sr[:, :, self.scale: -1 * self.scale, self.scale: -1 * self.scale]
            img_hr = img_hr[:, :, self.scale: -1 * self.scale, self.scale: -1 * self.scale]
        if self.max_rgb == 255:
            img_sr = img_sr / 255.0
            img_hr = img_hr / 255.0
        if self.to_y:
            multiplier = np.array([25.064, 129.057, 65.738]).reshape([3, 1, 1]) / 256
            img_sr = np.sum(img_sr * multiplier, axis=1)
            img_hr = np.sum(img_hr * multiplier, axis=1)
        diff = (img_sr - img_hr)
        mse = np.mean(np.power(diff, 2))
        sr_metric = -10 * math.log10(mse)
        # sr_metric = nn.PSNR()(Tensor(img_sr), Tensor(img_hr))
        return sr_metric
        # return self._convert_data(sr_metric)[0]

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return 'MAX'

    def __call__(self, output, target, *args, **kwargs):
        """Forward and calculate accuracy."""
        return self


@ClassFactory.register(ClassType.METRIC)
class SSIM(Metric):
    """Calculate IoU between output and target."""

    __metric_name__ = 'SSIM'

    def __init__(self, to_y=True, scale=2, max_rgb=1):
        self.to_y = to_y
        self.scale = scale
        self.max_rgb = max_rgb
        self.data_num = 0
        self.sum = 0

    def update(self, *inputs):
        """Update the metric."""
        if len(inputs) != 2:
            raise ValueError('SSIM need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        res = self.compute_sr_metric(y_pred, y)
        n = y_pred.shape[0]
        self.data_num += n
        self.sum = self.sum + res * n
        self.pfm = self.sum / self.data_num

    def eval(self):
        """Get the metric."""
        return self.pfm

    def clear(self):
        """Reset the metric."""
        self.data_num = 0
        self.sum = 0
        self.pfm = 0

    def compute_sr_metric(self, img_sr, img_hr):
        """Compute sr metric."""
        if len(img_sr.shape) == 5:
            result = 0.
            for ind in range(img_sr.size(4)):
                result += self.compute_metric(img_sr[:, :, :, :, ind],
                                              img_hr[:, :, :, :, ind])
            return result / img_sr.size(4)
        else:
            return self.compute_metric(img_sr, img_hr)

    def compute_metric(self, img_sr, img_hr):
        """Compute super solution metric according metric type.

        :param img_sr: predicted tensor (4D)
        :type img_sr: numpy.array
        :param img_hr: target tensor (4D)
        :type img_hr: numpy.array
        :return: Average metric of the batch
        :rtype: float
        """
        if self.scale != 0:
            img_sr = img_sr[:, :, self.scale: -1 * self.scale, self.scale: -1 * self.scale]
            img_hr = img_hr[:, :, self.scale: -1 * self.scale, self.scale: -1 * self.scale]
        if self.max_rgb == 255:
            img_sr = img_sr / 255.0
            img_hr = img_hr / 255.0
        if self.to_y:
            multiplier = np.array([25.064, 129.057, 65.738]).reshape([3, 1, 1]) / 256
            img_sr = np.sum(img_sr * multiplier, axis=1)
            img_hr = np.sum(img_hr * multiplier, axis=1)
            sr_metric = 0
        return sr_metric

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return 'MAX'

    def __call__(self, output, target, *args, **kwargs):
        """Forward and calculate accuracy."""
        return self
