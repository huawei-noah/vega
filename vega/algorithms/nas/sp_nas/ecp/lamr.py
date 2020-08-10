# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Calculate log average miss-rate(LAMR)."""
import matplotlib.pyplot as plt
import numpy as np


def lamr(recall, fppi):
    """Calculate log average miss-rate(LAMR).

    :param recall: recall
    :type recall: numpy
    :param fppi: False Positive per Image
    :type fppi: numpy
    :return: LAMR
    :rtype: numpy
    """
    ref = np.power(10, np.linspace(-2, 0, 9))
    result = np.zeros(ref.shape)
    for i in range(len(ref)):
        j = np.argwhere(fppi <= ref[i]).flatten()
        if j.size:
            result[i] = recall[j[-1]]
        else:
            result[i] = 0
    return np.exp(np.mean(np.log(np.maximum(1e-10, 1 - result))))


def create_lamr(title, label, mr, fppi):
    """Create figure of LAMR."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog()
    ax.set_xlabel('FPPI')
    ax.set_ylabel('Miss Rate')
    ax.grid(b=True, which='major', linestyle='-')
    ax.yaxis.grid(b=True, which='minor', linestyle='--')
    ax.set_xlim([5 * 1e-5, 10])
    ax.set_ylim([8 * 1e-3, 1])
    ax.plot(fppi, mr)
    if title:
        ax.set_title(title)
    if label:
        ax.legend([label])
    return fig
