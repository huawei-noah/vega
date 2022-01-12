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

"""This module contains simple helper functions."""
from __future__ import absolute_import
from __future__ import division

import errno
import os
import os.path as osp
import sys
import torch
import vega


def mkdir_if_missing(dirname):
    """Create dirname if it is missing.

    :param dirname: name of path
    :type dirname: str
    """
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """Writes console output to external text file.

    :param fpath: directory to save logging file.
    :type fpath: str
    """

    def __init__(self, fpath=None):
        """Initialize method."""
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        """Close file."""
        self.close()

    def __enter__(self):
        """Pass."""
        pass

    def __exit__(self, *args):
        """Exit file."""
        self.close()

    def write(self, msg):
        """Write messages to log.

        :param msg: messages to be printed
        :type msg: str
        """
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
            self.file.flush()

    def flush(self):
        """Flush the console."""
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        """Close the file."""
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        """Initialize method."""
        self.reset()

    def reset(self):
        """Reset current value."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update current value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def TensorNorm(tensor):
    """Convert tensor's range from [0,1] to [0,255].

    :param tensor: tensor to be converted
    :type tensor: troch.FloatTensor
    """
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = tensor.cpu()
    if (isinstance(tensor, torch.FloatTensor)):
        if (tensor.dim() == 3):
            NormTensor = 255 * tensor
            return NormTensor.to(torch.uint8).cpu()
        elif (tensor.dim() == 4 and tensor.shape[0] == 1):
            NormTensor = 255 * tensor[0]
            return NormTensor.to(torch.uint8).cpu()
        else:
            raise TypeError(
                'Please check the dim of the input tensor, only 3 dim and 4 dim in 1 x C x H x W is accepted')
    else:
        raise TypeError("Only accept torch.FloatTensor")
    return -1


def find_best_PSNR(HR, SR, crop_size):
    """Calculate PSNR and find best PSNR between HR and SR with pixel offset.

    :param HR: HR image
    :type HR: torch.FloatTensor/torch.cuda.FloatTensor
    :param SR: SR image
    :type SR: torch.FloatTensor/torch.cuda.FloatTensor
    :param crop_size: pixel offset when calculating psnr during evaluation, default: 10
    :type crop_size: int
    :return: maximum psnr
    :rtype: Float
    """
    if (crop_size == 0):
        return 20 * torch.log10(1 / torch.sqrt(torch.mean((HR - SR) ** 2))).cpu().item()

    SR = SR.squeeze()
    HR = HR.squeeze()
    SR_crop = SR[:, crop_size:-crop_size, crop_size:-crop_size]
    PSNR_list = torch.zeros((2 * crop_size + 1, 2 * crop_size + 1)).to()
    for i in range(2 * crop_size + 1):
        for j in range(2 * crop_size + 1):
            HR_crop = HR[:, i:i + SR_crop.shape[1], j:j + SR_crop.shape[2]]
            if vega.is_npu_device():
                psnr = 20 * torch.log10(1 / torch.sqrt(torch.mean((HR_crop.cpu() - SR_crop.cpu()) ** 2)))
            else:
                psnr = 20 * torch.log10(1 / torch.sqrt(torch.mean((HR_crop - SR_crop) ** 2)))
            PSNR_list[i, j] = psnr.detach().cpu().item()
            del HR_crop
            del psnr
    max_psnr = PSNR_list.max()
    del PSNR_list
    return max_psnr.cpu().item()
