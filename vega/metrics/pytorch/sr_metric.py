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

"""Metric of super solution task."""
import math
import torch
import numpy as np
import cv2
from vega.metrics.pytorch.metrics import MetricBase
from vega.common import ClassFactory, ClassType


def tensor_to_np_images(tensor):
    """Convert a 4D tensor of BCHW to a 4D image of HWC (in numpy).

    :param tensor: A 4D tensor with BHWC
    :type tensor: torch.tensor
    :return: image in numpy
    :rtype: np.array (with dtype np.uint8), in range 0~255
    """
    img_np = tensor.detach().cpu().numpy()
    img_np = np.transpose(img_np, (0, 2, 3, 1))
    return img_np.round().astype(np.uint8)


def crop_np_border(image, crop_border):
    """Remove the border of image, with the border size of crop_border.

    :param image: an image denoted by a numpy array
    :type image: np.array
    :param crop_border: the size of border to be removed
    :type crop_border: int
    :return: croped image
    :rtype: np.array
    """
    if crop_border == 0:
        return image
    return image[:, crop_border:-crop_border, crop_border:-crop_border]


def bgr_to_y(image):
    """Convert a bgr image to grayscale.

    :param image: an image denoted by a numpy array, which is HWC, and channels are in gbr
    :type image: np.array
    :return: the grayscale image
    :rtype: np.array (with dtype np.float32), in range 0~255
    """
    image = image.astype(np.float32) / 255.0
    # be consistent with common calculation
    coef = [25.064 / 256.0, 129.057 / 256.0, 65.738 / 256.0]
    return np.dot(image, coef) * 255.0


def quantize(img):
    """Quantize the output of model.

    :param img: the input image
    :type img: ndarray
    :return: the image after quantize
    :rtype: ndarray
    """
    pixel_range = 255
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calculate_psnr(sr, hr, normalize=False, dataset=None):
    """Caculate the psnr on y channel.

    :param sr: the  predictied  image
    :type sr: ndarray
    :param hr: the high resolution image
    :type hr: ndarray
    :param dataset: dataset, defaults to None
    :type dataset: Data
    :return: psnr_y
    :rtype: float
    """
    if hr.nelement() == 1:
        return 0
    if normalize:
        sr = quantize(sr)
    diff = (sr - hr)
    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)


def ssim(x, y):
    """Calculate ssim value of x (3D) in respect to y (3D).

    :param x: preprocessed predicted tensor (3D)
    :type x: torch.Tensor
    :param y: preprocessed groundtruth tensor (3D)
    :type y: torch.Tensor
    """
    # pre-computation
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    flatten_x = torch.flatten(x, start_dim=1)
    flatten_y = torch.flatten(y, start_dim=1)
    tot_pixel = x.size(1) * x.size(2)

    # calculate miu
    miux = torch.mean(x, dim=(1, 2))
    miuy = torch.mean(y, dim=(1, 2))
    mean_subtracted_x = flatten_x - miux.unsqueeze(1)
    mean_subtracted_y = flatten_y - miuy.unsqueeze(1)

    # calculate phi
    supportx = torch.sum(mean_subtracted_x ** 2, dim=1)
    phix = torch.sqrt(supportx / (tot_pixel - 1))
    supporty = torch.sum(mean_subtracted_y ** 2, dim=1)
    phiy = torch.sqrt(supporty / (tot_pixel - 1))
    phixy = torch.sum(mean_subtracted_x * mean_subtracted_y, dim=1) / (tot_pixel - 1)

    # calculate ssim
    result = torch.mean(((2 * miux * miuy + C1) * (2 * phixy + C2)) /   # noqa W504
                        ((miux ** 2 + miuy ** 2 + C1) * (phix ** 2 + phiy ** 2 + C2))).item()
    return result


def mean_ssim(img1, img2):
    """Calculate mean ssim value of img1 (2D) in respect to img2 (2D).

    :param img1: predicted image in 2D
    :type img1: np.array
    :param img2: image of ground truth in 2D
    :type img2: np.array
    :return: ssim of img1 in respect to img2
    :rtype: float
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """Calculate ssim value of img1 (2D or 3D) in respect to img2 (2D or 3D).

    :param img1: predicted image
    :type img1: np.array
    :param img2: image of ground truth
    :type img2: np.array
    :return: ssim of img1 in respect to img2
    :rtype: float
    """
    if img1.ndim == 2:
        return mean_ssim(img1, img2)
    else:
        ssims = []
        for i in range(3):
            ssims.append(mean_ssim(img1[:, :, i], img2[:, :, i]))
        return np.array(ssims).mean()


def preprocess(tensor, to_y=True, crop_border=0):
    """Convert tensor of BGR to grayscale, and crop border.

    :param tensor: tensor
    :type tensor: torch.Tensor
    :param to_y: whether convert image from format bgr to format y
    :type to_y: bool
    :param crop_border: number of pixels to crop
    :type crop_border: int
    :return: grayscale tensor (if to_y is true), with border cropped, within range 0~1
    :rtype: torch.Tensor, in range 0~1
    """
    # crop boarders
    if crop_border > 0:
        tensor = tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
    # convert to y
    tensor /= 255.0
    if to_y:
        multiplier = torch.tensor([25.064, 129.057, 65.738]).view([3, 1, 1]).to(tensor.device) / 256.0
        tensor = torch.sum(tensor * multiplier, dim=1)
    return tensor


def compute_metric(img_sr, img_hr, method='psnr', to_y=True, scale=2, max_rgb=1):
    """Compute super solution metric according metric type.

    :param img_sr: predicted tensor (4D)
    :type img_sr: torch.Tensor
    :param img_hr: target tensor (4D)
    :type img_sr: torch.Tensor
    :param method: sr calculate method, psnr or ssim
    :type method: str
    :param to_y: whether convert image from format bgr to format y
    :type to_y: bool
    :param crop_border: number of pixels to crop
    :type crop_border: int
    :return: Average PSNR of the batch
    :rtype: float
    """
    # img_sr and img_hr has to be in 0~255
    if max_rgb == 1:
        img_sr = img_sr * 255.0
        img_hr = img_hr * 255.0
    if method == 'psnr':
        sr, hr = preprocess(img_sr, to_y, scale), preprocess(img_hr, to_y, scale)
        return calculate_psnr(sr, hr)
    elif method == 'ssim':
        img_sr_np = crop_np_border(tensor_to_np_images(img_sr), scale)
        img_hr_np = crop_np_border(tensor_to_np_images(img_hr), scale)
        if to_y:
            img_sr_np = bgr_to_y(img_sr_np)
            img_hr_np = bgr_to_y(img_hr_np)
        result = 0.
        for batch_idx in range(img_hr_np.shape[0]):
            result += calculate_ssim(img_sr_np[batch_idx], img_hr_np[batch_idx])
        return result / img_hr_np.shape[0]
    else:
        raise Exception('Wrong segmetation metric type, should be psnr or ssim')


def compute_sr_metric(img_sr, img_hr, method='psnr', to_y=True, scale=2, max_rgb=1):
    """Compute super solution metric according metric type.

    :param img_sr: predicted tensor (4D or 5D)
    :type img_sr: torch.Tensor
    :param img_hr: target tensor (4D or 5D)
    :type img_sr: torch.Tensor
    :param method: sr calculate method, psnr or ssim
    :type method: str
    :param to_y: whether convert image from format bgr to format y
    :type to_y: bool
    :param crop_border: number of pixels to crop
    :type crop_border: int
    :return: Average PSNR of the batch
    :rtype: float
    """
    if len(img_sr.size()) == 5:
        result = 0.
        for ind in range(img_sr.size(4)):
            result += compute_metric(img_sr[:, :, :, :, ind],
                                     img_hr[:, :, :, :, ind],
                                     method=method,
                                     to_y=to_y,
                                     scale=scale,
                                     max_rgb=max_rgb)
        return result / img_sr.size(4)
    else:
        return compute_metric(img_sr, img_hr, method=method, to_y=to_y, scale=scale, max_rgb=max_rgb)


@ClassFactory.register(ClassType.METRIC, alias='PSNR')
class PSNR(MetricBase):
    """Calculate SR metric between output and target."""

    def __init__(self, to_y=True, scale=2, max_rgb=1):
        self.method = "psnr"
        self.to_y = to_y
        self.sum = 0.
        self.pfm = 0.
        self.data_num = 0
        self.scale = scale
        self.max_rgb = max_rgb

    def __call__(self, output, target, *args, **kwargs):
        """Calculate SR metric.

        :param output: output of segmentation network
        :param target: ground truth from dataset
        :return: confusion matrix sum
        """
        # force channels first
        if output.size(1) != 1 and output.size(1) != 3:
            output = output.transpose(2, 3).transpose(1, 2)
        if target.size(1) != 1 and target.size(1) != 3:
            target = target.transpose(2, 3).transpose(1, 2)
        if isinstance(output, tuple):
            output = output[0]
        res = compute_sr_metric(output, target, self.method, self.to_y, self.scale, self.max_rgb)
        n = output.size(0)
        self.data_num += n
        self.sum = self.sum + res * n
        self.pfm = self.sum / self.data_num
        return res

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.sum = 0.
        self.pfm = 0.
        self.data_num = 0

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return self.pfm


@ClassFactory.register(ClassType.METRIC, alias='SSIM')
class SSIM(MetricBase):
    """Calculate SR metric between output and target."""

    def __init__(self, to_y=True, scale=2, max_rgb=1):
        self.method = "ssim"
        self.to_y = to_y
        self.sum = 0.
        self.pfm = 0.
        self.data_num = 0
        self.scale = scale
        self.max_rgb = max_rgb

    def __call__(self, output, target, *args, **kwargs):
        """Calculate SR metric.

        :param output: output of segmentation network
        :param target: ground truth from dataset
        :return: confusion matrix sum
        """
        # force channels first
        if output.size(1) != 1 and output.size(1) != 3:
            output = output.transpose(2, 3).transpose(1, 2)
        if target.size(1) != 1 and target.size(1) != 3:
            target = target.transpose(2, 3).transpose(1, 2)
        if isinstance(output, tuple):
            output = output[0]
        res = compute_sr_metric(output, target, self.method, self.to_y, self.scale, self.max_rgb)
        n = output.size(0)
        self.data_num += n
        self.sum = self.sum + res * n
        self.pfm = self.sum / self.data_num
        return res

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.sum = 0.
        self.pfm = 0.
        self.data_num = 0

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return self.pfm
