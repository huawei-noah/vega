# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of Gan task.

This is not a official metric for IS score and FID. in order to publish paper,
you can use code from https://github.com/lzhbrian/metrics.

"""
import torch
import numpy as np
from vega.common import ClassFactory, ClassType
import os
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torchvision.models.inception import inception_v3
from scipy.stats import entropy


def inception_score(imgs, model_checkpoint, cuda=True, batch_size=100, resize=True, splits=1):
    """Compute the inception score of the generated images imgs."""
    N = len(imgs)
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(
        pretrained=False, transform_input=False).type(dtype)
    if model_checkpoint is None:
        model_checkpoint = "/workspace/code_paper/inception_v3_google-1a9a5a14.pth"
    if not os.path.isfile(model_checkpoint):
        raise Exception(f"Pretrained model is not existed, model={model_checkpoint}")
    checkpoint = torch.load(model_checkpoint)
    inception_model.load_state_dict(checkpoint)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


@ClassFactory.register(ClassType.METRIC)
class GANMetric(object):
    """Calculate SR metric between output and target."""

    def __init__(self, model_checkpoint=None, latent_dim=120):
        self.model_checkpoint = model_checkpoint
        self.sum = 0.
        self.pfm = 0.
        self.latent_dim = latent_dim

    def __call__(self, output=None, target=None, model=None, **kwargs):
        """Calculate SR metric.

        :param output: output of segmentation network
        :param target: ground truth from dataset
        :return: confusion matrix sum
        """
        if model is not None:
            img_list = list()
            eval_iter = 50000 // 100
            self.sum = 50000
            for iter_idx in range(eval_iter):
                z = torch.cuda.FloatTensor(
                    np.random.normal(0, 1, (100, self.latent_dim)))
                # generate a batch of images
                gen_imgs = model(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0)
                img_list.extend(list(gen_imgs))
            mean, std = inception_score(img_list, self.model_checkpoint)
            self.pfm = mean
            return mean
        else:
            raise Exception("Must give a model")

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.sum = 0.
        self.pfm = 0.
        self.data_num = 0

    def summary(self):
        """Summary all cached records, here is the last pfm record."""
        return self.pfm

    @property
    def results(self):
        """Return metrics results."""
        res = {}
        if self.model is None:
            res["value"] = self.pfm
        return res
