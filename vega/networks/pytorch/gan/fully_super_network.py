# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined GAN Blocks For Image Generation."""
import torch
from torch import nn
import numpy as np
from .fully_basic_blocks import Cell, OptimizedDisBlock, DisCell
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.NETWORK)
class Generator(nn.Module):
    """Generator class."""

    def __init__(self, latent_dim, bottom_width, gf_dim, genotypes):
        super(Generator, self).__init__()
        self.gf_dim = gf_dim
        self.bottom_width = bottom_width

        self.base_latent_dim = latent_dim // 3
        self.l1 = nn.Linear(self.base_latent_dim,
                            (self.bottom_width ** 2) * self.gf_dim)
        self.l2 = nn.Linear(self.base_latent_dim, ((
            self.bottom_width * 2) ** 2) * self.gf_dim)
        self.l3 = nn.Linear(self.base_latent_dim, ((
            self.bottom_width * 4) ** 2) * self.gf_dim)
        self.cell1 = Cell(self.gf_dim, self.gf_dim, 'nearest',
                          genotypes[0], num_skip_in=0)
        self.cell2 = Cell(self.gf_dim, self.gf_dim, 'bilinear',
                          genotypes[1], num_skip_in=1)
        self.cell3 = Cell(self.gf_dim, self.gf_dim, 'nearest',
                          genotypes[2], num_skip_in=2)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(self.gf_dim), nn.ReLU(), nn.Conv2d(
                self.gf_dim, 3, 3, 1, 1), nn.Tanh()
        )

    def forward(self, z):
        """Call Generator."""
        h = self.l1(z[:, :self.base_latent_dim])\
            .view(-1, self.gf_dim, self.bottom_width, self.bottom_width)

        n1 = self.l2(z[:, self.base_latent_dim:self.base_latent_dim * 2])\
            .view(-1, self.gf_dim, self.bottom_width * 2, self.bottom_width * 2)

        n2 = self.l3(z[:, self.base_latent_dim * 2:])\
            .view(-1, self.gf_dim, self.bottom_width * 4, self.bottom_width * 4)

        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1 + n1, (h1_skip_out, ))
        __, h3 = self.cell3(h2 + n2, (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)

        return output


@ClassFactory.register(ClassType.NETWORK)
class Discriminator(nn.Module):
    """Discriminator class."""

    def __init__(self, df_dim, genotypes, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(3, self.ch)
        self.block2 = DisCell(
            self.ch, self.ch, activation=activation, genotype=genotypes[0])
        self.block3 = DisCell(
            self.ch, self.ch, activation=activation, genotype=genotypes[1])
        self.block4 = DisCell(
            self.ch, self.ch, activation=activation, genotype=genotypes[2])
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        """Call Discriminator."""
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output


@ClassFactory.register(ClassType.NETWORK)
class GAN(nn.Module):
    """GAN."""

    def __init__(self, generator, discriminator, latent_dim, gen_bs):
        super(GAN, self).__init__()
        self.generator = ClassFactory.get_cls(
            ClassType.NETWORK, generator.pop('type'))(**generator)
        self.latent_dim = latent_dim
        self.gen_bs = gen_bs
        self.discriminator = ClassFactory.get_cls(
            ClassType.NETWORK, discriminator.pop('type'))(**discriminator)

    def forward(self, imgs, step_name):
        """Call GAN."""
        if step_name == 'dis':
            z = torch.cuda.FloatTensor(np.random.normal(
                0, 1, (imgs.shape[0], self.latent_dim)))
            real_imgs = imgs
            real_validity = self.discriminator(real_imgs)
            fake_imgs = self.generator(z).detach()
            assert fake_imgs.size() == real_imgs.size()
            fake_validity = self.discriminator(fake_imgs)
            return (real_validity, fake_validity)
        else:
            gen_z = torch.cuda.FloatTensor(np.random.normal(
                0, 1, (self.gen_bs, self.latent_dim)))
            gen_imgs = self.generator(gen_z)
            fake_validity = self.discriminator(gen_imgs)
            return fake_validity
