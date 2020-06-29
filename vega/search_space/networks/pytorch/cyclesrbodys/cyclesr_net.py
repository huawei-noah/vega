# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is the class for Cyclesr model."""
import logging
import random

import numpy
import torch

from .trans_model import TransModel
from .networks import initialize, requires_grad
from .srmodels import VDSR, SRResNet
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


def define_SR(opt, use_cuda, use_horovod):
    """SR model creation.

    :param config: SR model configures
    :type config: dict
    :param use_cuda: if use cuda
    :type use_cuda: bool
    :return: SR model
    :rtype: nn.Module
    """
    # logging.info("==> We are using SR model: {}".format(config.which_model))
    logging.info("==> Norm type in {} is {}".format(opt.name, opt.SR_norm_type))
    if (opt.name == "VDSR"):
        net = VDSR(in_nc=opt.input_nc, out_nc=opt.input_nc, nf=opt.SR_nf, nb=opt.SR_nb,
                   norm_type=opt.SR_norm_type, upscale=opt.upscale, act_type='relu',
                   up_mode='pixelshuffle')
    if (opt.name == "SRResNet"):
        net = SRResNet(in_nc=opt.input_nc, out_nc=opt.input_nc, nf=opt.SR_nf, nb=opt.SR_nb,
                       upscale=opt.upscale, norm_type=opt.SR_norm_type, act_type='relu',
                       up_mode='pixelshuffle')
    [net] = initialize([net], use_cuda=use_cuda, use_horovod=use_horovod)
    return net


@NetworkFactory.register(NetTypes.CUSTOM)
class CycleSRModel(TransModel):
    """CycleSRModel Class definition.

    :param config: cyclesr model configures
    :type config: dict
    :param args: general configures which including basic configures for training(gpus, epochs and so on)
    :type args: dict
    """

    def __init__(self, cfg):
        """Initialize method."""
        self.use_cuda = cfg.use_cuda
        self.use_horovod = cfg.use_horovod
        self.SR_lr = cfg.SR_lr
        self.cyc_lr = cfg.cyc_lr
        super(CycleSRModel, self).__init__(cfg)
        self.max_norm = cfg.grad_clip
        self.loss_names.append("G_SR")
        self.loss_names.append("SR")
        self.loss_SR = 0
        self.loss_G_SR = 0
        self.SR_lam = cfg.SR_lam
        self.cycleSR_lam = cfg.cycleSR_lam
        logging.info("Now we are using CycleGan with SR")

        self.G_SR = None
        self.HR = None
        self.LR = None
        self.SR = None
        # add model names
        self.model_names.append("SR")
        self.netSR = define_SR(cfg.VDSR, self.use_cuda, self.use_horovod)
        self.criterionSR = torch.nn.MSELoss().cuda()
        # initialize optimizers
        self.optimizer_SR = torch.optim.Adam(self.netSR.parameters(), lr=cfg.SR_lr, betas=(0.5, 0.999))

    def forward(self):
        """Forward."""
        self.fake_Y = self.netG(self.real_X)  # G(X)
        self.rec_X = self.netF(self.fake_Y)  # F(G(X))
        self.fake_X = self.netF(self.real_Y)  # F(Y)
        self.rec_Y = self.netG(self.fake_X)  # G(F(Y))

        self.G_SR = self.netSR(self.fake_Y)  # SR(G(X))

    def set_mode(self, mode):
        """Set the mode of model to train."""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
            if mode == 'eval':
                net.eval()
            elif mode == 'train':
                net.train()
            else:
                raise ValueError("Not recognize mode %s.".format(mode))

    def update_G(self, lam):
        """Update generators.

        :param lam: the weights of SR loss when update G_A, G_B
        :type lam: int
        """
        # Identity loss
        self.loss_idt_Y = self.criterionIdt(self.netG(self.real_Y), self.real_Y) * self.lambda_cycle * self.lambda_idt
        self.loss_idt_X = self.criterionIdt(self.netF(self.real_X), self.real_X) * self.lambda_cycle * self.lambda_idt

        total_idt = self.loss_idt_Y + self.loss_idt_X
        # Adversial loss D_Y(G(X))
        self.loss_G = self.criterionGAN(self.netD_Y(self.fake_Y), self.real_label)
        # Adversial loss D_X(F(Y))
        self.loss_F = self.criterionGAN(self.netD_X(self.fake_X), self.real_label)
        # Reconstruction loss X ||F(G(X)) - X||
        self.loss_rec_X = self.criterionCycle(self.rec_X, self.real_X) * self.lambda_cycle
        # Reconstruction loss Y ||G(F(Y)) - Y||
        self.loss_rec_Y = self.criterionCycle(self.rec_Y, self.real_Y) * self.lambda_cycle
        self.loss_total = self.loss_G + self.loss_F + self.loss_rec_Y + self.loss_rec_X + total_idt

        # This part is for SR ##################
        self.loss_G_SR = self.criterionSR(self.G_SR, self.HR) * lam
        self.loss_total = self.loss_total + self.loss_G_SR
        self.loss_total.backward()

    def update_SR(self, LR, HR, lam):
        """Calculate the loss for SR model.

        :param LR: LR image
        :type LR: torch.FloatTensor
        :param HR: HR image
        :type HR: torch.FloatTensor
        :param lam: the weights of SR loss for updating SR
        :type lam: int
        """
        self.LR = LR.data.cuda()
        self.HR = HR.data.cuda()

        self.SR = self.netSR(self.LR)
        self.loss_SR = lam * self.criterionSR(self.SR, self.HR)
        self.loss_SR.backward()

    def optimize_SR(self, LR, HR, lam):
        """Optimize SR model.

        :param LR: LR image
        :type LR: torch.FloatTensor
        :param HR: HR image
        :type HR: torch.FloatTensor
        :param lam: the weights of SR loss for updating SR
        :type lam: int
        """
        requires_grad([self.netSR], True)
        self.optimizer_SR.zero_grad()

        self.update_SR(LR, HR, lam)

        torch.nn.utils.clip_grad_norm_(self.netSR.parameters(), max_norm=self.max_norm)

        self.optimizer_SR.step()

    def optimize_transmodel(self, lam):
        """Calculate losses, gradients, and update network weights; called in every training iteration.

        :param lam: the weights of SR loss when update generator G.
        :type lam: int
        """
        # forward
        self.forward()
        # Following original GAN's optimize order, first, optimize D.
        requires_grad([self.netD_X, self.netD_Y], True)
        self.optimizer_D_X.zero_grad()
        self.optimizer_D_Y.zero_grad()
        self.update_D()
        torch.nn.utils.clip_grad_norm_(self.netD_X.parameters(), max_norm=self.max_norm)
        torch.nn.utils.clip_grad_norm_(self.netD_Y.parameters(), max_norm=self.max_norm)
        self.optimizer_D_X.step()
        self.optimizer_D_Y.step()
        # Then optimize G.
        requires_grad([self.netSR], False)
        requires_grad([self.netD_X, self.netD_Y], False)
        self.optimizer_G.zero_grad()
        self.update_G(lam)
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.max_norm)
        torch.nn.utils.clip_grad_norm_(self.netF.parameters(), max_norm=self.max_norm)
        self.optimizer_G.step()

    def optimize_CycleSR(self, input, epoch):
        """Optimize CycleSR model.

        :param epoch: current epoch
        :type epoch: int
        """
        self.real_X = input['X'].cuda()
        self.real_Y = input['Y'].cuda()
        self.HR = input['HR'].cuda()
        self.batch_size = self.real_X.shape[0]
        if epoch < 6:
            self.optimize_transmodel(0)
        elif 6 <= epoch < 7:
            self.optimize_SR(self.real_X, self.HR, 1)
        else:
            self.optimize_transmodel(self.cycleSR_lam)
            self.optimize_SR(self.fake_Y, self.HR, self.SR_lam)
