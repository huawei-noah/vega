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

"""This is the class for Translation model."""
import itertools
import random
from collections import OrderedDict

import logging
import torch

import vega
from .networks import Generator, PatchDiscriminator, initialize, requires_grad


class TransModel:
    """This class implements the translation model, such as cyclegan.

    :param opt: translation model configures
    :type opt: dict
    """

    def __init__(self, cfg):
        opt = cfg.cyclegan
        self.use_cuda = cfg.use_cuda
        self.use_distributed = cfg.use_distributed
        """Initialize method."""
        self.loss_names = ['D_Y', 'G', 'rec_X']
        self.model_names = ['G', 'F', 'D_X', 'D_Y']

        self.netG = Generator(opt.input_nc, opt.output_nc, opt.ngf, norm_type=opt.norm,
                              act_type='relu', up_mode=opt.up_mode, name='G')
        self.netF = Generator(opt.input_nc, opt.output_nc, opt.ngf, norm_type=opt.norm,
                              act_type='relu', up_mode=opt.up_mode, name='F')
        self.netD_X = PatchDiscriminator(opt.output_nc, opt.ndf, norm_type=opt.norm, act_type='leakyrelu', name='D_X')
        self.netD_Y = PatchDiscriminator(opt.output_nc, opt.ndf, norm_type=opt.norm, act_type='leakyrelu', name='D_Y')
        # Initialize networks.
        [self.netG, self.netF, self.netD_X, self.netD_Y] = initialize(
            [self.netG, self.netF, self.netD_X, self.netD_Y],
            use_cuda=self.use_cuda, use_distributed=self.use_distributed)
        self.fake_X_, self.fake_Y_ = None, None
        self.lambda_idt = opt.lambda_identity
        self.lambda_cycle = opt.lambda_cycle

        # define loss functions
        if vega.is_npu_device():
            self.criterionGAN = torch.nn.MSELoss().npu()
        else:
            self.criterionGAN = torch.nn.MSELoss().cuda()  # define GAN loss
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netF.parameters()),
                                            lr=self.cyc_lr, betas=(0.5, 0.999))
        self.optimizer_D_X = torch.optim.Adam(self.netD_X.parameters(), lr=self.cyc_lr, betas=(0.5, 0.999))
        self.optimizer_D_Y = torch.optim.Adam(self.netD_Y.parameters(), lr=self.cyc_lr, betas=(0.5, 0.999))
        self.fake_X_buffer = ShuffleBuffer(opt.buffer_size)
        self.fake_Y_buffer = ShuffleBuffer(opt.buffer_size)

    def forward(self):
        """Forward."""
        self.fake_Y = self.netG(self.real_X)  # G(X)
        self.rec_X = self.netF(self.fake_Y)  # F(G(X))
        self.fake_X = self.netF(self.real_Y)  # F(Y)
        self.rec_Y = self.netG(self.fake_X)  # G(F(Y))

    def cal_adversial_loss(self, netD, real, fake):
        """Calculate adversial loss for the discriminator.

        :param netD: the discriminator D
        :type netD: nn.Module
        :param real: real images
        :type real: tensor
        :param fake: fake images
        :type fake: tensor
        :return: discriminator loss
        :rtype: torch.FloatTensor
        """
        pred_real = netD(real)
        if vega.is_npu_device():
            self.real_label = torch.tensor(1.0).expand_as(pred_real).npu()
        else:
            self.real_label = torch.tensor(1.0).expand_as(pred_real).cuda()
        loss_D_real = self.criterionGAN(pred_real, self.real_label)
        pred_fake = netD(fake.detach())
        if vega.is_npu_device():
            self.fake_label = torch.tensor(0.0).expand_as(pred_fake).npu()
        else:
            self.fake_label = torch.tensor(0.0).expand_as(pred_fake).cuda()
        loss_D_fake = self.criterionGAN(pred_fake, self.fake_label)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def update_D(self):
        """Update discriminators."""
        # Discriminator X.
        fake_X = self.fake_X_buffer.choose(self.fake_X)
        self.loss_D_X = self.cal_adversial_loss(self.netD_X, self.real_X, fake_X)

        # Discriminator Y.
        fake_Y = self.fake_Y_buffer.choose(self.fake_Y)
        self.loss_D_Y = self.cal_adversial_loss(self.netD_Y, self.real_Y, fake_Y)

    def update_G(self):
        """Update generators."""
        # Identity loss
        self.loss_idt_Y = self.criterionIdt(self.netG(self.real_Y),
                                            self.real_Y) * self.lambda_cycle * self.lambda_idt
        self.loss_idt_X = self.criterionIdt(self.netF(self.real_X),
                                            self.real_X) * self.lambda_cycle * self.lambda_idt

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
        self.loss_total.backward()

    def optimize_transmodel(self, input):
        """Optimize translation model."""
        if vega.is_npu_device():
            self.real_X = input['X'].npu()
            self.real_Y = input['Y'].npu()
        else:
            self.real_X = input['X'].cuda()
            self.real_Y = input['Y'].cuda()
        self.batch_size = self.real_X.shape[0]
        self.forward()
        # Following original GAN's optimize order, first, optimize D.
        requires_grad([self.netD_X, self.netD_Y], True)
        self.optimizer_D_X.zero_grad()
        self.optimizer_D_Y.zero_grad()
        self.update_D()
        self.optimizer_D_X.step()
        self.optimizer_D_Y.step()
        # Then optimize G.
        requires_grad([self.netD_X, self.netD_Y], False)
        self.optimizer_G.zero_grad()
        self.update_G()
        self.optimizer_G.step()

    def update_learning_rate(self, epoch, cyc_lr, SR_lr, n_epoch, n_epoch_decay):
        """Update learning rates for all the networks; called at the end of every epoch.

        :param epoch: current epoch
        :type epoch: int
        :param cyc_lr: learning rate of cyclegan
        :type cyc_lr: float
        :param SR_lr: learning rate of SR model
        :type SR_lr: float
        :param n_epoch: number of epochs with the initial learning rate
        :type n_epoch: int
        :param n_epoch_decay: number of epochs to linearly decay learning rate to zero
        :type n_epoch_decay: int
        """
        tmp_cyc_lr = cyc_lr - max(0, epoch - n_epoch) * cyc_lr / n_epoch_decay
        tmp_SR_lr = SR_lr - max(0, epoch - n_epoch) * SR_lr / n_epoch_decay
        logging.info("*********** epoch: {} **********".format(epoch))
        if epoch < 6:
            self.adjust_lr('G', self.optimizer_G, tmp_cyc_lr)
            self.adjust_lr('D_X', self.optimizer_D_X, tmp_cyc_lr)
            self.adjust_lr('D_Y', self.optimizer_D_Y, tmp_cyc_lr)
            self.adjust_lr('SR', self.optimizer_SR, 0)

        elif epoch < 7:
            self.adjust_lr('G', self.optimizer_G, 0)
            self.adjust_lr('D_X', self.optimizer_D_X, 0)
            self.adjust_lr('D_Y', self.optimizer_D_Y, 0)
            self.adjust_lr('SR', self.optimizer_SR, tmp_SR_lr)

        else:
            self.adjust_lr('G', self.optimizer_G, tmp_cyc_lr)
            self.adjust_lr('D_X', self.optimizer_D_X, tmp_cyc_lr)
            self.adjust_lr('D_Y', self.optimizer_D_Y, tmp_cyc_lr)
            self.adjust_lr('SR', self.optimizer_SR, tmp_SR_lr)
        logging.info("*********************************")

    def adjust_lr(self, name, optimizer, lr):
        """Adjust learning rate for the corresponding model.

        :param name: name of model
        :type name: str
        :param optimizer: the optimizer of the corresponding model
        :type optimizer: torch.optim
        :param lr: learning rate to be adjusted
        :type lr: float
        """
        for p in optimizer.param_groups:
            p['lr'] = lr
            logging.info('==> ' + name + ' learning rate: %.7f' % lr)

    def get_current_losses(self):
        """Return traning losses."""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret


class ShuffleBuffer():
    """Random choose previous generated images or ones produced by the latest generators.

    :param buffer_size: the size of image buffer
    :type buffer_size: int
    """

    def __init__(self, buffer_size):
        """Initialize the ImagePool class.

        :param buffer_size: the size of image buffer
        :type buffer_size: int
        """
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.images = []

    def choose(self, images, prob=0.5):
        """Return an image from the pool.

        :param images: the latest generated images from the generator
        :type images: list
        :param prob: probability (0~1) of return previous images from buffer
        :type prob: float
        :return: Return images from the buffer
        :rtype: list
        """
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.images.append(image)
                return_images.append(image)
                self.num_imgs += 1
            else:
                p = random.uniform(0, 1)
                if p < prob:
                    idx = random.randint(0, self.buffer_size - 1)
                    stored_image = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(stored_image)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images
