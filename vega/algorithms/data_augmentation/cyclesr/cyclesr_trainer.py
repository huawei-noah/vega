# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is the class for cyclesr trainworker."""
import datetime
import logging
import itertools
# common pkgs
import os
import sys
import time

# using for transfering results from training env to bucket
import numpy as np
# torch tools
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from vega.datasets.pytorch.common.dataset import Dataset
from vega.core.common import FileOps, init_log
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.pytorch import Trainer
from vega.search_space.networks import NetworkDesc
from .utils import Logger, AverageMeter
from .utils import TensorNorm

try:
    import horovod.torch as hvd
except Exception:
    # logging.warning("horovod not been installed, {}".format(str(e)))
    pass
# data-processing module
from .utils import find_best_PSNR


@ClassFactory.register(ClassType.TRAINER)
class CyclesrTrainer(Trainer):
    """TrainWorker for cyclesr task under unpaired setting.

    :param config: configuration produced by generator, or read directly from config file
    :type config: vega.core.Config or dict
    :param args: general args for the autoML task
    :type args: dict
    :param train_data: train dataset. TrainWorker will not produce a new one if specified here.
    :type train_data: torch.utils.data.Dataset
    :param valid_data: validataion dataset. TrainWorker will not produce a new one if specified here.
    :type valid_data: torch.utils.data.Dataset
    """

    def __init__(self, model, id):  # config, args=None, train_data=None, valid_data=None):
        """Initialize method."""
        super(CyclesrTrainer, self).__init__(model, id)  # config, args, train_data, valid_data)

    def _init_dataloader(self, mode):
        """Decode train dataset and validation dataset.

        :return: train dataset and validataion dataset
        :rtype: tuple of torch.utils.data.Dataset
        """
        dataset = Dataset(mode=mode)
        if self.horovod:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank())
            dataset.sampler = sampler
        return dataset

    def _init_model(self):
        """Initialize the model architecture for full train step.

        :return: train model
        :rtype: class
        """
        logging.info('Initializing model')
        if 'model_desc' in self.cfg and self.cfg.model_desc is not None:
            if self.horovod:
                hvd.join()
            model_desc = self.cfg.model_desc
            self.model_desc = self.cfg.model_desc
            net_desc = NetworkDesc(model_desc)
            model = net_desc.to_model()
            return model
        else:
            return None

    def batch_psnr(self, HR, SR):
        """Calculate the mean psnr in a batch.

        :param HR: HR image
        :type HR: torch FloatTensor
        :param SR: SR image
        :type SR: torch FloatTensor
        :return: mean psnr in a batch
        :rtype: Float
        """
        psnr = 20 * torch.log10(1 / torch.sqrt(torch.mean((HR - SR) ** 2, [1, 2, 3])))
        psnr = psnr.mean().item()
        return psnr

    def _train(self, trainloader, writer, epoch, model, print_freq=10):
        """Train process.

        :param trainloader: train dataset
        :type trainloader: torch.utils.data.DataLoader
        :param writer: record enent files to log dir
        :type writer: tensorboardX.SummaryWriter
        :param epoch: current epoch
        :type epoch: int
        :param model: cyclesr model with train mode
        :type model: CycleSRModel class(nn.Module)
        :param print_freq: frequency of showing training results on console
        :type print_freq: int
        """
        loss_sr = AverageMeter()
        loss_ga = AverageMeter()
        loss_cycA = AverageMeter()
        PSNRes = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()
        num_batches = len(trainloader)
        for batch_idx, data in enumerate(trainloader):
            model.set_mode('train')
            step = epoch * num_batches + batch_idx
            data_time.update(time.time() - end)
            #######################################################################
            model.optimize_CycleSR(data, epoch)

            # caclute psnr during training
            losses = model.get_current_losses()
            for name, loss in losses.items():
                writer.add_scalar("loss" + name, loss, step)  # store the loss in tensorboardX
            batchsize = data['X'].size(0)
            loss_sr.update(losses['SR'], batchsize)
            loss_ga.update(losses['G'], batchsize)
            loss_cycA.update(losses['rec_X'], batchsize)
            # logging.info("HR: {}. SR: {}".format(model.HR.data))
            if epoch < 6:
                psnr = self.batch_psnr(model.HR.data, model.G_SR.data)
            else:
                psnr = self.batch_psnr(model.HR.data, model.SR.data)
            PSNRes.update(psnr, batchsize)
            writer.add_scalar("training_psnr", psnr, step)  # store the psnr

            batch_time.update(time.time() - end)
            # print result
            if (batch_idx + 1) % print_freq == 0:
                if not self.cfg.cuda or (self.cfg.cuda and self.is_chief):
                    logging.info('[epoch {0},iter {1}/{2}]\t'
                                 'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                                 'Data {data_time.val:.3f}({data_time.avg:.3f})\t'
                                 'SR MSE {mse.val:.5f}({mse.avg:.5f})\t'
                                 'psnr {psnr.val:.3f}({psnr.avg:.3f})\t'
                                 'G_A {loss_ga.val:.5f}({loss_ga.avg:.5f})\t'
                                 'Cycle_A {loss_cycA.val:.5f}({loss_cycA.avg:.5f})'
                                 .format(epoch, batch_idx + 1, num_batches, batch_time=batch_time, data_time=data_time,
                                         mse=loss_sr, psnr=PSNRes, loss_ga=loss_ga, loss_cycA=loss_cycA))
            end = time.time()

    def getValImg(self, dataset, val_num=5):
        """Get val_num images for showing outputs of cycleGAN during training.

        :param dataset: valid dataset
        :type dataset: torch.utils.data.Dataset
        :param val_num: number of selected images, defualt: 5
        :type val_num: int
        :return: list of selected valid images
        :rtype: list
        """
        val_imgs = []
        for i in range(val_num):
            img = dataset[(i * (len(dataset) - 1)) // 5]
            img["X"] = torch.unsqueeze(img['X'], 0)
            img['Y'] = torch.unsqueeze(img['Y'], 0)
            img['HR'] = torch.unsqueeze(img['HR'], 0)
            val_imgs.append(img)
        return val_imgs

    def _evalGAN(self, model, imgs, epoch, writer):
        """Save images to event file.

        :param model: cyclesr model
        :type model: CycleSRModel class(nn.Module)
        :param imgs: list of selected valid images
        :type imgs: list
        :param epoch: current epoch
        :type epoch: int
        :param writer: record enent files to log dir
        :type writer: tensorboardX.SummaryWriter
        """
        model.set_mode('eval')
        with torch.no_grad():
            for i, img in enumerate(imgs):
                real_X = img['X'].cuda()
                real_Y = img['Y'].cuda()
                HR = img['HR'].cuda()
                fake_Y = model.netG(real_X)  # G(X)
                rec_X = model.netF(fake_Y)  # F(G(X))
                fake_X = model.netF(real_Y)  # F(Y)
                rec_Y = model.netG(fake_X)  # G(F(Y))

                G_SR = model.netSR(fake_Y)  # SR(G(X))
                writer.add_image("G_SR" + str(i), TensorNorm((G_SR[0])), epoch)
                writer.add_image("HR" + str(i), TensorNorm((HR[0])), epoch)
                writer.add_image("Real_bicubic" + str(i), TensorNorm((real_X[0])), epoch)
                writer.add_image("Fake_unknown" + str(i), TensorNorm((fake_Y[0])), epoch)
                writer.add_image("Real_unknown" + str(i), TensorNorm((real_Y[0])), epoch)
                writer.add_image("Fake_bicubic" + str(i), TensorNorm((fake_X[0])), epoch)
                writer.add_image("Rec_bicubic" + str(i), TensorNorm((rec_X[0])), epoch)
                writer.add_image("Rec_unknown" + str(i), TensorNorm((rec_Y[0])), epoch)

    def _valid(self, model, val_dataloader, epoch, eval_epoch, writer, ps_offset=10, val_sr_num=20):
        """Validate process of cyclesr.

        :param model: cyclesr model
        :type model: CycleSRModel class(nn.Module)
        :param val_dataloader: validate dataset
        :type val_dataloader: torch.utils.data.DataLoader
        :param epoch: current epoch
        :type epoch: int
        :param eval_epoch: frequency of evaluation
        :type eval_epoch: int
        :param writer: record enent files to log dir
        :type writer: tensorboardX.SummaryWriter
        :param ps_offset: pixel offset when calculating psnr during evaluation, default: 10
        :type ps_offset: int
        :param val_sr_num: number of selected images for testing sr model
        :type val_sr_num: int
        :return: mean psnr of whole validation images or None
        :rtype: int or None
        """
        SRnet = model.netSR
        SRnet.eval()
        val_PSNR = []
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                val_LR = data['Y']
                if "HR" in data.keys():
                    HR = data['HR']
                else:
                    HR = None
                SR = SRnet(val_LR.cuda())
                SR = torch.clamp(SR, 0.0, 1.0)
                if i < val_sr_num:
                    if i == 0:
                        logging.info('Saving real LR test images to tensorboard......')
                    writer.add_image("Val_SR" + str(i), TensorNorm((SR)), epoch)
                    if epoch == eval_epoch:
                        writer.add_image('Val_LR' + str(i), TensorNorm((val_LR)), epoch)
                        if HR is not None:
                            writer.add_image('Val_HR' + str(i), TensorNorm((HR)), epoch)
                    if i == val_sr_num - 1:
                        logging.info('***** Save Done! *****')
                else:
                    if HR is None:
                        return None
                val_PSNR.append(find_best_PSNR(HR.cuda(), SR, ps_offset) if HR is not None else None)
            if all(val_PSNR):
                ave_PSNR = np.asarray(val_PSNR).mean()
            else:
                ave_PSNR = None
            return ave_PSNR

    def train_process(self):
        """Whole train and validate process for the fully train cyclesr."""
        # self._init_all_settings()
        init_log(log_file="worker_{}.txt".format(self.worker_id))
        if self.cfg.cuda:
            self._init_cuda_setting()
        self.model = self._init_model()
        if self.horovod:
            self._horovod_init_optimizer()
            self._init_horovod_setting()
        self.train_data = self._init_dataloader('train')
        self.valid_data = self._init_dataloader('test')
        train_dataloader = self.train_data.dataloader
        valid_dataloader = self.valid_data.dataloader

        writer = SummaryWriter(self.get_local_worker_path())

        start_time = time.time()
        train_time = 0
        best_psnr = -np.inf
        best_epoch = 0
        logging.info("==> Start training")
        val_gan_imgs = self.getValImg(self.train_data, val_num=5)
        for epoch in range(self.cfg.epoch_count, self.cfg.n_epoch + self.cfg.n_epoch_decay + 1):
            self.model.update_learning_rate(
                epoch,
                self.cfg.model_desc.custom.cyc_lr,
                self.cfg.model_desc.custom.SR_lr,
                self.cfg.n_epoch,
                self.cfg.n_epoch_decay)
            start_train_time = time.time()
            self._train(train_dataloader, writer, epoch, self.model, print_freq=self.cfg.print_freq)
            train_time += round(time.time() - start_train_time)
            # validation
            ###############################################################################
            if (epoch % self.cfg.eval_epoch == 0):
                logging.info("==> Validng")
                self._evalGAN(self.model, val_gan_imgs, epoch, writer)
                val_ave_psnr = self._valid(self.model, valid_dataloader, epoch, self.cfg.eval_epoch, writer,
                                           self.cfg.val_ps_offset)
                if val_ave_psnr is not None:
                    logging.info("==> Current ave psnr is {:.3f}".format(val_ave_psnr))
                    if val_ave_psnr > best_psnr:
                        best_psnr = val_ave_psnr
                        best_epoch = epoch
                    logging.info(
                        "==> Best PSNR on val dataset {:.3f}, achieved at epoch {}".format(best_psnr, best_epoch))
                    best_model_name = 'model_best'
                    self._save_checkpoint(best_model_name)
                model_name = 'epoch' + str(epoch)
                logging.info("Saving checkpoints to {}".format(model_name))
                self._save_checkpoint(model_name)
                self._backup()
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        train_time = str(datetime.timedelta(seconds=train_time))
        logging.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
        self._backup()
        logging.info("***** Move Done! *****")

    def _save_checkpoint(self, epoch):
        """Save model weights.

        :param epoch: current epoch
        :type epoch: int
        """
        for name in self.model.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_dir = os.path.join(self.get_local_worker_path(), str(epoch))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self.model, 'net' + name)

                if self.cfg.cuda and torch.cuda.is_available():
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    torch.save(net.module.state_dict(), save_path)
                    # net.cuda()
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def _init_horovod_setting(self):
        """Init horovod setting."""
        self.is_chief = True
        # SR
        hvd.broadcast_parameters(self.model.netSR.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.model.optimizer_SR, root_rank=0)
        # G F
        hvd.broadcast_parameters(self.model.netG.state_dict(), root_rank=0)
        hvd.broadcast_parameters(self.model.netF.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.model.optimizer_G, root_rank=0)
        # D_X
        hvd.broadcast_parameters(self.model.netD_X.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.model.optimizer_D_X, root_rank=0)
        # D_Y
        hvd.broadcast_parameters(self.model.netD_Y.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.model.optimizer_D_Y, root_rank=0)
        if hvd.rank() != 0:
            self.is_chief = False
        else:
            self.is_chief = True

    def _horovod_init_optimizer(self):
        # SR optimizer
        self.model.optimizer_SR = hvd.DistributedOptimizer(
            self.model.optimizer_SR,
            named_parameters=self.model.netSR.named_parameters(),
            compression=hvd.Compression.none
        )
        # G optimizer
        self.model.optimizer_G = hvd.DistributedOptimizer(
            self.model.optimizer_G,
            named_parameters=itertools.chain(self.model.netG.named_parameters(), self.model.netF.named_parameters()),
            compression=hvd.Compression.none
        )
        # D_X optimizer
        self.model.optimizer_D_X = hvd.DistributedOptimizer(
            self.model.optimizer_D_X,
            named_parameters=self.model.netD_X.named_parameters(),
            compression=hvd.Compression.none
        )
        # D_Y optimizer
        self.model.optimizer_D_Y = hvd.DistributedOptimizer(
            self.model.optimizer_D_Y,
            named_parameters=self.model.netD_Y.named_parameters(),
            compression=hvd.Compression.none
        )
