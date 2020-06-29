# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""CARS trainer."""
import os
import logging
import copy
import json
import numpy as np
from collections import namedtuple
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from .utils import eval_model_parameters
from .nsga3 import CARS_NSGA
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space import SearchSpace
from vega.search_space.search_algs import SearchAlgorithm
from vega.search_space.search_algs.nsga_iii import SortAndSelectPopulation
from vega.datasets.pytorch import Dataset
from vega.core.common import FileOps, Config, DefaultConfig
from vega.core.metrics.pytorch import Metrics
from vega.search_space.networks.pytorch import CARSDartsNetwork
from vega.core.trainer.callbacks import Callback, ModelStatistics

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


@ClassFactory.register(ClassType.CALLBACK)
class CARSTrainerCallback(Callback):
    """A special callback for CARSTrainer."""

    def __init__(self):
        super(CARSTrainerCallback, self).__init__()
        self.alg_policy = ClassFactory.__configs__.search_algorithm.policy

    def before_train(self, epoch, logs=None):
        """Be called before the training process."""
        # Use zero valid_freq to supress default valid step
        self.trainer.auto_save_ckpt = False
        self.trainer.auto_save_perf = False
        self.trainer.valid_freq = 0
        cudnn.benchmark = True
        cudnn.enabled = True
        self.search_alg = SearchAlgorithm(SearchSpace())
        self.set_algorithm_model(self.trainer.model)
        # setup alphas
        n_individual = self.alg_policy.num_individual
        self.alphas = torch.cat([self.trainer.model.random_single_path().unsqueeze(0)
                                 for i in range(n_individual)], dim=0)
        self.trainer.train_loader = self.trainer._init_dataloader(mode='train')
        self.trainer.valid_loader = self.trainer._init_dataloader(mode='val')

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoach."""
        self.epoch = epoch
        self.trainer.lr_scheduler.step()

    def train_step(self, batch):
        """Replace the default train_step function."""
        self.trainer.model.train()
        input, target = batch
        self.trainer.optimizer.zero_grad()
        for j in range(self.alg_policy.num_individual_per_iter):
            i = np.random.randint(0, self.alg_policy.num_individual, 1)[0]
            if self.epoch < self.alg_policy.warmup:
                logits = self.trainer.model.forward_random(input)
            else:
                logits = self.trainer.model(input, self.alphas[i])
            loss = self.trainer.loss(logits, target)
            loss.backward(retain_graph=True)
            if self.epoch < self.alg_policy.warmup:
                break
        nn.utils.clip_grad_norm(
            self.trainer.model.parameters(), self.trainer.cfg.grad_clip)
        self.trainer.optimizer.step()
        return {'loss': loss.item(),
                'train_batch_output': logits}

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.search_evol_arch(epoch)

    def set_algorithm_model(self, model):
        """Set model to algorithm.

        :param model: network model
        :type model: torch.nn.Module
        """
        self.search_alg.set_model(model)

    def search_evol_arch(self, epoch):
        """Update architectures.

        :param epoch: The current epoch
        :type epoch: int
        :param valid_queue: valid dataloader
        :type valid_queue: dataloader
        :param model: The model to be trained
        :type model: nn.Module
        """
        if epoch >= self.alg_policy.start_ga_epoch and \
                (epoch - self.alg_policy.start_ga_epoch) % self.alg_policy.ga_interval == 0:
            self.save_model_checkpoint(
                self.trainer.model, 'weights_{}.pt'.format(epoch))
            for generation in range(self.alg_policy.num_generation):
                fitness = np.zeros(
                    int(self.alg_policy.num_individual * (1 + self.alg_policy.expand)))
                model_sizes = np.zeros(
                    int(self.alg_policy.num_individual * (1 + self.alg_policy.expand)))
                genotypes = []
                # generate offsprings using mutation and cross-over
                offsprings = self.search_alg.gen_offspring(self.alphas)
                self.alphas = torch.cat((self.alphas, offsprings), dim=0)
                # calculate fitness (accuracy) and #parameters
                for i in range(int(self.alg_policy.num_individual * (1 + self.alg_policy.expand))):
                    fitness[i], _ = self.search_infer_step(self.alphas[i])
                    genotypes.append(self.genotype_namedtuple(self.alphas[i]))
                    model_sizes[i] = self.eval_model_sizes(self.alphas[i])
                    logging.info('Valid_acc for invidual {} %f, size %f'.format(
                        i), fitness[i], model_sizes[i])
                # update population using pNSGA-III (CARS_NSGA)
                logging.info('############## Begin update alpha ############')
                if self.alg_policy.nsga_method == 'nsga3':
                    _, _, keep = SortAndSelectPopulation(
                        np.vstack((1 / fitness, model_sizes)), self.alg_policy.num_individual)
                elif self.alg_policy.nsga_method == 'cars_nsga':
                    nsga_objs = [model_sizes]
                    keep = CARS_NSGA(fitness, nsga_objs,
                                     self.alg_policy.num_individual)
                drop = list(set(list(
                    range(int(self.alg_policy.num_individual *
                              (1 + self.alg_policy.expand))))) - set(keep.tolist()))
                logging.info('############## KEEP ############')
                fitness_keep = []
                size_keep = []
                genotype_keep = []
                for i in keep:
                    logging.info('KEEP Valid_acc for invidual {} %f, size %f, genotype %s'.format(i), fitness[i],
                                 model_sizes[i], genotypes[i])
                    fitness_keep.append(fitness[i])
                    size_keep.append(model_sizes[i])
                    genotype_keep.append(genotypes[i])
                logging.info('############## DROP ############')
                for i in drop:
                    logging.info('DROP Valid_acc for invidual {} %f, size %f, genotype %s'.format(i), fitness[i],
                                 model_sizes[i], genotypes[i])
                if self.alg_policy.select_method == 'uniform':
                    selected_genotypes, selected_acc, selected_model_sizes = \
                        self.select_uniform_pareto_front(
                            np.array(fitness_keep), np.array(size_keep), genotype_keep)
                else:  # default: first
                    selected_genotypes, selected_acc, selected_model_sizes = \
                        self.select_first_pareto_front(
                            np.array(fitness_keep), np.array(size_keep), genotype_keep)
                ga_epoch = int(
                    (epoch - self.alg_policy.start_ga_epoch) / self.alg_policy.ga_interval)
                self.save_genotypes(selected_genotypes, selected_acc, selected_model_sizes,
                                    'genotype_selected_{}.txt'.format(ga_epoch))
                self.save_genotypes(genotype_keep, np.array(fitness_keep), np.array(size_keep),
                                    'genotype_keep_{}.txt'.format(ga_epoch))
                self.save_genotypes_to_json(genotype_keep, np.array(fitness_keep), np.array(size_keep),
                                            'genotype_keep_jsons', ga_epoch)
                self.alphas = self.alphas[keep].clone()
                logging.info('############## End update alpha ############')

    def search_infer_step(self, alpha):
        """Infer in search stage.

        :param valid_queue: valid dataloader
        :type valid_queue: dataloader
        :param model: The model to be trained
        :type model: nn.Module
        :param alpha: encoding of a model
        :type alpha: array
        :return: Average top1 acc and loss
        :rtype: nn.Tensor
        """
        metrics = Metrics(self.trainer.cfg.metric)
        self.trainer.model.eval()
        with torch.no_grad():
            for step, (input, target) in enumerate(self.trainer.valid_loader):
                input = input.cuda()
                target = target.cuda(non_blocking=True)
                logits = self.trainer.model(input, alpha)
                metrics(logits, target)
        top1 = metrics.results[0]
        return top1

    def select_first_pareto_front(self, fitness, obj, genotypes):
        """Select models in the first pareto front.

        :param fitness: fitness, e.g. accuracy
        :type fitness: ndarray
        :param obj: objectives (model sizes, FLOPS, latency etc)
        :type obj: ndarray
        :param genotypes: genotypes for searched models
        :type genotypes: list
        :return: The selected samples
        :rtype: list
        """
        F, _, selected_idx = SortAndSelectPopulation(np.vstack(
            (1 / fitness, obj)), self.alg_policy.pareto_model_num)
        selected_genotypes = []
        selected_acc = []
        selected_model_sizes = []
        for idx in selected_idx:
            selected_genotypes.append(genotypes[idx])
            selected_acc.append(fitness[idx])
            selected_model_sizes.append(obj[idx])
        return selected_genotypes, selected_acc, selected_model_sizes

    def select_uniform_pareto_front(self, fitness, obj, genotypes):
        """Select models in the first pareto front.

        :param fitness: fitness, e.g. accuracy
        :type fitness: ndarray
        :param obj: objectives (model sizes, FLOPS, latency etc)
        :type obj: ndarray
        :param genotypes: genotypes for searched models
        :type genotypes: list
        :return: The selected samples
        :rtype: list
        """
        # preprocess
        max_acc = fitness.max()
        keep = (fitness > max_acc * 0.5)
        fitness = fitness[keep]
        obj = obj[keep]
        genotypes = [i for (i, v) in zip(genotypes, keep) if v]
        max_obj = obj.max()
        min_obj = obj.min()
        grid_num = self.alg_policy.pareto_model_num
        grid = np.linspace(min_obj, max_obj, num=grid_num + 1)
        selected_idx = []
        for idx in range(grid_num):
            keep = (obj <= grid[idx]) | (obj > grid[idx + 1])
            sub_fitness = np.array(fitness)
            sub_fitness[keep] = 0
            selected_idx.append(sub_fitness.argmax())
        selected_genotypes = []
        selected_acc = []
        selected_model_sizes = []
        for idx in selected_idx:
            selected_genotypes.append(genotypes[idx])
            selected_acc.append(fitness[idx])
            selected_model_sizes.append(obj[idx])
        return selected_genotypes, selected_acc, selected_model_sizes

    def eval_model_sizes(self, alpha):
        """Calculate model size for a genotype.

        :param genotype: genotype for searched model
        :type genotype: list
        :return: The number of parameters
        :rtype: Float
        """
        normal = alpha[:self.trainer.model.len_alpha].data.cpu().numpy()
        reduce = alpha[self.trainer.model.len_alpha:].data.cpu().numpy()
        child_desc = self.search_alg.codec.calc_genotype([normal, reduce])
        child_cfg = copy.deepcopy(self.search_alg.codec.darts_cfg.super_network)
        child_cfg.normal.genotype = child_desc[0]
        child_cfg.reduce.genotype = child_desc[1]
        net = CARSDartsNetwork(child_cfg)
        model_size = eval_model_parameters(net)
        return model_size

    def genotype_namedtuple(self, alpha):
        """Obtain genotype.

        :param alpha: alpha for cell
        :type alpha: Tensor
        :return: genotype
        :rtype: Genotype
        """
        normal = alpha[:self.trainer.model.len_alpha].data.cpu().numpy()
        reduce = alpha[self.trainer.model.len_alpha:].data.cpu().numpy()
        child_desc = self.search_alg.codec.calc_genotype([normal, reduce])
        _multiplier = 4
        concat = range(2 + self.trainer.model._steps -
                       _multiplier, self.trainer.model._steps + 2)
        genotype = Genotype(
            normal=child_desc[0], normal_concat=concat,
            reduce=child_desc[1], reduce_concat=concat
        )
        return genotype

    def save_model_checkpoint(self, model, model_name):
        """Save checkpoint for a model.

        :param model: A model
        :type model: nn.Module
        :param model_name: Path to save
        :type model_name: string
        """
        worker_path = self.trainer.get_local_worker_path()
        save_path = os.path.join(worker_path, model_name)
        _path, _ = os.path.split(save_path)
        if not os.path.isdir(_path):
            os.makedirs(_path)
        torch.save(model, save_path)
        logging.info("checkpoint saved to %s", save_path)

    def save_genotypes(self, genotypes, acc, obj, save_name):
        """Save genotypes.

        :param genotypes: Genotype for models
        :type genotypes: namedtuple Genotype
        :param acc: accuracy
        :type acc: ndarray
        :param obj: objectives, etc. FLOPs or number of parameters
        :type obj: ndarray
        :param save_name: Path to save
        :type save_name: string
        """
        worker_path = self.trainer.get_local_worker_path()
        save_path = os.path.join(worker_path, save_name)
        _path, _ = os.path.split(save_path)
        if not os.path.isdir(_path):
            os.makedirs(_path)
        with open(save_path, "w") as f:
            for idx in range(len(genotypes)):
                f.write('{}\t{}\t{}\n'.format(
                    acc[idx], obj[idx], genotypes[idx]))
        logging.info("genotypes saved to %s", save_path)

    def save_genotypes_to_json(self, genotypes, acc, obj, save_folder, ga_epoch):
        """Save genotypes.

        :param genotypes: Genotype for models
        :type genotypes: namedtuple Genotype
        :param acc: accuracy
        :type acc: ndarray
        :param obj: objectives, etc. FLOPs or number of parameters
        :type obj: ndarray
        :param save_name: Path to save
        :type save_name: string
        """
        if self.trainer.cfg.darts_template_file == "{default_darts_cifar10_template}":
            template = DefaultConfig().data.default_darts_cifar10_template
        elif self.trainer.cfg.darts_template_file == "{default_darts_imagenet_template}":
            template = DefaultConfig().data.default_darts_imagenet_template
        else:
            worker_path = self.trainer.get_local_worker_path()
            _path = os.path.join(
                worker_path, save_folder + '_{}'.format(ga_epoch))
            if not os.path.isdir(_path):
                os.makedirs(_path)
            base_file = os.path.basename(self.trainer.cfg.darts_template_file)
            local_template = FileOps.join_path(
                self.trainer.local_output_path, base_file)
            FileOps.copy_file(
                self.trainer.cfg.darts_template_file, local_template)
            with open(local_template, 'r') as f:
                template = json.load(f)

        for idx in range(len(genotypes)):
            template_cfg = Config(template)
            template_cfg.super_network.normal.genotype = genotypes[idx].normal
            template_cfg.super_network.reduce.genotype = genotypes[idx].reduce
            self.trainer.output_model_desc(idx, template_cfg)
