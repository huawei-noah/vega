# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer program for ESR_EA."""
import os
import torch
import logging
import json
import pandas as pd
from copy import deepcopy
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space import SearchSpace
from vega.search_space.codec import Codec
from vega.search_space.networks import NetworkDesc
from vega.core.common import FileOps, Config
from vega.core.metrics.pytorch import Metrics
from vega.core.trainer.callbacks import Callback


@ClassFactory.register(ClassType.CALLBACK)
class SREATrainerCallback(Callback):
    """Construct the trainer of ESR-EA."""

    def before_train(self, logs=None):
        """Be called before the whole train process."""
        self.cfg = self.trainer.cfg
        self.trainer.auto_save_ckpt = False
        self.trainer.auto_save_perf = False
        self.result_path = FileOps.join_path(
            self.trainer.local_base_path, "result")

    def make_batch(self, batch):
        """Make batch for each training step."""
        input = batch["LR"]
        target = batch["HR"]
        if self.cfg.cuda and not self.cfg.is_detection_trainer:
            input = input.cuda() / 255.0
            target = target.cuda() / 255.0
        return input, target

    def after_epoch(self, epoch, logs=None):
        """Be called after one epoch training."""
        # Get summary perfs from logs from built-in MetricsEvaluator callback.
        self.performance = logs.get('summary_perfs', None)
        gflops = self.performance['gflops']
        kparams = self.performance['kparams']
        cur_valid_perfs = self.performance['cur_valid_perfs']
        best_valid_perfs = self.performance['best_valid_perfs']
        perfs = {'gflops': gflops, 'kparams': kparams,
                 'cur_valid_perf': list(cur_valid_perfs.values())[0],
                 'best_valid_perf': list(best_valid_perfs.values())[0]
                 }
        best_changed = self.performance['best_valid_perfs_changed']
        if best_changed:
            self._save_checkpoint(perfs)

    def after_train(self, logs=None):
        """Be called after the whole train process."""
        # Extract performance logs. This can be moved into builtin callback
        # if we can unify the performace content
        gflops = self.performance['gflops']
        kparams = self.performance['kparams']
        cur_valid_perfs = self.performance['cur_valid_perfs']
        best_valid_perfs = self.performance['best_valid_perfs']
        perfs = {'gflops': gflops, 'kparams': kparams,
                 'cur_valid_perf': list(cur_valid_perfs.values())[0],
                 'best_valid_perf': list(best_valid_perfs.values())[0]
                 }
        self._save_checkpoint(perfs, "latest.pth")
        self._save_performance(perfs, self.trainer.model.desc)
        if self.cfg.get('save_model_desc', False):
            self._save_model_desc()

    def _save_model_desc(self):
        """Save model desc."""
        search_space = SearchSpace()
        codec = Codec(self.cfg.codec, search_space)
        pareto_front_df = pd.read_csv(FileOps.join_path(
            self.result_path, "pareto_front.csv"))
        codes = pareto_front_df['Code']
        for i in range(len(codes)):
            search_desc = Config()
            search_desc.custom = deepcopy(search_space.search_space.custom)
            search_desc.modules = deepcopy(search_space.search_space.modules)
            code = codes.loc[i]
            search_desc.custom.code = code
            search_desc.custom.method = 'full'
            codec.decode(search_desc.custom)
            self.trainer.output_model_desc(i, search_desc)

    def _save_performance(self, performance, model_desc=None):
        """Save result of the model, and calculate pareto front.

        :param performance: The dict that contains all the result needed
        :param model_desc: config of the model
        """
        performance_str = json.dumps(performance, indent=4, sort_keys=True)
        self.trainer._save_performance(performance_str)
        method = model_desc.method
        code = model_desc.code
        metric_method = self.cfg.metric.method
        FileOps.make_dir(FileOps.join_path(self.result_path))
        result_file_name = FileOps.join_path(
            self.result_path, "{}.csv".format(method))
        header = "Code,GFlops,KParams,{0},Best {0},Worker_id\n".format(
            metric_method)
        if not os.path.exists(result_file_name):
            with open(result_file_name, 'w') as file:
                file.write(header)
        with open(result_file_name, 'a') as file:
            file.write('{},{},{},{},{},{}\n'.format(
                code, performance['gflops'], performance['kparams'],
                performance["cur_valid_perf"],
                performance["best_valid_perf"],
                self.trainer.worker_id
            ))
        logging.info("Model result saved to {}".format(result_file_name))
        self._save_pareto_front("GFlops", "Best {}".format(metric_method))

    def _save_pareto_front(self, metric_x, metric_y):
        """Save pareto front of the searched models.

        :param metric_x: x axis of pareto front
        :param metric_y: y axis of pareto front
        """
        df_all = pd.read_csv(FileOps.join_path(self.result_path, "random.csv"))
        mutate_csv = FileOps.join_path(self.result_path, 'mutate.csv')
        if os.path.exists(mutate_csv):
            df_mutate = pd.read_csv(mutate_csv)
            df_all = pd.concat([df_all, df_mutate], ignore_index=True)
        current_best = 0
        df_result = pd.DataFrame(columns=df_all.columns)
        df_all = df_all.sort_values(by=metric_x)
        for index, row in df_all.iterrows():
            if row[metric_y] > current_best:
                current_best = row[metric_y]
                df_result.loc[len(df_result)] = row
        result_file_name = FileOps.join_path(
            self.result_path, "pareto_front.csv")
        df_result.to_csv(result_file_name, index=False)
        logging.info("Pareto front updated to {}".format(result_file_name))

    def _save_checkpoint(self, performance=None, model_name="best.pth"):
        """Save the trained model.

        :param performance: dict of all the result needed
        :param model_name: name of the result file
        :return: the path of the saved file
        """
        model_save_path = FileOps.join_path(self.trainer.get_local_worker_path(), model_name)
        torch.save({
            'model_state_dict': self.trainer.model.state_dict(),
            **performance
        }, model_save_path)

        torch.save(self.trainer.model.state_dict(), model_save_path)
        logging.info("model saved to {}".format(model_save_path))
        return model_save_path
