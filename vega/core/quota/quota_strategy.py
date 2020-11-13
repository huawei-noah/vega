# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Sample Filter."""
import os
import logging
import copy
from zeus.common.general import General
from zeus.report import Report
from zeus.common.task_ops import TaskOps
from vega.core.pipeline.conf import PipelineConfig, PipeStepConfig
from vega.core.pipeline.pipe_step import PipeStep


class QuotaStrategy(object):
    """Config parameters adjustment according to runtime setting."""

    def __init__(self):
        self.restrict_config = General.quota.restrict
        self.max_runtime = General.quota.runtime
        self.epoch_time = 0.
        self.params_dict = {}
        self.temp_trials = copy.deepcopy(self.restrict_config.trials)

    def adjust_pipeline_config(self, cfg):
        """Adjust pipeline config according."""
        cfg_cp = copy.deepcopy(cfg)
        cfg_tiny = copy.deepcopy(cfg)
        workers_num = self._calc_workers_num()
        General.parallel_search = False
        self._get_time_params(cfg_cp)
        self._simulate_tiny_pipeline(cfg_tiny)
        General.parallel_search = cfg.general.parallel_search
        cfg_new = self._modify_pipeline_config(cfg_cp, workers_num, self.epoch_time, self.params_dict)
        logging.info('Adjust runtime config successfully.')
        return cfg_new

    def _simulate_tiny_pipeline(self, cfg_tiny):
        """Simulate tiny pipeline by using one sample one epoch."""
        report = Report()
        for i, step_name in enumerate(PipelineConfig.steps):
            step_cfg = cfg_tiny.get(step_name)
            step_cfg.trainer.distributed = False
            step_cfg.trainer.epochs = 1
            self.restrict_config.trials[step_name] = 1
            General.step_name = step_name
            PipeStepConfig.from_json(step_cfg)
            pipestep = PipeStep()
            if i == 0:
                pipestep.do()
                record = report.get_step_records(step_name)[-1]
                self.epoch_time = record.runtime
                _worker_path = TaskOps().local_base_path
                if os.path.exists(_worker_path):
                    os.system('rm -rf {}'.format(_worker_path))
            if step_cfg.pipe_step.type == 'NasPipeStep':
                self.params_dict[step_name]['max_samples'] = pipestep.generator.search_alg.max_samples
            _file = os.path.join(TaskOps().step_path, ".generator")
            if os.path.exists(_file):
                os.system('rm {}'.format(_file))

    def _get_time_params(self, cfg):
        """Get time parameters from config."""
        for step_name in PipelineConfig.steps:
            params = dict()
            step_cfg = cfg.get(step_name)
            pipe_type = step_cfg.pipe_step.type
            params['pipe_type'] = pipe_type
            params['epochs'] = cfg[step_name].trainer.epochs
            self.params_dict[step_name] = params

    def _modify_pipeline_config(self, cfg, workers_num, epoch_time, params_dict):
        """Modify pipeline config according to simulated results."""
        nas_time_dict, ft_time_dict = dict(), dict()
        for step_name in PipelineConfig.steps:
            step_time = epoch_time * params_dict[step_name]['epochs']
            if 'max_samples' in params_dict[step_name]:
                step_time = step_time * params_dict[step_name]['max_samples'] / workers_num
                nas_time_dict[step_name] = step_time
            else:
                ft_time_dict[step_name] = step_time
        ft_total_time = sum([value for key, value in ft_time_dict.items()])
        if ft_total_time > self.max_runtime:
            raise Exception('Fully train runtime must larger than setting max runtime.')
        nas_total_time = sum([value for key, value in nas_time_dict.items()])
        left_time = self.max_runtime - ft_total_time
        if left_time < nas_total_time:
            scale = left_time / nas_total_time
            for key, value in nas_time_dict.items():
                self.restrict_config.trials[key] = int(params_dict[key]['max_samples'] * scale) + 1
                self.restrict_config.duration[key] = float(left_time * value / nas_total_time)
        else:
            self.restrict_config.trials = copy.deepcopy(self.temp_trials)
        return cfg

    def _calc_workers_num(self):
        """Calculate workers numbers."""
        if not General._parallel:
            return 1
        import torch
        world_size = General.env.world_size
        devices_per_node = torch.cuda.device_count()
        worker_num = (world_size * devices_per_node) // General.devices_per_trainer
        return worker_num
