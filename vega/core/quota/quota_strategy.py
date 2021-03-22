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
import zeus
from zeus.common.general import General
from zeus.report import ReportServer
from zeus.common.task_ops import TaskOps
from vega.core.pipeline.conf import PipelineConfig, PipeStepConfig
from vega.core.pipeline.pipe_step import PipeStep


class QuotaStrategy(object):
    """Config parameters adjustment according to runtime setting."""

    def __init__(self):
        self.restrict_config = General.quota.restrict
        self.affinity_config = General.quota.affinity
        self.max_runtime = General.quota.strategy.runtime
        self.only_search = General.quota.strategy.only_search
        self.epoch_time = 0.
        self.params_dict = {}
        self.temp_trials = copy.deepcopy(self.restrict_config.trials)
        self._backup_quota_config()

    def adjust_pipeline_config(self, cfg):
        """Adjust pipeline config according."""
        cfg_cp = copy.deepcopy(cfg)
        cfg_tiny = copy.deepcopy(cfg)
        workers_num = self._calc_workers_num()
        General.parallel_search = False
        self._get_time_params(cfg_cp)
        self._simulate_tiny_pipeline(cfg_tiny)
        General.parallel_search = cfg.general.parallel_search
        self._modify_pipeline_config(workers_num, self.epoch_time, self.params_dict)
        if zeus.is_npu_device():
            os.environ['RANK_TABLE_FILE'] = os.environ['ORIGIN_RANK_TABLE_FILE']
            os.environ['RANK_SIZE'] = os.environ['ORIGIN_RANK_SIZE']
        logging.info('Adjust runtime config successfully.')

    def _simulate_tiny_pipeline(self, cfg_tiny):
        """Simulate tiny pipeline by using one sample one epoch."""
        report = ReportServer()
        for i, step_name in enumerate(PipelineConfig.steps):
            step_cfg = cfg_tiny.get(step_name)
            if step_cfg.pipe_step.type != 'SearchPipeStep':
                continue
            step_cfg.trainer.distributed = False
            step_cfg.trainer.epochs = 1
            self.restrict_config.trials[step_name] = 1
            General.step_name = step_name
            PipeStepConfig.from_dict(step_cfg)
            pipestep = PipeStep()
            if i == 0:
                pipestep.do()
                record = report.get_step_records(step_name)[-1]
                self.epoch_time = record.runtime
                _worker_path = TaskOps().local_base_path
                if os.path.exists(_worker_path):
                    os.system('rm -rf {}'.format(_worker_path))
            if step_cfg.pipe_step.type == 'SearchPipeStep':
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
            if not cfg[step_name].get('trainer', None):
                continue
            params['epochs'] = cfg[step_name].trainer.epochs
            self.params_dict[step_name] = params

    def _modify_pipeline_config(self, workers_num, epoch_time, params_dict):
        """Modify pipeline config according to simulated results."""
        self._restore_quota_config()
        nas_time_dict, ft_time_dict = dict(), dict()
        for step_name in params_dict:
            step_time = epoch_time * params_dict[step_name]['epochs']
            if 'max_samples' in params_dict[step_name]:
                step_time = step_time * params_dict[step_name]['max_samples'] / workers_num
                nas_time_dict[step_name] = step_time
            else:
                ft_time_dict[step_name] = step_time
        nas_total_time = sum([value for key, value in nas_time_dict.items()])
        if nas_total_time == 0:
            return
        ft_total_time = sum([value for key, value in ft_time_dict.items()])
        left_time = self.max_runtime
        if not self.only_search:
            if ft_total_time > 0.9 * self.max_runtime:
                ft_total_time = 0.9 * self.max_runtime
            left_time = self.max_runtime - ft_total_time
        scale = left_time / nas_total_time
        for key, value in nas_time_dict.items():
            self.restrict_config.duration[key] = float(scale * value)
        self.restrict_config.trials = copy.deepcopy(self.temp_trials)
        logging.info('Max duration modified as {}'.format(self.restrict_config.duration))

    def _backup_quota_config(self):
        self.temp_trials = copy.deepcopy(self.restrict_config.trials)
        self.temp_flops, self.temp_params, self.temp_latency = \
            self.restrict_config.flops, self.restrict_config.params, self.restrict_config.latency
        self.restrict_config.flops, self.restrict_config.params, self.restrict_config.latency = None, None, None
        self.temp_affinity_type = self.affinity_config.type
        self.affinity_config.type = None

    def _restore_quota_config(self):
        self.restrict_config.trials = self.temp_trials
        self.restrict_config.flops, self.restrict_config.params, self.restrict_config.latency = \
            self.temp_flops, self.temp_params, self.temp_latency
        self.affinity_config.type = self.temp_affinity_type

    def _calc_workers_num(self):
        """Calculate workers numbers."""
        if not General.parallel_search:
            return 1
        if zeus.is_gpu_device():
            import torch
            world_size = General.env.world_size
            devices_per_node = torch.cuda.device_count()
            worker_num = (world_size * devices_per_node) // General.devices_per_trainer
        elif zeus.is_npu_device():
            world_devices = int(os.environ['RANK_SIZE'])
            worker_num = world_devices // General.devices_per_trainer
        return worker_num
