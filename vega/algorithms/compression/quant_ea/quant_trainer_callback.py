# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""TrainWorker for searching quantization model."""
import os
import pandas as pd
import torch
import torch.utils.data
from vega.core.trainer.pytorch import Trainer
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import Config
from vega.search_space import SearchSpace
from vega.search_space.codec import Codec
from vega.datasets.pytorch import Dataset
from .quant_codec import QuantCodec
from vega.core.common.file_ops import FileOps
from .utils.quant_model import Quantizer
from .utils.flops_params_counter import cal_model_flops, cal_model_params
from vega.core.visual import dump_trainer_visual_info
from vega.core.trainer.callbacks import Callback


@ClassFactory.register(ClassType.CALLBACK)
class QuantTrainerCallback(Callback):
    """Callback class for Quant Trainer."""

    def before_train(self, logs=None):
        """Be called before the train process."""
        self.cfg = self.trainer.cfg
        self.trainer.auto_save_ckpt = False
        self.trainer.auto_save_perf = False
        self.save_weight = self.cfg.save_weight
        self.device = self.cfg.device
        self.base_net_desc = SearchSpace().cfg
        self.model_code = self.base_net_desc.backbone.get(
            'model_code', None)
        self.model = self.trainer.model
        if self.model is None:
            self.model = self.trainer._init_model()
        self.model._init_weights()
        self.model = self._quantize_model(self.model).to(self.device)
        self.flops_count, _ = cal_model_flops(self.model,
                                              self.device,
                                              input_size=(1, 3, 32, 32))
        self.params_count, _ = cal_model_params(self.model,
                                                self.device,
                                                input_size=(1, 3, 32, 32))
        if not self.validate():
            return
        self.trainer.build(model=self.model)

    def after_epoch(self, epoch, logs=None):
        """Be called after one epoch training."""
        self.summary_perfs = logs.get('summary_perfs', None)
        if self.summary_perfs['best_valid_perfs_changed']:
            self._save_best_model()

    def after_train(self, logs=None):
        """Be called after the whole train process."""
        self.metric = list(self.summary_perfs['best_valid_perfs'].values())[0][0]
        self.save_metrics_value()
        if self.cfg.get('save_model_desc', False):
            self._save_model_desc()

    def _save_model_desc(self):
        """Save final model desc of NAS."""
        pf_file = FileOps.join_path(self.trainer.local_output_path, self.trainer.step_name, "pareto_front.csv")
        if not FileOps.exists(pf_file):
            return
        with open(pf_file, "r") as file:
            pf = pd.read_csv(file)
        pareto_fronts = pf["encoding"].tolist()
        search_space = SearchSpace()
        codec = QuantCodec('QuantCodec', search_space)
        for i, pareto_front in enumerate(pareto_fronts):
            pareto_front = [int(x) for x in pareto_front[1:-1].split(',')]
            model_desc = Config()
            model_desc.modules = search_space.search_space.modules
            model_desc.backbone = codec.decode(pareto_front)._desc.backbone
            self.trainer.output_model_desc(i, model_desc)

    def _quantize_model(self, model):
        """Quantize the model.

        :param input: pytorch model
        :type input: nn.Module
        :return: quantized pytorch model
        :rtype: nn.Module
        """
        q = Quantizer()
        if self.model_code is not None:
            length = len(self.model_code)
            nbit_w_list = self.model_code[:length // 2]
            nbit_a_list = self.model_code[length // 2:]
        else:
            nbit_w_list = model.nbit_w_list
            nbit_a_list = model.nbit_a_list
        print('current code:', nbit_w_list, nbit_a_list)
        model = q.quant_model(model, nbit_w_list, nbit_a_list)
        return model

    def save_metrics_value(self):
        """Save the metric value of the trained model.

        :return: save_path (local) and s3_path (remote). If s3_path not specified, then s3_path is None
        :rtype: a tuple of two str
        """
        pd_path = FileOps.join_path(
            self.trainer.local_output_path, self.trainer.step_name, "performace.csv")
        FileOps.make_base_dir(pd_path)
        encoding = self.model.nbit_w_list + self.model.nbit_a_list
        df = pd.DataFrame(
            [[encoding, self.flops_count, self.params_count, self.metric]],
            columns=["encoding", "flops", "parameters", self.cfg.get("valid_metric", "acc")])
        if not os.path.exists(pd_path):
            with open(pd_path, "w") as file:
                df.to_csv(file, index=False)
        else:
            with open(pd_path, "a") as file:
                df.to_csv(file, index=False, header=False)
        if self.trainer.backup_base_path is not None:
            FileOps.copy_folder(self.trainer.local_output_path,
                                self.trainer.backup_base_path)

    def validate(self):
        """Check whether the model fits in the #flops range or #parameter range specified in config.

        :return: true or false, which specifies whether the model fits in the range
        :rtype: bool
        """
        limits_config = self.cfg.get("limits", dict())
        if "flop_range" in limits_config:
            flop_range = limits_config["flop_range"]
            if self.flops_count < flop_range[0] or self.flops_count > flop_range[1]:
                return False
        if "param_range" in limits_config:
            param_range = limits_config["param_range"]
            if self.params_count < param_range[0] or self.params_count > param_range[1]:
                return False
        return True

    def _save_best_model(self):
        save_path = FileOps.join_path(
            self.trainer.get_local_worker_path(), self.trainer.step_name, "best_model.pth")
        FileOps.make_base_dir(save_path)
        torch.save(self.model.state_dict(), save_path)
        if self.trainer.backup_base_path is not None:
            _dst = FileOps.join_path(self.trainer.backup_base_path, "workers",
                                     str(self.trainer.worker_id))
            FileOps.copy_folder(self.trainer.get_local_worker_path(), _dst)
