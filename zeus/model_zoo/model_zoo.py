# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Model zoo."""
import zeus
import logging
import os
from zeus.networks.network_desc import NetworkDesc
from zeus.common.general import General


class ModelZoo(object):
    """Model zoo."""

    @classmethod
    def set_location(cls, location):
        """Set model zoo location.

        :param location: model zoo location.
        :type localtion: str.

        """
        General.model_zoo.model_zoo_path = location

    @classmethod
    def get_model(cls, model_desc=None, pretrained_model_file=None):
        """Get model from model zoo.

        :param network_name: the name of network, eg. ResNetVariant.
        :type network_name: str or None.
        :param network_desc: the description of network.
        :type network_desc: str or None.
        :param pretrained_model_file: path of model.
        :type pretrained_model_file: str.
        :return: model.
        :rtype: model.

        """
        try:
            network = NetworkDesc(model_desc)
            model = network.to_model()
        except Exception as e:
            logging.error("Failed to get model, model_desc={}, msg={}".format(
                model_desc, str(e)))
            raise e
        logging.info("Model was created.")
        if zeus.is_torch_backend() and pretrained_model_file:
            model = cls._load_pretrained_model(model, pretrained_model_file)
        elif zeus.is_ms_backend() and pretrained_model_file:
            model = cls._load_pretrained_model(model, pretrained_model_file)
        return model

    @classmethod
    def _load_pretrained_model(cls, model, pretrained_model_file):
        if zeus.is_torch_backend():
            import torch
            if not os.path.isfile(pretrained_model_file):
                raise "Pretrained model is not existed, model={}".format(pretrained_model_file)
            logging.info("load model weights from file, weights file={}".format(pretrained_model_file))
            checkpoint = torch.load(pretrained_model_file)
            model.load_state_dict(checkpoint)
        elif zeus.is_ms_backend():
            from mindspore.train.serialization import load_checkpoint
            load_checkpoint(pretrained_model_file, net=model)
        return model

    @classmethod
    def select_compressed_models(cls, model_zoo_file, standard, num):
        """Select compressed model by model filter."""
        from zeus.model_zoo.compressed_model_filter import CompressedModelFilter
        model_filter = CompressedModelFilter(model_zoo_file)
        model_desc_list = model_filter.select_satisfied_model(standard, num)
        return model_desc_list
