# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Model zoo."""
import torch
import logging
import os
from datetime import datetime
from vega.model_zoo.torch_vision_model import get_torchvision_model_file
from vega.search_space.networks import NetworkDesc, NetTypes
from vega.core import FileOps, TaskOps, DefaultConfig


class ModelZoo(object):
    """Model zoo."""

    @classmethod
    def set_location(cls, location):
        """Set model zoo location.

        :param location: model zoo location.
        :type localtion: str.

        """
        cfg_data = {
            "general": {
                "model_zoo": {
                    "model_zoo_path": location
                }
            }
        }
        DefaultConfig().data = cfg_data

    @classmethod
    def get_model(cls, network_desc=None, pretrained=False):
        """Get model from model zoo.

        :param network_name: the name of network, eg. ResNetVariant.
        :type network_name: str or None.
        :param network_desc: the description of network.
        :type network_desc: str or None.
        :param pretrained: pre-trained model or not.
        :type pretrained: bool.
        :return: model.
        :rtype: model.

        """
        try:
            network = NetworkDesc(network_desc)
            model = network.to_model()
        except Exception as e:
            logging.error("Failed to get model, network_desc={}, msg={}".format(
                network_desc, str(e)))
            raise e
        logging.info("Model was created, model_desc={}".format(network_desc))
        if pretrained is True:
            logging.info("Load pretrained model.")
            model = cls._load_pretrained_model(network, model)
            logging.info("Pretrained model was loaded.")
        return model

    @classmethod
    def _load_pretrained_model(cls, network, model):
        task = TaskOps(DefaultConfig().data.general)
        if network.model_type == NetTypes.TORCH_VISION_MODEL:
            model_file_name = get_torchvision_model_file(network.model_name)
            full_path = "{}/torchvision_models/checkpoints/{}".format(task.model_zoo_path, model_file_name)
        else:
            model_file_name = "{}-{}.pth".format(network.model_name, network.md5)
            full_path = "{}/models/{}".format(task.model_zoo_path, model_file_name)
        logging.info("load model from model zoo, model={}".format(full_path))
        if ":" in full_path:
            temp_file = "{}.{}".format(model_file_name, datetime.now().strftime('%S.%f')[:-3])
            temp_file = FileOps.join_path(task.temp_path, temp_file)
            logging.info("Download model from model zoo, dest={}".format(temp_file))
            FileOps.copy_file(full_path, temp_file)
            logging.info("Model was downloaded.")
            full_path = temp_file
        if not os.path.isfile(full_path):
            raise "Pretrained model is not existed, model={}".format(full_path)
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint)
        return model
