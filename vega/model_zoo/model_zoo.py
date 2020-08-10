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
from vega.model_zoo.torch_vision_model import get_torchvision_model_file
from vega.search_space.networks import NetworkDesc, NetTypes
from vega.core.common import TaskOps
from vega.core.common.general import General


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
    def get_model(cls, model_desc=None, model_checkpoint=None):
        """Get model from model zoo.

        :param network_name: the name of network, eg. ResNetVariant.
        :type network_name: str or None.
        :param network_desc: the description of network.
        :type network_desc: str or None.
        :param model_checkpoint: path of model.
        :type model_checkpoint: str.
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
        logging.debug("model_desc={}".format(model_desc))
        if model_checkpoint is not None:
            logging.info("Load model with weight.")
            model = cls._load_pretrained_model(network, model, model_checkpoint)
            logging.info("Model was loaded.")
        return model

    @classmethod
    def _load_pretrained_model(cls, network, model, model_checkpoint):
        if not model_checkpoint and network._model_type == NetTypes.TORCH_VISION_MODEL:
            model_file_name = get_torchvision_model_file(network._model_name)
            full_path = "{}/torchvision_models/checkpoints/{}".format(
                TaskOps().model_zoo_path, model_file_name)
        else:
            full_path = model_checkpoint
        logging.info("load model weights from file.")
        logging.debug("Weights file: {}".format(full_path))
        if not os.path.isfile(full_path):
            raise "Pretrained model is not existed, model={}".format(full_path)
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint)
        return model

    @classmethod
    def infer(cls, model, dataloader):
        """Infer the result."""
        model.eval()
        infer_result = []
        with torch.no_grad():
            model.cuda()
            for _, input in enumerate(dataloader):
                if isinstance(input, list):
                    input = input[0]
                logits = model(input.cuda())
                if isinstance(logits, tuple):
                    logits = logits[0]
                infer_result.extend(logits)
            return infer_result
