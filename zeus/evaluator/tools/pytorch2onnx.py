# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The tools to convert the pytorch model to onnx model."""
from torch.autograd import Variable
import torch
import subprocess
import logging
import os


def pytorch2onnx(model, input_shape):
    """Convert the pytorch model to onnx model.

    :param model: pytorch model class
    :type model: class
    :param input_shape: the shape of input
    :type input_shape: list
    :param onnx_save_path: the path and filename to save the onnx model file
    :type onnx_save_path: str
    """
    # model.load_state_dict(torch.load(weight))
    # Export the trained model to ONNX
    dump_input = Variable(torch.randn(input_shape))
    if os.path.exists("./torch_model.onnx"):
        os.remove("./torch_model.onnx")
    if os.path.exists("./torch_model_sim.onnx"):
        os.remove("./torch_model_sim.onnx")
    torch.onnx.export(model, dump_input, "./torch_model.onnx")
    try:
        subprocess.call("python3 -m onnxsim ./torch_model.onnx ./torch_model_sim.onnx", shell=True)
    except Exception as e:
        logging.error("{}".format(str(e)))
    onnx_model = "./torch_model_sim.onnx"

    return onnx_model
