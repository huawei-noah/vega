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
from vega.common.general import General


def pytorch2onnx(model, input_shape, base_save_dir):
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
    torch.onnx.export(model, dump_input, "{}/torch_model.onnx".format(base_save_dir))
    # try:
    #     subprocess.call(
    #         f"{General.python_command} -m onnxsim {base_save_dir}/torch_model.onnx "
    #         f"{base_save_dir}/torch_model_sim.onnx", shell=True)
    # except Exception as e:
    #     logging.error("{}".format(str(e)))
    # onnx_model = f"{base_save_dir}/torch_model_sim.onnx"
    onnx_model = f"{base_save_dir}/torch_model.onnx"
    return onnx_model
