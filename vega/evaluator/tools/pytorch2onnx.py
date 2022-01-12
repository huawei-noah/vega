# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The tools to convert the pytorch model to onnx model."""
from torch.autograd import Variable
import torch


def pytorch2onnx(model, input_shape, base_save_dir, opset_version=9):
    """Convert the pytorch model to onnx model.

    :param model: pytorch model class
    :type model: class
    :param input_shape: the shape of input
    :type input_shape: list
    :param onnx_save_path: the path and filename to save the onnx model file
    :type onnx_save_path: str
    """
    dump_input = Variable(torch.randn(input_shape))
    if hasattr(model, "get_ori_model"):
        model = model.get_ori_model()
    torch.onnx.export(model, dump_input, "{}/torch_model.onnx".format(base_save_dir), opset_version=opset_version)
    onnx_model = f"{base_save_dir}/torch_model.onnx"
    return onnx_model
