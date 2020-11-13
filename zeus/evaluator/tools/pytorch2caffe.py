# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The tools to convert the pytorch model to onnx model."""
import sys
import torch
from torch.autograd import Variable
import logging
import pickle
import os

sys.path.append('../../third_party/PytorchToCaffe-master/')
import pytorch_to_caffe  # noqa


def pytorch2caffe(model, input_shape):
    """Convert the pytorch model to onnx model.

    :param model: pytorch model class
    :type model: class
    :param input_shape: the shape of input
    :type input_shape: list
    :param onnx_save_path: the path and filename to save the onnx model file
    :type onnx_save_path: str
    """
    name = 'torch2caffe'
    model.eval()
    input = Variable(torch.ones(input_shape))
    pytorch_to_caffe.trans_net(model, input, name)
    if os.path.exists("./torch2caffe.prototxt"):
        os.remove("./torch2caffe.prototxt")
    if os.path.exists("./torch2caffe.caffemodel"):
        os.remove("./torch2caffe.caffemodel")
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
    logging.info("pytorch2caffe finished.")

    return '{}.prototxt'.format(name), '{}.caffemodel'.format(name)


if __name__ == "__main__":
    model_file = sys.argv[1]
    shape_file = sys.argv[2]
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(shape_file, "rb") as f:
        input_shape = pickle.load(f)
    pytorch2caffe(model, input_shape)
