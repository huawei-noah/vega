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

abs_path = os.path.abspath(__file__)
third_party_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(abs_path))))
PytorchToCaffe_path = os.path.join(third_party_path, "third_party/PytorchToCaffe-master/")
sys.path.append(PytorchToCaffe_path)


def pytorch2caffe(model, input_shape, save_dir):
    """Convert the pytorch model to onnx model.

    :param model: pytorch model class
    :type model: class
    :param input_shape: the shape of input
    :type input_shape: list
    :param onnx_save_path: the path and filename to save the onnx model file
    :type onnx_save_path: str
    """
    import pytorch_to_caffe  # noqa
    name = 'torch2caffe'
    model = model.cpu()
    model.eval()
    input = Variable(torch.ones(input_shape))
    pytorch_to_caffe.trans_net(model, input, name)
    prototxt_file = os.path.join(save_dir, "torch2caffe.prototxt")
    caffemodel_file = os.path.join(save_dir, "torch2caffe.caffemodel")
    pytorch_to_caffe.save_prototxt(prototxt_file)
    pytorch_to_caffe.save_caffemodel(caffemodel_file)
    logging.info("pytorch2caffe finished.")


if __name__ == "__main__":
    model_file = sys.argv[1]
    shape_file = sys.argv[2]
    save_dir = os.path.dirname(model_file)
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(shape_file, "rb") as f:
        input_shape = pickle.load(f)
    pytorch2caffe(model, input_shape, save_dir)
