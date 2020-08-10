# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The EvaluateService of client."""
import os
import numpy as np
from requests import get, post
import logging
from .pytorch2onnx import pytorch2onnx
import subprocess
import pickle


def evaluate(framework, backend, remote_host, model, weight, test_data, input_shape=None):
    """Evaluate interface of the EvaluateService.

    :param evaluate_config: a dict inlcuding framework, backend and remote_host
    :type evaluate_config: dict
    :param model: model file, .pb file for tensorflow and .prototxt for caffe, and a model class for Pytorch
    :type model: str or Class
    :param weight: .caffemodel file for caffe
    :type weight: str
    :param test_data: binary file, .data or .bin
    :type test_data: str
    :return: the latency in Davinci or Bolt
    :rtype: float
    """
    if framework not in ["Tensorflow", "Caffe", "Pytorch"]:
        raise ValueError("The backend only support Tensorflow, Caffe and Pytorch.")

    if backend not in ["Davinci", "Bolt"]:
        raise ValueError("The backend only support Davinci and Bolt.")
    else:
        if framework == "Pytorch":
            if input_shape is None:
                raise ValueError("To convert the pytorch model to onnx model, the input shape must be provided.")
            elif backend == "Bolt":
                model = pytorch2onnx(model, input_shape)
            else:
                with open("torch_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                with open("input_shape.pkl", "wb") as f:
                    pickle.dump(input_shape, f)
                env = os.environ.copy()
                command_line = ["bash", "../vega/core/evaluator/tools/pytorch2caffe.sh",
                                "torch_model.pkl", "input_shape.pkl"]
                try:
                    ret = subprocess.check_output(command_line, env=env)
                except subprocess.CalledProcessError as exc:
                    logging.error("convert torch model to caffe model failed.\
                                  the return code is: {}." .format(exc.returncode))
                model = "torch2caffe.prototxt"
                weight = "torch2caffe.caffemodel"
                framework = "Caffe"

        model_file = open(model, "rb")
        weight_file = open(weight, "rb") if framework == "Caffe" else None
        data_file = open(test_data, "rb")
        upload_data = {"model_file": model_file, "weight_file": weight_file, "data_file": data_file}
        evaluate_config = {"framework": framework, "backend": backend, "remote_host": remote_host}
        post(remote_host, files=upload_data, data=evaluate_config, proxies={"http": None}).json()
        evaluate_result = get(remote_host, proxies={"http": None}).json()
        if evaluate_result.get("status_code") != 200:
            logging.error("Evaluate failed! The return code is {}, the timestmap is {}."
                          .format(evaluate_result["status_code"], evaluate_result["timestamp"]))
        else:
            logging.info("Evaluate sucess! The latency is {}.".format(evaluate_result["latency"]))
    return evaluate_result
