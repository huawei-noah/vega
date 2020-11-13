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
from requests import get, post
import logging
from .pytorch2onnx import pytorch2onnx
import subprocess
import pickle


def evaluate(backend, hardware, remote_host, model, weight, test_data, input_shape=None):
    """Evaluate interface of the EvaluateService.

    :param backend: the backend can be one of "tensorflow", "caffe" and "pytorch"
    :type backend: str
    :param hardware: the backend can be one of "Davinci", "Bolt"
    :type hardware: str
    :param remote_host: the remote host ip and port of evaluate service
    :type remote_host: str
    :param model: model file, .pb file for tensorflow and .prototxt for caffe, and a model class for Pytorch
    :type model: str or Class
    :param weight: .caffemodel file for caffe
    :type weight: str
    :param test_data: binary file, .data or .bin
    :type test_data: str
    :return: the latency in Davinci or Bolt
    :rtype: float
    """
    if backend not in ["tensorflow", "caffe", "pytorch"]:
        raise ValueError("The backend only support tensorflow, caffe and pytorch.")

    if hardware not in ["Davinci", "Bolt"]:
        raise ValueError("The hardware only support Davinci and Bolt.")
    else:
        if backend == "pytorch":
            if input_shape is None:
                raise ValueError("To convert the pytorch model to onnx model, the input shape must be provided.")
            elif hardware == "Bolt":
                model = pytorch2onnx(model, input_shape)
            else:
                with open("torch_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                with open("input_shape.pkl", "wb") as f:
                    pickle.dump(input_shape, f)
                env = os.environ.copy()
                command_line = ["bash", "../../zeus/evaluator/tools/pytorch2caffe.sh",
                                "torch_model.pkl", "input_shape.pkl"]
                try:
                    subprocess.check_output(command_line, env=env)
                except subprocess.CalledProcessError as exc:
                    logging.error("convert torch model to caffe model failed.\
                                  the return code is: {}.".format(exc.returncode))
                model = "torch2caffe.prototxt"
                weight = "torch2caffe.caffemodel"
                backend = "caffe"
        elif backend == "tensorflow":
            pb_model_file = "./model.pb"
            if os.path.exists(pb_model_file):
                os.remove(pb_model_file)
            freeze_graph(model, weight, pb_model_file)

        model_file = open(model, "rb")
        weight_file = open(weight, "rb") if backend == "caffe" else None
        data_file = open(test_data, "rb")
        upload_data = {"model_file": model_file, "weight_file": weight_file, "data_file": data_file}
        evaluate_config = {"backend": backend, "hardware": hardware, "remote_host": remote_host}
        post(remote_host, files=upload_data, data=evaluate_config, proxies={"http": None}).json()
        evaluate_result = get(remote_host, proxies={"http": None}).json()
        if evaluate_result.get("status_code") != 200:
            logging.error("Evaluate failed! The return code is {}, the timestmap is {}."
                          .format(evaluate_result["status_code"], evaluate_result["timestamp"]))
        else:
            logging.info("Evaluate sucess! The latency is {}.".format(evaluate_result["latency"]))
    return evaluate_result


def freeze_graph(model, input_checkpoint, output_graph_file):
    """Freeze the tensorflow graph.

    :param model: the tensorflow model
    :type model: str
    :param model: checkpoint file of tensorflow
    :type model: str
    :param output_graph_file: the file to save the freeze graph model
    :type output_graph_file: str
    """
    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    input_node = tf.placeholder(tf.float32, shape=(1, 3, None, None))
    output_node = model(input_node, training=False)
    output_node_names = [output_node.name.split(":")[0]]
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names)

        with tf.gfile.GFile(output_graph_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        logging.info("%d ops in the final graph." % len(output_graph_def.node))
