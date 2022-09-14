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

"""The EvaluateService of client."""
import logging
import os
import numpy as np
from .rest import post


def evaluate(backend, hardware, remote_host, model, weight, test_data, input_shape=None, reuse_model=False, job_id=None,
             quantize=False, repeat_times=10, precision='FP32', cal_metric=False, muti_input=False, **kwargs):
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
    _check_backend_hardware_shape(backend, hardware, input_shape)

    if not reuse_model:
        base_save_dir = os.path.dirname(test_data)
        model, weight, backend = preprocessing_model(backend, hardware, model, weight, input_shape,
                                                     base_save_dir, quantize, test_data, **kwargs)
        model_file = open(model, "rb")
        data_file = open(test_data, "rb")
        if backend == "caffe":
            weight_file = open(weight, "rb")
            upload_data = {"model_file": model_file, "weight_file": weight_file, "data_file": data_file}
        else:
            upload_data = {"model_file": model_file, "data_file": data_file}
    else:
        data_file = open(test_data, "rb")
        upload_data = {"data_file": data_file}

    evaluate_config = {"backend": backend, "hardware": hardware, "remote_host": remote_host, "reuse_model": reuse_model,
                       "job_id": job_id, "repeat_times": repeat_times, "precision": precision, "cal_metric": cal_metric,
                       'muti_input': muti_input}
    if backend == 'tensorflow':
        shape_list = [str(s) for s in input_shape]
        shape_cfg = {"input_shape": "Placeholder:" + ",".join(shape_list)}
        evaluate_config.update(shape_cfg)
    if backend == "tensorflow" and hardware == "Kirin990_npu":
        out_node_name = _get_pb_out_nodes(model)
        out_node_cfg = {"out_nodes": out_node_name}
        evaluate_config.update(out_node_cfg)

    evaluate_result = _post_request(remote_host, upload_data, test_data, evaluate_config)

    data_file.close()
    if "model_file" in locals():
        model_file.close()
    if "weight_file" in locals():
        weight_file.close()

    if not kwargs.get("save_intermediate_file", False):
        if os.path.exists(model):
            os.remove(model)
        if weight and os.path.isfile(weight) and os.path.exists(weight):
            os.remove(weight)
        if os.path.exists(test_data):
            os.remove(test_data)

    return evaluate_result


def _check_backend_hardware_shape(backend, hardware, input_shape):
    if backend not in ["tensorflow", "caffe", "pytorch", "mindspore"]:
        raise ValueError("The backend only support tensorflow, caffe, pytorch and mindspore.")

    if hardware not in ["Davinci", "Bolt", "Kirin990_npu"]:
        raise ValueError("The hardware only support Davinci and Bolt.")

    if input_shape is None:
        raise ValueError("The input shape must be provided.")


def _post_request(remote_host, upload_data, test_data, evaluate_config):
    evaluate_result = post(host=remote_host, files=upload_data, data=evaluate_config)
    if evaluate_result.get("status") != "success":
        logging.warning(
            "Evaluate failed and will try again, the status is {}, the timestamp is {}, \
            the error message is {}.".format(
                evaluate_result.get("status"), evaluate_result.get("timestamp"), evaluate_result.get("error_message")))
        evaluate_config["reuse_model"] = True
        upload_data = {"data_file": open(test_data, "rb")}
        retry_times = 4
        for i in range(retry_times):
            evaluate_result = post(host=remote_host, files=upload_data, data=evaluate_config)
            if evaluate_result.get("status") == "success":
                logging.info("Evaluate success! The latency is {}.".format(evaluate_result["latency"]))
                break
            else:
                if i == 3:
                    logging.error(
                        "Evaluate failed, the status is {},the timestamp is {}, the retry times is {}, the error \
                        message is {}.".format(evaluate_result.get("status"), evaluate_result.get("timestamp"),
                                               i + 1, evaluate_result.get("error_message")))
                else:
                    logging.warning(
                        "Evaluate failed, the status is {},the timestamp is {}, the retry times is {}, the error \
                        message is {}.".format(evaluate_result.get("status"), evaluate_result.get("timestamp"), i + 1,
                                               evaluate_result.get("error_message")))
    else:
        logging.info("Evaluate success! The latency is {}.".format(evaluate_result["latency"]))

    return evaluate_result


def preprocessing_model(backend, hardware, model, weight, input_shape, base_save_dir, quantize, test_data, **kwargs):
    """Preprocess the model.

    :param backend: the backend can be one of "tensorflow", "caffe" , "pytorch" and "mindspore".
    :type backend: str
    :param hardware: the backend can be one of "Davinci", "Bolt"
    :type hardware: str
    :param model: model file, .pb file for tensorflow and .prototxt for caffe, and a model class for Pytorch
    :type model: str or Class
    :param weight: .caffemodel file for caffe
    :type weight: str
    :param input_shape: the shape of input data
    :type input_shape: list
    :param base_save_dir: the save dir of the preprocessed model and weight
    :type base_save_dir: str
    """
    if backend == "pytorch":
        if kwargs.get("custom", None) is not None:
            model = kwargs.get("custom").export_model(model)
        else:
            from .pytorch2onnx import pytorch2onnx
            opset_version = kwargs["opset_version"]
            model = pytorch2onnx(model, input_shape, base_save_dir, opset_version)
        backend = "onnx"
    elif backend == "tensorflow":
        pb_model_file = os.path.join(base_save_dir, "tf_model.pb")
        if os.path.exists(pb_model_file):
            os.remove(pb_model_file)

        freeze_graph(model, weight, pb_model_file, input_shape, quantize, test_data)
        model = pb_model_file
    elif backend == "mindspore":
        if kwargs.get("custom", None) is not None:
            model = kwargs.get("custom").export_model(model)
        else:
            from mindspore.train.serialization import export
            from mindspore import Tensor
            fake_input = np.random.random(input_shape).astype(np.float32)
            save_name = os.path.join(base_save_dir, "ms2air.air")
            export(model, Tensor(fake_input), file_name=save_name, file_format='AIR')
            model = save_name
    return model, weight, backend


def freeze_graph(model, weight_file, output_graph_file, input_shape, quantize, test_data):
    """Freeze the tensorflow graph.

    :param model: the tensorflow model
    :type model: str
    :param output_graph_file: the file to save the freeze graph model
    :type output_graph_file: str
    """
    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    with tf.Graph().as_default():
        input_holder_shape = (None,) + tuple(input_shape[1:])
        input_holder = tf.placeholder(dtype=tf.float32, shape=input_holder_shape)
        model.training = False
        output = model(input_holder)
        if isinstance(output, tuple):
            output_name = [output[0].name.split(":")[0]]
        else:
            output_name = [output.name.split(":")[0]]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if weight_file is not None:
                saver = tf.train.Saver()
                last_weight_file = tf.train.latest_checkpoint(weight_file)
                if last_weight_file:
                    saver.restore(sess, last_weight_file)
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_name)

            with tf.gfile.FastGFile(output_graph_file, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
    if quantize:
        from .quantize_model import quantize_model
        quantize_model(output_graph_file, test_data, input_holder, output)


def _get_pb_out_nodes(pb_file):
    """Get the out nodes of pb model.

    :param pb_file: the pb model file
    :type pb_file: str
    """
    import tensorflow as tf
    new_graph = tf.Graph()
    with new_graph.as_default():
        with tf.gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        tensor_name_list = [tensor.name for tensor in new_graph.as_graph_def().node]
    out_node = tensor_name_list[-1]
    out_node_name = str(out_node) + ":0"
    return out_node_name
