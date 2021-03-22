# -*- coding: utf-8 -*-
"""The tools to inference onnx model in cpu to get the benchmark output."""
import onnxruntime
from data_convert import binary2np


def inference_onnx_cpu(onnx_file, input_data_path, input_shape, input_dtype):
    """Inference of the onnx in cpu to get the benchmark output.

    :param onnx_file: the onnx file
    :type onnx_file: str
    :param input_data_path: input data file, the .bin or .data format
    :type input_data_path: str
    :param input_shape: the shape of the input
    :type input_shape: list
    :param input_dtype: the dtype of input
    :type input_dtype: str
    """
    input_data = binary2np(input_data_path, input_shape, input_dtype)
    sess = onnxruntime.InferenceSession(onnx_file)
    output_nodes = sess.get_outputs()[0].name
    input_nodes = sess.get_inputs()[0].name
    res = sess.run([output_nodes], {input_nodes: input_data})
    res.tofile("expect_out.data")
