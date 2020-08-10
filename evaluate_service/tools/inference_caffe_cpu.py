# -*- coding: utf-8 -*-
"""The tools to inference caffe model in cpu to get the benchmark output."""
import caffe
from data_convert import binary2np


def inference_caffe_cpu(model_file, weight_file, input_data_path, input_shape, input_dtype, output_node_name):
    """Inference of the tensorflow in cpu to get the benchmark output.

    :param model_file: the caffe model, .pbtotxt file
    :type model_file: str
    :param weight_file: the caffe weight, .caffemodel
    :type weight_file: str
    :param input_data_path: input data file, the .bin or .data format
    :type input_data_path: str
    :param input_shape: the shape of the input data
    :type input_shape: list
    :param input_dtype: the dtype of the input data
    :type input_dtype: str
    :param output_node_name: the output_node name in the graph
    :type output_node_name: str
    """
    net = caffe.Net(model_file, weight_file, caffe.TEST)
    input_data = binary2np(input_data_path, input_shape, input_dtype)
    net.blobs['data'].data[...] = input_data
    net.forward()
    res = net.blobs[output_node_name].data[0]

    res.tofile("expect_out.data")
