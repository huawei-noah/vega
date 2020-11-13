# -*- coding: utf-8 -*-
"""The tools to inference tensorflow model in cpu to get the benchmark output."""
import tensorflow as tf
from tensorflow.python.platform import gfile
from data_convert import binary2np


def inference_tf_cpu(pb_file, input_data_path, input_shape, input_dtype, input_node_name, output_node_name):
    """Inference of the tensorflow in cpu to get the benchmark output.

    :param pb_file: the tensorflow model, .pb file
    :type pb_file: str
    :param input_data_path: input data file, the .bin or .data format
    :type input_data_path: str
    :param input_shape: the shape of the input data
    :type input_shape: list
    :param input_dtype: the dtype of the input data
    :type input_dtype: str
    :param input_node_name: the input_node_name in the graph
    :type input_node_name: str
    :param output_node_name: the output_node name in the graph
    :type output_node_name: str
    """
    sess = tf.compat.v1.Session()
    with gfile.FastGFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        # import the graph
        tf.import_graph_def(graph_def, name='')

    # Intialize
    sess.run(tf.global_variables_initializer())

    input_data = binary2np(input_data_path, input_shape, input_dtype)
    op = sess.graph.get_tensor_by_name(output_node_name)
    input_data_graph = sess.graph.get_tensor_by_name(input_node_name)
    res = sess.run(op, feed_dict={input_data_graph: input_data})

    res.tofile("expect_out.data")
