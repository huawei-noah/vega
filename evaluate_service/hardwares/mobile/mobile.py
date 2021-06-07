# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The hardware of mobile."""
import subprocess
import logging
import os
from evaluate_service.class_factory import ClassFactory
import datetime
import numpy as np


@ClassFactory.register()
class Mobile(object):
    """Mobile class."""

    def __init__(self, optional_params):
        self.current_path = os.path.dirname(os.path.abspath(__file__))

    def convert_model(self, backend, model, weight, **kwargs):
        """Convert the tf/caffe/mindspore model to botl model in mobile.

        :param backend: the backend can be one of "tensorflow", "caffe" and "mindspore"
        :type backend: str
        :param model: the model file need to convert
        :type model: str
        :param weight: the weight file need to convert
        :type weight: str
        """
        self.mobile_dir = "/sdcard/evaluate_service/" + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        model_name = os.path.basename(model).split(".")[0]
        log_save_path = os.path.dirname(model)
        precision = "FP32"
        command_line = ["bash", self.current_path + "/model_convert.sh", backend, self.mobile_dir,
                        model, weight, model_name, precision, log_save_path]
        try:
            subprocess.check_output(command_line)
        except subprocess.CalledProcessError as exc:
            logging.error("convert model to bolt failed. The return message is : {}.".format(exc))

    def inference(self, converted_model, input_data, **kwargs):
        """Inference in Davinci.

        :param converted_model: converted model file
        :type backend: str
        :param input_data: the input data file
        :type model: str
        """
        output_data_path = os.path.dirname(input_data)
        command_line = ["bash", self.current_path + "/inference_bolt.sh",
                        converted_model, input_data, self.mobile_dir, output_data_path]
        try:
            subprocess.check_output(command_line)
        except subprocess.CalledProcessError as exc:
            logging.error("inference failed. the return message is : {}.".format(exc))

        result_file = os.path.join(output_data_path, "BoltResult.txt")

        latency = self._get_latency(os.path.join(output_data_path, "ome.log"))
        output = self._get_output(result_file)
        return latency, output

    def _get_latency(self, log_file):
        """Get latency from the log file."""
        logging.info("The log file is {}.".format(log_file))
        command_line = ["bash", self.current_path + "/get_latency_from_log.sh", log_file]
        try:
            latency = subprocess.check_output(command_line)
            return str(latency, 'utf-8').split("\n")[0]
        except subprocess.CalledProcessError as exc:
            logging.error("get_latency_from_log failed. the return message is : {}.".format(exc))

    def _get_output(self, result_file):
        """Get output data of bolt."""
        with open(result_file, 'r') as f:
            values = f.readlines()
        output = []
        for index, value in enumerate(values):
            if index == 0:
                shapes = value.strip().split(",")
                shapes = [int(i) for i in shapes]
            else:
                output.append(np.float(value))
        output = np.array(output).reshape(shapes).tolist()
        return output
