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
import csv


@ClassFactory.register()
class Kirin990_npu(object):
    """Kirin990_npu class."""

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
        om_save_path = kwargs["save_dir"]
        input_shape = kwargs["input_shape"]
        out_nodes = kwargs["out_nodes"]
        log_save_path = os.path.dirname(model)
        command_line = ["bash", self.current_path + "/model_convert.sh", backend,
                        model, weight, om_save_path, log_save_path, input_shape, out_nodes]
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
        self.mobile_dir = "/sdcard/evaluate_service/" + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        output_data_path = os.path.dirname(input_data)
        if not os.path.isfile(converted_model):
            converted_model = os.path.join(converted_model, "kirin990_npu.om")

        command_line = ["bash", self.current_path + "/inference_kirin990_npu.sh",
                        converted_model, input_data, self.mobile_dir, output_data_path]
        try:
            subprocess.check_output(command_line)
        except subprocess.CalledProcessError as exc:
            logging.error("inference failed. the return message is : {}.".format(exc))

        result_file = os.path.join(output_data_path, "result.csv")

        latency = self._get_latency(result_file)
        return latency, 0

    def _get_latency(self, result_file):
        """Get latency from the result file."""
        logging.info("The result file is {}.".format(result_file))

        time_start = []
        time_end = []
        with open(result_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if "inference begin" in row[0]:
                    time_start.append(float(row[2]))
                if "inference end" in row[0]:
                    time_end.append(float(row[2]))

        latency = (sum(time_end) - sum(time_start)) / len(time_end) / 1000
        return latency
