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

"""The hardware of davinci."""
import datetime
import logging
import os
import subprocess

import numpy as np
from evaluate_service.class_factory import ClassFactory


@ClassFactory.register()
class Davinci(object):
    """Davinci class."""
    supported_devices = ["ATLAS200DK", "Ascend310", "Ascend310P1", "Ascend310P2", "Ascend310P3", "Ascend310P4"]

    def __init__(self, optional_params):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.optional_params = optional_params
        self.davinci_environment_type = optional_params.get("davinci_environment_type")

    def convert_model(self, backend, model, weight, **kwargs):
        """Convert the tf/caffe/mindspore/onnx model to om model in Davinci.

        :param backend: the backend can be one of "tensorflow", "caffe", "mindspore" and "onnx"
        :type backend: str
        :param model: the model file need to convert
        :type model: str
        :param weight: the weight file need to convert
        :type weight: str
        """
        om_save_path = kwargs["save_dir"]
        input_shape = kwargs["input_shape"]
        precision = kwargs['precision']
        log_save_path = os.path.dirname(model)

        command_line = ["bash", self.current_path + "/model_convert.sh", self.davinci_environment_type, backend, model,
                        weight, om_save_path, log_save_path, input_shape, precision]
        try:
            subprocess.check_output(command_line)
        except subprocess.CalledProcessError as exc:
            logging.error("convert model to om model failed. the return message is  : {}.".format(exc))

    def _get_200dk_infer_cmd(self, save_path):
        app_dir = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        example_dir = self.current_path + "/samples/atlas200dk"
        ddk_user_name = self.optional_params.get("ddk_user_name")
        ddk_host_ip = self.optional_params.get("ddk_host_ip")
        atlas_host_ip = self.optional_params.get("atlas_host_ip")
        command_line = ["bash", self.current_path + "inference_atlas200.sh",
                        save_path, example_dir, ddk_user_name, ddk_host_ip, atlas_host_ip, app_dir]
        return command_line

    def _compile_atlas300(self, save_path):
        # compile the Davinci program
        example_dir = self.current_path + "/samples/atlas300"
        command_line = ["bash", self.current_path + "/compile_atlas300.sh",
                        example_dir, save_path]
        try:
            subprocess.check_output(command_line)
        except subprocess.CalledProcessError as exc:
            logging.error("compile failed. the return message is : {}.".format(exc))

    def inference(self, converted_model, input_data, is_last=False, cal_metric=False, muti_input=False, **kwargs):
        """Inference in Davinci.

        :param converted_model: converted model file
        :type backend: str
        :param input_data: the input data file
        :type model: str
        """
        if os.path.isfile(converted_model):
            share_dir = os.path.dirname(converted_model)
        else:
            share_dir = converted_model
            converted_model = os.path.join(converted_model, "davinci_model.om")
        log_save_path = os.path.dirname(input_data)
        if self.davinci_environment_type == "ATLAS200DK":
            command_line = self._get_200dk_infer_cmd(save_path=log_save_path)
            result_file = os.path.join(log_save_path, "result_file")
        else:
            if not os.path.exists(os.path.join(share_dir, "main")) and not muti_input:
                self._compile_atlas300(share_dir)
            # execute the Davinci program

            command_line = ["bash", self.current_path + "/inference_atlas300.sh",
                            input_data, converted_model, share_dir, log_save_path, str(muti_input)]
            result_file = os.path.join(log_save_path, "result.txt")

        try:
            subprocess.check_output(command_line)
        except subprocess.CalledProcessError as exc:
            logging.error("inference failed. the return message is : {}.".format(exc))

        latency = self._get_latency(os.path.join(log_save_path, "ome.log"))
        if cal_metric:
            output = self._get_output(result_file)
        else:
            output = None
        if is_last:
            os.remove(input_data)
            os.remove(converted_model)
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
        """Get output data of Davinci."""
        if self.davinci_environment_type == "ATLAS200DK":
            with open(result_file, 'r') as f:
                data = f.readlines()
                labels = []
                for index, line in enumerate(data):
                    if index == 0:
                        continue
                    label = line.split(":")[-1]
                    label = np.float(label)
                    labels.append(label)
                    labels = [labels]
        else:
            with open(result_file, 'r') as f:
                values = f.readlines()
            labels = []
            for value in values:
                labels.append(float(value))
        return labels
