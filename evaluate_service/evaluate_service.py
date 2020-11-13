# -*- coding: utf-8 -*-
"""The Evaluate Service of the service."""
from flask import Flask, request
from flask_restful import Resource, Api
from werkzeug import secure_filename
import subprocess
import datetime
import os
import logging
import tensorflow as tf
import onnx
import numpy as np
from onnx_tf.backend import prepare
from config import *


app = Flask(__name__)
api = Api(app)

result = {"latency": "-1", "out_data": [], "status_code": 200, "timestamp": ""}

STATUS_CODE_MAP = {
    "SUCESS": 200,
    "FILE_IS_NONE": 400,
    "MODEL_CONVERT_FAILED": 401,
    "MODEL_EXE_FAILED": 402
}


class Evaluate(Resource):
    """Evaluate Service for service."""

    def __init__(self):
        # subprocess.Popen("source env.sh", shell=True,executable="/bin/bash")
        self.current_path = os.path.dirname(os.path.abspath(__file__))

    def get(self):
        """Interface to response to the get request of the client."""
        return result

    def post(self):
        """Interface to response to the post request of the client."""
        self.parse_paras()
        self.upload_files()

        self.convert_model()
        self.inference()

    def parse_paras(self):
        """Parse the parameters in the request from the client."""
        self.backend = request.form["backend"]
        self.hardware = request.form["hardware"]

    def upload_files(self):
        """Upload the files from the client to the service."""
        self.now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        result["timestamp"] = self.now_time
        logging.warning("The timestamp is {}.".format(self.now_time))
        self.upload_file_path = os.path.join(self.current_path, "out", self.now_time)
        os.makedirs(self.upload_file_path)

        model_file = request.files.get("model_file")
        if model_file is not None:
            model_file.save(self.upload_file_path + "/" + secure_filename(model_file.filename))

        data_file = request.files.get("data_file")
        if data_file is not None:
            data_file.save(self.upload_file_path + "/" + secure_filename(data_file.filename))

        weight_file = request.files.get("weight_file")
        if weight_file is not None:
            weight_file.save(self.upload_file_path + "/" + secure_filename(weight_file.filename))
        logging.warning("upload file sucess!")

    def convert_model(self):
        """Convert the tf/caffe model to om model in Davinc and convert the tf model to tflite model in Bolt."""
        if self.hardware == "Davinci":
            self.convert_model_Davinci()
        elif self.hardware == "Bolt":
            self.convert_model_Bolt()
        else:
            raise ValueError("The hardware only support Davinci and Bolt.")

    def inference(self):
        """Interface in Davinci or Bolt and return the latency."""
        if self.hardware == "Davinci":
            self.inference_Davinci()
        elif self.hardware == "Bolt":
            self.inference_Bolt()
        else:
            raise ValueError("The hardware only support Davinci and Bolt.")

    def convert_model_Davinci(self):
        """Convert the tf/caffe/Pytorch model to om model in Davinci."""
        if self.backend == "tensorflow":
            model_path = self.locate_file(self.upload_file_path, ".pb")
            om_save_path = self.upload_file_path
            if davinci_environment_type == "ATLAS200DK":
                command_line = ["bash", self.current_path + "/utils/atlas200_dk/tf2om.sh", model_path, om_save_path]
            else:
                command_line = ["bash", self.current_path + "/utils/tf2om.sh", model_path, om_save_path]
            try:
                subprocess.check_output(command_line)
            except subprocess.CalledProcessError as exc:
                logging.warning("convert tf model to om model failed. the return code is : {}." .format(exc.returncode))
                result["status_code"] = STATUS_CODE_MAP["MODEL_CONVERT_FAILED"]

        elif self.backend == "caffe":
            model_path = self.locate_file(self.upload_file_path, ".prototxt")
            weight_path = self.locate_file(self.upload_file_path, ".caffemodel")
            om_save_path = self.upload_file_path
            if davinci_environment_type == "ATLAS200DK":
                command_line = ["bash", self.current_path + "/utils/atlas200_dk/caffe2om.sh",
                                model_path, weight_path, om_save_path]
            else:
                command_line = ["bash", self.current_path + "/utils/caffe2om.sh",
                                model_path, weight_path, om_save_path]

            try:
                subprocess.check_output(command_line)
            except subprocess.CalledProcessError as exc:
                logging.warning("convert caffe model to om model failed. the return code is : {}."
                                .format(exc.returncode))
                result["status_code"] = STATUS_CODE_MAP["MODEL_CONVERT_FAILED"]

        elif self.backend == "pytorch":
            model_path = self.locate_file(self.upload_file_path, ".onnx")
            model_name = os.path.basename(model_path).split(".")[0]
            pb_model_path = os.path.join(self.upload_file_path, model_name + ".pb")
            self.onnx2tf(model_path, pb_model_path)
            om_save_path = self.upload_file_path
            if davinci_environment_type == "ATLAS200DK":
                command_line = ["bash", self.current_path + "/utils/atlas200_dk/tf2om.sh", pb_model_path, om_save_path]
            else:
                command_line = ["bash", self.current_path + "/utils/tf2om.sh", pb_model_path, om_save_path]
            try:
                subprocess.check_output(command_line)
            except subprocess.CalledProcessError as exc:
                logging.warning("convert tf model to om model failed. the return code is : {}." .format(exc.returncode))
                result["status_code"] = STATUS_CODE_MAP["MODEL_CONVERT_FAILED"]
        else:
            raise ValueError("The frmawork must be Tensorflow, Caffe or Pytorch.")

    def convert_model_Bolt(self):
        """Convert the tf model to tflite model in Bolt."""
        if self.backend == "tensorflow":
            model_path = self.locate_file(self.upload_file_path, ".pb")
            model_name = os.path.basename(model_path).split(".")[0]
            tflite_model_path = os.path.join(self.upload_file_path, model_name + ".tflite")
            self.tf2tflite(model_path, ["input_tensor"], ["softmax_tensor"], tflite_model_path)
            mobile_dir = "/data/evaluate_service/" + self.now_time
            log_save_path = self.upload_file_path
            precision = "FP32"
            command_line = ["bash", self.current_path + "/utils/tflite2bolt.sh", mobile_dir,
                            model_path, model_name, precision, log_save_path]
            try:
                subprocess.check_output(command_line)
            except subprocess.CalledProcessError as exc:
                logging.warning("convert tflite model to bolt model failed. the return code is : {}."
                                .format(exc.returncode))
                result["status_code"] = STATUS_CODE_MAP["MODEL_CONVERT_FAILED"]

        elif self.backend == "caffe":
            model_path = self.locate_file(self.upload_file_path, ".prototxt")
            weight_path = self.locate_file(self.upload_file_path, ".caffemodel")
            mobile_dir = "/data/evaluate_service/" + self.now_time
            model_name = os.path.basename(model_path).split(".")[0]
            weight_name = os.path.basename(weight_path).split(".")[0]
            log_save_path = self.upload_file_path
            if model_name != weight_name:
                logging.warning("The model name and weight name should be same.")
                os.rename(os.path.join(self.upload_file_path, weight_path + ".caffemodel"),
                          os.path.join(self.upload_file_path, model_path + ".caffemodel"))
                weight_path = self.locate_file(self.upload_file_path, ".caffemodel")
            precision = "FP32"
            command_line = ["bash", self.current_path + "/utils/caffe2bolt.sh", mobile_dir,
                            model_path, weight_path, model_name, precision, log_save_path]
            try:
                subprocess.check_output(command_line)
            except subprocess.CalledProcessError as exc:
                logging.warning("convert caffe model to bolt model failed. the return code is : {}."
                                .format(exc.returncode))
                result["status_code"] = STATUS_CODE_MAP["MODEL_CONVERT_FAILED"]

        elif self.backend == "pytorch":
            model_path = self.locate_file(self.upload_file_path, ".onnx")
            mobile_dir = "/sdcard/evaluate_service/" + self.now_time
            model_name = os.path.basename(model_path).split(".")[0]
            log_save_path = self.upload_file_path
            precision = "FP32"
            command_line = ["bash", self.current_path + "/utils/onnx2bolt.sh", mobile_dir,
                            model_path, model_name, precision, log_save_path]
            try:
                subprocess.check_output(command_line)
            except subprocess.CalledProcessError as exc:
                logging.warning("convert tflite model to bolt model failed. the return code is : {}."
                                .format(exc.returncode))
                result["status_code"] = STATUS_CODE_MAP["MODEL_CONVERT_FAILED"]
        else:
            raise ValueError("The frmawork must be Tensorflow, Caffe or Pytorch.")

    def inference_Davinci(self):
        """Interface in Davinci and return the latency."""
        om_model_path = self.locate_file(self.upload_file_path, '.om')
        if davinci_environment_type == "ATLAS200DK":
            task_dir = self.upload_file_path
            app_dir = self.now_time
            example_dir = self.current_path + "/samples/atlas200dk"
            command_line = ["bash", self.current_path + "/utils/atlas200_dk/inference_davinci.sh",
                            task_dir, example_dir, ddk_user_name, ddk_host_ip, atlas_host_ip, app_dir]
        elif davinci_environment_type == "ATLAS300":
            example_dir = self.current_path + "/samples/atlas300"
            command_line = ["bash", self.current_path + "/utils/atlas300/inference_davinci.sh",
                            self.upload_file_path, example_dir]
        else:
            try:
                data_path = self.locate_file(self.upload_file_path, '.bin')
            except:
                data_path = self.locate_file(self.upload_file_path, '.data')
            output_data_path = self.upload_file_path
            command_line = ["bash", self.current_path + "/utils/inference_davinci.sh",
                            om_model_path, data_path, output_data_path]

        try:
            subprocess.check_output(command_line)
        except subprocess.CalledProcessError as exc:
            logging.warning("inference failed. the return code is : {}." .format(exc.returncode))
            result["status_code"] = STATUS_CODE_MAP["MODEL_EXE_FAILED"]
        result["latency"] = self.get_latency_from_log()
        result["out_data"] = self.get_output_davinci()

    def inference_Bolt(self):
        """Interface in Bolt and return the latency."""
        mobile_dir = "/sdcard/evaluate_service/" + self.now_time
        data_path = self.locate_file(self.upload_file_path, ".bin")
        if self.backend == "tensorflow":
            suffix = ".pb"
        elif self.backend == "caffe":
            suffix = ".prototxt"
        else:
            suffix = ".onnx"
        bolt_model_name = os.path.basename(self.locate_file(self.upload_file_path,
                                           suffix)).split(".")[0] + "_f32.bolt"
        model_path = os.path.join(mobile_dir, bolt_model_name)
        output_data_path = self.upload_file_path
        command_line = ["bash", self.current_path + "/utils/inference_bolt.sh",
                        model_path, data_path, mobile_dir, output_data_path]
        try:
            subprocess.check_output(command_line)
        except subprocess.CalledProcessError as exc:
            logging.warning("inference failed. the return code is : {}." .format(exc.returncode))
            result["status_code"] = STATUS_CODE_MAP["MODEL_EXE_FAILED"]
        result["latency"] = self.get_latency_from_log()
        result["out_data"] = self.get_output_bolt()

    def locate_file(self, path, suffix):
        """Find the specific suffix file according to the path."""
        file_list = os.listdir(path)
        file_name = [file for file in file_list if file.endswith(suffix)][0]
        return os.path.join(path, file_name)

    def tf2tflite(self, pb_file, input_arrays, output_arrays, save_name):
        """Convert the tensorflow model to tflite mdoel.

        :param pb_file: tensorflow model file
        :type pb_file: str
        :param input_arrays: input tensor list
        :type input_arrays: list
        :param output_arrays: output tensor list
        :type output_arrays: str
        :param save_name: save name of tflite mmodel
        :type save_name: str
        """
        converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_file, input_arrays, output_arrays)
        tflite_model = converter.convert()
        with open(save_name, "wb") as f:
            f.write(tflite_model)

    def onnx2tf(self, onnx_file, pb_save_path):
        """Convert the onnx model to tensorflow model.

        :param onnx_file: onnx file
        :type onnx_file: str
        :param pb_save_path: the path and name to save the converted pb model
        :type pb_save_path: str
        """
        # Load the ONNX file
        model = onnx.load(onnx_file)
        # Import the ONNX model to Tensorflow
        tf_rep = prepare(model)
        tf_rep.export_graph(pb_save_path)

    def get_latency_from_log(self):
        """Get latency from the log file."""
        log_file = os.path.join(self.upload_file_path, "ome.log")
        logging.warning("The log file is {}.".format(log_file))
        # command_line = "cat  log.txt |grep costTime | awk -F  ' '  '{print $NF}' "
        command_line = ["bash", self.current_path + "/utils/get_latency_from_log.sh", log_file, self.hardware]
        try:
            latency = subprocess.check_output(command_line)
            return str(latency, 'utf-8').split("\n")[0]
        except subprocess.CalledProcessError as exc:
            logging.warning("get_latency_from_log failed. the return code is : {}." .format(exc.returncode))
            result['status_code'] = STATUS_CODE_MAP["FILE_IS_NONE"]

    def get_output_davinci(self):
        """Get output data of davinci."""
        result_file = os.path.join(self.upload_file_path, "result_file")
        with open(result_file, 'r') as f:
            data = f.readlines()
            labels = []
            for index, line in enumerate(data):
                if index == 0:
                    continue
                label = line.split(":")[-1]
                label = np.float(label)
                labels.append(label)
        return [labels]

    def get_output_bolt(self):
        """Get output data of bolt."""
        output_file = os.path.join(self.upload_file_path, "BoltResult.txt")
        with open(output_file, 'r') as f:
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


api.add_resource(Evaluate, '/')

if __name__ == '__main__':
    app.run(host=ddk_host_ip, port=listen_port)
