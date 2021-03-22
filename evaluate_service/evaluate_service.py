# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The Evaluate Service of the service."""
from flask import Flask, request
from flask_restful import Resource, Api
import glob
import multiprocessing
import time
import shutil

try:
    from werkzeug import secure_filename
except Exception:
    from werkzeug.utils import secure_filename
from class_factory import ClassFactory
from hardwares import *  # noqa F401
import datetime
import os
import logging
from config import ip_address, listen_port, optional_params, clean_interval
import traceback

app = Flask(__name__)
api = Api(app)


class Evaluate(Resource):
    """Evaluate Service for service."""

    def __init__(self):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.result = {"latency": "-1", "out_data": [], "status": "sucess", "timestamp": ""}

    def post(self):
        """Interface to response to the post request of the client."""
        self.parse_paras()
        self.upload_files()

        self.hardware_instance = ClassFactory.get_cls(self.hardware)(optional_params)

        if self.reuse_model == "True":
            logging.warning("Reuse the model, no need to convert the model.")
        else:
            try:
                self.hardware_instance.convert_model(backend=self.backend, model=self.model, weight=self.weight,
                                                     save_dir=self.share_dir, input_shape=self.input_shape,
                                                     out_nodes=self.out_nodes)
            except Exception:
                self.result["status"] = "Model convert failed."
                logging.error("[ERROR] Model convert failed!")
                traceback.print_exc()
        try:
            latency_sum = 0
            for repeat in range(self.repeat_times):
                latency, output = self.hardware_instance.inference(converted_model=self.share_dir,
                                                                   input_data=self.input_data)
                latency_sum += float(latency)
            self.result["latency"] = latency_sum / self.repeat_times
            self.result["out_data"] = output
        except Exception:
            self.result["status"] = "Inference failed."
            logging.error("[ERROR] Inference failed! ")
            traceback.print_exc()

        return self.result

    def parse_paras(self):
        """Parse the parameters in the request from the client."""
        self.backend = request.form["backend"]
        self.hardware = request.form["hardware"]
        self.reuse_model = request.form["reuse_model"]
        self.job_id = request.form["job_id"]
        self.input_shape = request.form.get("input_shape", type=str, default="")
        self.out_nodes = request.form.get("out_nodes", type=str, default="")
        self.repeat_times = int(request.form.get("repeat_times"))

    def upload_files(self):
        """Upload the files from the client to the service."""
        self.now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        self.result["timestamp"] = self.now_time
        logging.warning("The timestamp is {}.".format(self.now_time))
        self.upload_file_path = os.path.join(self.current_path, "out", self.now_time)
        self.share_dir = os.path.join(self.current_path, "out", self.job_id)
        os.makedirs(self.upload_file_path)

        model_file = request.files.get("model_file")
        if model_file is not None:
            self.model = self.upload_file_path + "/" + secure_filename(model_file.filename)
            model_file.save(self.model)

        data_file = request.files.get("data_file")
        if data_file is not None:
            self.input_data = self.upload_file_path + "/" + secure_filename(data_file.filename)
            data_file.save(self.input_data)

        weight_file = request.files.get("weight_file")
        if weight_file is not None:
            self.weight = self.upload_file_path + "/" + secure_filename(weight_file.filename)
            weight_file.save(self.weight)
        else:
            self.weight = ""
        logging.warning("upload file sucess!")


api.add_resource(Evaluate, '/')


def _clean_data_path():
    while True:
        _clean_time = time.time() - clean_interval
        _current_path = os.path.dirname(os.path.abspath(__file__))
        folder_pattern = "{}/out/*".format(_current_path)
        folders = glob.glob(folder_pattern)
        for folder in folders:
            if os.path.isdir(folder) and os.path.getctime(folder) < _clean_time:
                logging.info("remove old folder: {}".format(folder))
                try:
                    shutil.rmtree(folder)
                except Exception:
                    logging.warn("failed to remove {}".format(folder))
        time.sleep(3600)


if __name__ == '__main__':
    p = multiprocessing.Process(target=_clean_data_path, daemon=True)
    p.start()
    app.run(host=ip_address, port=listen_port, threaded=False)
