# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The Evaluate Service of the service."""
import os
import logging

try:
    import flask
    import flask_restful
    import werkzeug
except Exception:
    logging.warning(
        "The dependencies [Flask==1.1.2,Flask-RESTful==0.3.8, Werkzeug==1.0.1 ] have not been install, \
        and will install it automatically, if failed, please install it manually.")
    os.system("pip3 install Flask==1.1.2")
    os.system("pip3 install Flask-RESTful==0.3.8")
    os.system("pip3 install Werkzeug==1.0.1")

from flask import Flask, request
from flask_restful import Resource, Api

try:
    from werkzeug import secure_filename
except Exception:
    from werkzeug.utils import secure_filename

import glob
import multiprocessing
import time
import shutil
from evaluate_service.class_factory import ClassFactory
from .hardwares import *  # noqa F401
import datetime
import traceback
import argparse

app = Flask(__name__)
api = Api(app)


class Evaluate(Resource):
    """Evaluate Service for service."""

    def __init__(self):
        self.result = {"latency": "9999", "out_data": [], "status": "sucess", "timestamp": ""}

    @classmethod
    def _add_params(cls, work_path, optional_params):
        cls.current_path = work_path
        cls.optional_params = optional_params

    def post(self):
        """Interface to response to the post request of the client."""
        self.parse_paras()
        self.upload_files()

        self.hardware_instance = ClassFactory.get_cls(self.hardware)(self.optional_params)

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


def _clean_data_path(clean_interval, work_path):
    while True:
        _clean_time = time.time() - clean_interval
        # _current_path = os.path.dirname(os.path.abspath(__file__))
        folder_pattern = "{}/out/*".format(work_path)
        folders = glob.glob(folder_pattern)
        for folder in folders:
            if os.path.isdir(folder) and os.path.getctime(folder) < _clean_time:
                logging.warning("remove old folder: {}".format(folder))
                try:
                    shutil.rmtree(folder)
                except Exception:
                    logging.warning("failed to remove {}".format(folder))
        time.sleep(3600)


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate service")
    parser.add_argument("-i", "--host_ip", type=str, required=True, help="the ip of the evaluate service machine")
    parser.add_argument("-p", "--port", type=int, required=False, default=8888, help="the listening port")
    parser.add_argument("-w", "--work_path", type=str, required=True, help="the work dir to save the file")
    parser.add_argument("-t", "--davinci_environment_type", type=str, required=False, default="ATLAS300",
                        help="the type the davinci hardwares")
    parser.add_argument("-c", "--clean_interval", type=int, required=False, default=1 * 24 * 3600,
                        help="the time interval to clean the temp folder")
    parser.add_argument("-u", "--ddk_user_name", type=str, required=False, default="user",
                        help="the user to acess ATLAS200200 DK")
    parser.add_argument("-atlas_host_ip", "--atlas_host_ip", type=str, required=False, default=None,
                        help="the ip of ATLAS200200 DK")
    args = parser.parse_args()
    return args


def run():
    """Run the evaluate service."""
    args = _parse_args()
    ip_address = args.host_ip
    listen_port = args.port
    clean_interval = args.clean_interval
    work_path = args.work_path
    optional_params = {"davinci_environment_type": args.davinci_environment_type,
                       "ddk_user_name": args.ddk_user_name,
                       "atlas_host_ip": args.atlas_host_ip
                       }
    p = multiprocessing.Process(target=_clean_data_path, args=(clean_interval, work_path), daemon=True)
    p.start()
    Evaluate._add_params(work_path, optional_params)
    api.add_resource(Evaluate, '/')
    app.run(host=ip_address, port=listen_port, threaded=False)
