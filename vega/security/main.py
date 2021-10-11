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
    import flask_limiter
    import werkzeug
    import gevent
except Exception:
    logging.warning(
        "The dependencies [Flask==1.1.2,Flask-RESTful==0.3.8, Werkzeug==1.0.1 ] have not been install, \
        and will install it automatically, if failed, please install it manually.")
    os.system("pip3 install Flask==1.1.2")
    os.system("pip3 install Flask-RESTful==0.3.8")
    os.system("pip3 install Flask-Limiter==1.4")
    os.system("pip3 install Werkzeug==1.0.1")
    os.system("pip3 install gevent")

from flask import abort, Flask, request, Response
from flask_restful import Resource, Api
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

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
from .run_flask import run_flask, get_white_list, get_request_frequency_limit

app = Flask(__name__)
api = Api(app)

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100/minute"]
)


MAX_EVAL_EPOCHS = 10000


@app.before_request
def limit_remote_addr():
    """Set limit remote address."""
    client_ip = str(request.remote_addr)
    white_list = get_white_list()
    if white_list and client_ip not in white_list:
        abort(403)


class Evaluate(Resource):
    """Evaluate Service for service."""

    decorators = [limiter.limit(get_request_frequency_limit)]

    def __init__(self):
        self.result = {"latency": "9999", "out_data": [], "status": "sucess", "timestamp": "", "error_message": ""}

    @classmethod
    def _add_params(cls, work_path, optional_params):
        cls.current_path = work_path
        cls.optional_params = optional_params

    def post(self):
        """Interface to response to the post request of the client."""
        try:
            self.parse_paras()
            self.upload_files()
            self.hardware_instance = ClassFactory.get_cls(self.hardware)(self.optional_params)
        except Exception:
            self.result["status"] = "Params error."
            self.result["error_message"] = traceback.format_exc()
            logging.error("[ERROR] Params error!")
            traceback.print_exc()
            return self.result

        if self.reuse_model == "True":
            logging.warning("Reuse the model, no need to convert the model.")
        else:
            try:
                self.hardware_instance.convert_model(backend=self.backend, model=self.model, weight=self.weight,
                                                     save_dir=self.share_dir, input_shape=self.input_shape,
                                                     out_nodes=self.out_nodes, precision=self.precision)
            except Exception:
                self.result["status"] = "Model convert failed."
                self.result["error_message"] = traceback.format_exc()
                logging.error("[ERROR] Model convert failed!")
                traceback.print_exc()
                return self.result
        try:
            latency_sum = 0
            for repeat in range(min(self.repeat_times, 10)):
                latency, output = self.hardware_instance.inference(converted_model=self.share_dir,
                                                                   input_data=self.input_data)
                latency_sum += float(latency)
            self.result["latency"] = latency_sum / self.repeat_times
            self.result["out_data"] = output
        except Exception:
            self.result["status"] = "Inference failed."
            self.result["error_message"] = traceback.format_exc()
            logging.error("[ERROR] Inference failed! ")
            traceback.print_exc()
        return self.result

    def parse_paras(self):
        """Parse the parameters in the request from the client."""
        self.backend = request.form["backend"]
        self.hardware = request.form["hardware"]
        self.reuse_model = request.form["reuse_model"]
        self.job_id = self._check_get_job_id(request.form["job_id"])
        self.input_shape = request.form.get("input_shape", type=str, default="")
        self.out_nodes = request.form.get("out_nodes", type=str, default="")
        self.repeat_times = self._check_get_repeat_times(request.form.get("repeat_times"))
        self.precision = request.form.get("precision", type=str, default="FP32")

    @staticmethod
    def _check_get_repeat_times(repeat_times):
        """Check validation of input repeat_times."""
        _repeat_times = repeat_times
        try:
            _repeat_times = int(_repeat_times)
        except ValueError:
            logging.warning("repeat_times {} is not a valid integer".format(_repeat_times))
            abort(400, "repeat_times {} is not a valid integer".format(_repeat_times))
        if not 0 < _repeat_times <= MAX_EVAL_EPOCHS:
            logging.warning("repeat_times {} is not in valid range (1-{})".format(_repeat_times, MAX_EVAL_EPOCHS))
            abort(400, "repeat_times {} is not in valid range (1-{})".format(_repeat_times, MAX_EVAL_EPOCHS))
        return _repeat_times

    @staticmethod
    def _check_get_job_id(job_id):
        """Check validation of params."""
        import re
        if len(re.compile("[^_A-Za-z0-9]").findall(job_id)) > 0:
            logging.warning("job_id {} contains invalid characters".format(job_id))
            abort(400, "job_id {} contains invalid characters".format(job_id))
        return job_id

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
    parser.add_argument("-c", "--clean_interval", type=int, required=False, default=1 * 6 * 3600,
                        help="the time interval to clean the temp folder")
    parser.add_argument("-u", "--ddk_user_name", type=str, required=False, default="user",
                        help="the user to acess ATLAS200200 DK")
    parser.add_argument("-atlas_host_ip", "--atlas_host_ip", type=str, required=False, default=None,
                        help="the ip of ATLAS200200 DK")

    args = parser.parse_args()
    return args


def run():
    """Run the evaluate service."""
    os.umask(0o027)
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
    run_flask(app, host=ip_address, port=listen_port)
