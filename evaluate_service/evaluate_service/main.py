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

"""The Evaluate Service of the service."""
import logging
import os
import glob
import multiprocessing
import time
import shutil
import datetime
import traceback
import argparse
from flask import abort, Flask, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restful import Resource, Api

try:
    from werkzeug import secure_filename
except Exception:
    from werkzeug.utils import secure_filename
from evaluate_service import security
from evaluate_service.class_factory import ClassFactory
from .hardwares import Davinci
from .run_flask import run_flask, get_white_list, get_request_frequency_limit

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2048 * 1024 * 1024  # 2GB
api = Api(app)


@app.after_request
def after_request(response):
    """Add custom headers for Security reasons."""
    ContentSecurityPolicy = ''
    ContentSecurityPolicy += "default-src 'self'; "
    ContentSecurityPolicy += "script-src 'self' 'unsafe-inline'; "
    ContentSecurityPolicy += "style-src 'self' 'unsafe-inline'; "
    ContentSecurityPolicy += "img-src 'self' data:; "
    ContentSecurityPolicy += "connect-src 'self';"
    response.headers.add('Content-Security-Policy', ContentSecurityPolicy)
    response.headers.add('X-Content-Type-Options', 'nosniff')
    response.headers.add('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    response.headers.add('X-Frame-Options', 'deny')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('X-XSS-Protection', '1; mode=block')
    response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
    response.headers.add('Pragma', 'no-cache')
    response.headers.add('Expires', '0')
    response.headers.add('Connection', 'close')
    return response


limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100/minute"]
)


@app.before_request
def limit_remote_addr():
    """Set limit remote address."""
    client_ip = str(request.remote_addr)
    white_list = get_white_list()
    if Evaluate.security_mode and white_list is not None and client_ip not in white_list:
        abort(403)


class Evaluate(Resource):
    """Evaluate Service for service."""

    security_mode = False
    decorators = [limiter.limit(get_request_frequency_limit, exempt_when=lambda: not Evaluate.security_mode)]

    def __init__(self):
        self.result = {"latency": "9999", "out_data": [], "status": "success", "timestamp": "", "error_message": ""}

    @classmethod
    def _add_params(cls, work_path, security_mode, optional_params):
        cls.current_path = work_path
        cls.security_mode = security_mode
        cls.optional_params = optional_params

    def options(self):
        """Return options."""
        return {"message": "The method is not allowed for the requested URL."}, 405

    def post(self):
        """Interface to response to the post request of the client."""
        try:
            self.parse_paras()
            self.upload_files()
            self.hardware_instance = ClassFactory.get_cls(self.hardware)(self.optional_params)
        except Exception as e:
            self.result["status"] = "Params error."
            self.result["error_message"] = str(e)
            logging.error("[ERROR] Params error!")
            logging.debug(traceback.print_exc())
            return self.result, 400

        if self.reuse_model:
            logging.warning("Reuse the model, no need to convert the model.")
        else:
            try:
                self.hardware_instance.convert_model(backend=self.backend, model=self.model, weight=self.weight,
                                                     save_dir=self.share_dir, input_shape=self.input_shape,
                                                     out_nodes=self.out_nodes, precision=self.precision)
            except Exception as e:
                self.result["status"] = "Model convert failed."
                self.result["error_message"] = str(e)
                logging.error("[ERROR] Model convert failed!")
                logging.debug(traceback.print_exc())
                return self.result, 400
        try:
            latency_sum = 0
            for repeat in range(self.repeat_times):
                is_last = True if repeat == self.repeat_times - 1 else False
                latency, output = self.hardware_instance.inference(converted_model=self.share_dir,
                                                                   input_data=self.input_data,
                                                                   is_last=is_last,
                                                                   cal_metric=self.cal_metric,
                                                                   muti_input=self.muti_input)
                latency_sum += float(latency)
            self.result["latency"] = latency_sum / self.repeat_times
            self.result["out_data"] = output
        except Exception as e:
            self.result["status"] = "Inference failed."
            self.result["error_message"] = str(e)
            logging.error("[ERROR] Inference failed! ")
            logging.debug(traceback.print_exc())
            return self.result, 400
        return self.result, 200

    def parse_paras(self):
        """Parse the parameters in the request from the client."""
        self.backend = request.form["backend"]
        self.hardware = request.form["hardware"]
        self.reuse_model = bool(request.form["reuse_model"].upper() == "TRUE")
        self.cal_metric = request.form.get("cal_metric", type=str, default="False")
        self.cal_metric = bool(self.cal_metric.upper() == "TRUE")
        self.muti_input = request.form.get("muti_input", type=str, default="False")
        self.muti_input = bool(self.muti_input.upper() == "TRUE")
        self.job_id = request.form["job_id"]
        self.input_shape = request.form.get("input_shape", type=str, default="")
        self.out_nodes = request.form.get("out_nodes", type=str, default="")
        self.repeat_times = int(request.form.get("repeat_times"))
        self.precision = request.form.get("precision", type=str, default="FP32")
        if self.security_mode:
            security.args.check_backend(self.backend)
            security.args.check_hardware(self.hardware)
            security.args.check_job_id(self.job_id)
            security.args.check_input_shape(self.input_shape)
            security.args.check_out_nodes(self.out_nodes)
            security.args.check_repeat_times(self.repeat_times)
            security.args.check_precision(self.precision)

    def upload_files(self):
        """Upload the files from the client to the service."""
        self.now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        self.result["timestamp"] = self.now_time
        logging.warning("The timestamp is {}.".format(self.now_time))
        self.upload_file_path = os.path.join(self.current_path, "out", self.now_time)
        self.share_dir = os.path.join(self.current_path, "out", self.job_id)
        if not os.path.exists(self.upload_file_path):
            os.makedirs(self.upload_file_path)
        if not os.path.exists(self.share_dir):
            os.makedirs(self.share_dir)
        patterns = [".pkl", ".pth", ".pt", ".pb", ".ckpt", ".air", '.om',
                    ".onnx", ".caffemodel", ".pbtxt", ".prototxt"]
        model_file = request.files.get("model_file")
        if model_file is not None:
            self.model = self.upload_file_path + "/" + secure_filename(model_file.filename)
            if os.path.splitext(self.model)[1] not in patterns:
                raise ValueError(f'{model_file.filename} file type is not supported.')
            model_file.save(self.model)

        data_file = request.files.get("data_file")
        if data_file is not None:
            self.input_data = self.upload_file_path + "/" + secure_filename(data_file.filename)
            if not os.path.basename(self.input_data) == 'input.bin':
                raise ValueError(f'data {data_file.filename} file is not supported.')
            data_file.save(self.input_data)

        weight_file = request.files.get("weight_file")
        if weight_file is not None:
            self.weight = self.upload_file_path + "/" + secure_filename(weight_file.filename)
            if os.path.splitext(self.weight)[1] not in patterns:
                raise ValueError(f'{weight_file.filename} file type is not supported.')
            weight_file.save(self.weight)
        else:
            self.weight = ""
        logging.warning("upload file success!")


def _clean_data_path(clean_interval, work_path):
    while True:
        _clean_time = time.time() - clean_interval
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
    parser.add_argument("-t", "--davinci_environment_type", type=str, required=False, default="Ascend310",
                        choices=Davinci.supported_devices,
                        help="the type of the davinci hardwares")
    parser.add_argument("-c", "--clean_interval", type=int, required=False, default=1 * 6 * 3600,
                        help="the time interval to clean the temp folder")
    parser.add_argument("-u", "--ddk_user_name", type=str, required=False, default="user",
                        help="the user access to ATLAS 200 DK")
    parser.add_argument("-atlas_host_ip", "--atlas_host_ip", type=str, required=False, default=None,
                        help="the ip of ATLAS 200 DK")
    parser.add_argument("-s", "--security_mode", action='store_true',
                        help="enable safe mode")
    args = parser.parse_args()
    return args


def run():
    """Run the evaluate service."""
    args = _parse_args()
    ip_address = args.host_ip
    listen_port = args.port
    clean_interval = args.clean_interval
    work_path = args.work_path
    security_mode = args.security_mode
    if security_mode:
        os.umask(0o077)
    optional_params = {"davinci_environment_type": args.davinci_environment_type,
                       "ddk_user_name": args.ddk_user_name,
                       "atlas_host_ip": args.atlas_host_ip
                       }
    p = multiprocessing.Process(target=_clean_data_path, args=(clean_interval, work_path), daemon=True)
    p.start()
    Evaluate._add_params(work_path, args.security_mode, optional_params)
    api.add_resource(Evaluate, '/')

    run_flask(app, host=ip_address, port=listen_port, security_mode=security_mode)
