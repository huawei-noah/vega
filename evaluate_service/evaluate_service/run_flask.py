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

"""Run Flask."""

import configparser
import logging
import ssl
import os
from multiprocessing import Process
import gevent
from gevent import pywsgi
from .security.utils import create_context
from .security.verify_config import check_risky_files

white_list = None
request_frequency_limit = "100/minute"
max_content_length = 1000 * 1000 * 1000


def get_request_frequency_limit():
    """Get request frequncy limit."""
    global request_frequency_limit
    return request_frequency_limit


def get_max_content_length():
    """Get max contect length."""
    global max_content_length
    return max_content_length


def get_white_list():
    """Get white list."""
    global white_list
    return white_list


def load_security_setting():
    """Load security settings."""
    home = os.environ['HOME']
    config_file = os.path.join(home, ".vega/vega.ini")
    if not check_risky_files([config_file]):
        return False
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    config = dict(cfg._sections)
    for k in config:
        config[k] = dict(config[k])

    return config


def run_flask(app, host, port, security_mode):
    """Run flask."""
    if security_mode:
        app.config['MAX_CONTENT_LENGTH'] = get_max_content_length()
        config = load_security_setting()
        if not config:
            return False
        ca_cert = config.get('security').get('ca_cert')
        server_cert = config.get('security').get('server_cert')
        server_secret_key = config.get('security').get('server_secret_key')
        encrypted_password = config.get('security').get('encrypted_password')
        key_component_1 = config.get('security').get('key_component_1')
        key_component_2 = config.get('security').get('key_component_2')
        ciphers = config.get('security').get('ciphers')
        cipher_suites = "ECDHE-ECDSA-AES128-CCM:ECDHE-ECDSA-AES256-CCM:ECDHE-ECDSA-AES128-GCM-SHA256" \
                        ":ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384" \
                        ":DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384:DHE-DSS-AES128-GCM-SHA256" \
                        ":DHE-DSS-AES256-GCM-SHA384:DHE-RSA-AES128-CCM:DHE-RSA-AES256-CCM"

        if ciphers:
            ciphersList = [cipher for cipher in ciphers.split(':') if cipher in cipher_suites.split(':')]
            if ciphersList == []:
                raise ssl.SSLError("The ciphers are invalid, please check.")
            else:
                ciphers = ':'.join(ciphersList)
        else:
            ciphers = cipher_suites
        
        if not check_risky_files((ca_cert, server_cert, server_secret_key, key_component_1, key_component_2)):
            return
        try:
            if encrypted_password == "":
                ssl_context = create_context(ca_cert, server_cert, server_secret_key, ciphers)
            else:
                ssl_context = create_context(ca_cert, server_cert, server_secret_key, ciphers,
                                             encrypted_password, key_component_1, key_component_2)
        except Exception:
            logging.error("Fail to create context.")
            return False

        server = pywsgi.WSGIServer((host, port), app, ssl_context=ssl_context)
        if "limit" in config:
            global white_list
            global request_frequency_limit
            global max_content_length
            if "white_list" in config["limit"]:
                white_list = config["limit"]["white_list"].replace(" ", "").split(',')
            if "request_frequency_limit" in config["limit"]:
                request_frequency_limit = config["limit"]["request_frequency_limit"]
            if "max_content_length" in config["limit"]:
                max_content_length = int(config["limit"]["max_content_length"])
    else:
        server = pywsgi.WSGIServer((host, port), app)
    logging.warning("Start the evaluate service.")
    server.init_socket()
    server._stop_event.clear()

    def _server_forever():
        server.start_accepting()
        logging.info("server started.")
        server._stop_event.wait()
        gevent.wait()

    p = Process(target=_server_forever)
    p.start()
