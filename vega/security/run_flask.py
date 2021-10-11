# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run Flask."""

import configparser
import getpass
import logging
import re
import os
import ssl
import stat
import gevent
from gevent import pywsgi


security_mode = True
cert_pem_file = ""
secret_key_file = ""
white_list = None
request_frequency_limit = "100/minute"
max_content_length = 1000 * 1000 * 1000


def load_security_setting():
    """Load security settings."""
    home = os.environ['HOME']
    config_file = os.path.join(home, ".vega/vega.ini")
    if not os.path.exists(config_file):
        print(f"Not found configure file: {config_file}")
        return False
    config = configparser.ConfigParser()
    config.read(config_file)
    if "limit" in config:
        global white_list
        global request_frequency_limit
        global max_content_length
        if "white_list" in config["limit"]:
            white_list = config["limit"]["white_list"].split(',')
        if "request_frequency_limit" in config["limit"]:
            request_frequency_limit = config["limit"]["request_frequency_limit"]
        if "max_content_length" in config["limit"]:
            max_content_length = int(config["limit"]["max_content_length"])
    if "security" not in config or "enable" not in config["security"]:
        print(f"Invalid config file: {config_file},security field must be included")
        return False
    global security_mode
    security_mode = True if config["security"]["enable"].upper() == "TRUE" else False
    if security_mode:
        if "https" not in config or \
                "cert_pem_file" not in config["https"] or \
                "secret_key_file" not in config["https"]:
            print(f"Invalid config file: {config_file},https field must be included")
            return False
        https_config = config["https"]
        global cert_pem_file
        global secret_key_file
        if not os.path.exists(https_config['cert_pem_file']):
            print(f"CERT file ({https_config['cert_pem_file']}) is not existed.")
            return False
        if not os.path.exists(https_config['secret_key_file']):
            print(f"KEY file ({https_config['secret_key_file']}) is not existed.")
            return False
        cert_pem_file = https_config['cert_pem_file']
        secret_key_file = https_config['secret_key_file']

        if not check_cert_key_file(cert_pem_file, secret_key_file):
            return False
    return True


def check_cert_key_file(cert_file, key_file):
    """Check if cert and key file are risky."""
    res = True
    for file in (cert_file, key_file):
        if not os.stat(file).st_uid == os.getuid():
            logging.error("File <{}> is not owned by current user".format(file))
            res = False
        if os.path.islink(file):
            logging.error("File <{}> should not be soft link".format(file))
            res = False
        if os.stat(file).st_mode & 0o0077:
            logging.error("file <{}> is accessible by group/other users".format(file))
            res = False

    return res


def get_white_list():
    """Get white list."""
    global white_list
    return white_list


def get_request_frequency_limit():
    """Get request frequncy limit."""
    global request_frequency_limit
    return request_frequency_limit


def get_max_content_length():
    """Get max contect length."""
    global max_content_length
    return max_content_length


def check_password_rule(password):
    """Check password rule."""
    digit_regex = re.compile(r'\d')
    upper_regex = re.compile(r'[A-Z]')
    lower_regex = re.compile(r'[a-z]')

    if len(password) < 8:
        print("The length of your password must >= 8")
        return False

    if len(digit_regex.findall(password)) == 0:
        print("Your password must contains digit letters")
        return False

    if len(upper_regex.findall(password)) == 0:
        print("Your password must contains capital letters")
        return False

    if len(lower_regex.findall(password)) == 0:
        print("Your password must contains lowercase letters")
        return False

    return True


def get_secret_key_passwd():
    """Get secret key password."""
    password = getpass.getpass("Please input password of your server key: ")

    if not check_password_rule(password):
        print("You should re-generate your server cert/key by a password with following rules:")
        print("1. equals to or longer than 8 letters")
        print("2. contains at least one digit letter")
        print("3. contains at least one capital letter")
        print("4. contains at least one lowercase letter")
        return None

    return password


def run_flask(app, host, port):
    """Run flask."""
    if not load_security_setting():
        return

    app.config['MAX_CONTENT_LENGTH'] = get_max_content_length()

    global security_mode
    if security_mode:
        ciphers = "ECDHE-ECDSA-AES128-CCM:ECDHE-ECDSA-AES256-CCM:ECDHE-ECDSA-AES128-GCM-SHA256"\
                  ":ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384"\
                  ":DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384:DHE-DSS-AES128-GCM-SHA256"\
                  ":DHE-DSS-AES256-GCM-SHA384:DHE-RSA-AES128-CCM:DHE-RSA-AES256-CCM"
        password = get_secret_key_passwd()
        if password is None:
            return
        global cert_pem_file
        global secret_key_file
        context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        context.set_ciphers(ciphers)
        context.load_cert_chain(certfile=cert_pem_file, keyfile=secret_key_file, password=password)
        server = pywsgi.WSGIServer((host, port), app, ssl_context=context)
    else:
        server = pywsgi.WSGIServer((host, port), app)

    server.init_socket()
    server._stop_event.clear()

    def server_forever():
        server.start_accepting()
        print("server started.")
        server._stop_event.wait()
        gevent.wait()

    from multiprocessing import Process
    p = Process(target=server_forever)
    p.start()
