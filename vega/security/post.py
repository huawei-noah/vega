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

"""Rest post operation in security mode."""

import urllib
import ssl
import json
import logging
import requests

from .conf import get_config
from .utils import create_context
from .args import check_msg
from .verify_cert import verify_cert


def post(host, files, data):
    """Post a REST requstion in security mode."""
    sec_cfg = get_config('client')

    ca_file = sec_cfg.ca_cert
    cert_pem_file = sec_cfg.client_cert
    secret_key_file = sec_cfg.client_secret_key
    encrypted_password = sec_cfg.encrypted_password
    key_component_1 = sec_cfg.key_component_1
    key_component_2 = sec_cfg.key_component_2
    ciphers = sec_cfg.ciphers
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

    if not cert_pem_file or not secret_key_file or not ca_file:
        logging.error("CERT file is not existed.")

    if not verify_cert(ca_file, cert_pem_file):
        logging.error(f"The cert {ca_file} and {cert_pem_file} are invalid, please check.")

    if encrypted_password == "":
        context = create_context(ca_file, cert_pem_file, secret_key_file, ciphers)
    else:
        context = create_context(ca_file, cert_pem_file, secret_key_file, ciphers, encrypted_password, key_component_1,
                                 key_component_2)
    if host.lower().startswith('https') is False:
        raise Exception(f'The host {host} must start with https')
    prepped = requests.Request(method="POST", url=host, files=files, data=data).prepare()
    request = urllib.request.Request(host, data=prepped.body, method='POST')
    request.add_header("Content-Type", prepped.headers['Content-Type'])
    response = urllib.request.urlopen(request, context=context)  # nosec
    result = json.loads(response.read().decode('utf8'))
    check_msg(dict((key, value) for key, value in result.items() if key != 'error_message'))
    return result
