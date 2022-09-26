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

"""Context utils."""
import ssl
import sys
import logging


def create_context(ca_file, cert_pem_file, secret_key_file, ciphers, key_mm=None, key_component_1=None, key_component_2=None):
    """Create the SSL context."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    context.options |= ssl.OP_NO_TLSv1
    context.options |= ssl.OP_NO_TLSv1_1
    if sys.version_info >= (3, 7):
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.options |= ssl.OP_NO_TLSv1_2
        context.options |= ssl.OP_NO_RENEGOTIATION
    context.options -= ssl.OP_ALL
    context.verify_mode = ssl.CERT_REQUIRED
    context.set_ciphers(ciphers)
    if key_mm is not None:
        from .kmc.kmc import decrypt
        logging.debug("Using encrypted key.")
        if key_component_1 is None or key_component_2 is None:
            logging.error("For encrypted key, the component must be provided.")
        decrypt_mm = decrypt(cert_pem_file, secret_key_file, key_mm, key_component_1, key_component_2)
        context.load_cert_chain(cert_pem_file, secret_key_file, password=decrypt_mm)
    else:
        context.load_cert_chain(cert_pem_file, secret_key_file)
    context.load_verify_locations(ca_file)
    return context
