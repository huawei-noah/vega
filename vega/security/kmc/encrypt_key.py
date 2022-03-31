# -*- coding:utf-8 -*-

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

"""Load the Certificate and encrypt the passwd."""

import argparse
import getpass
import logging
import subprocess
from OpenSSL.crypto import load_certificate, FILETYPE_PEM, load_privatekey
from . import kmc
from .utils import check_password_rule


def encrypt_mm(origin_mm, key_component_1, key_component_2):
    """Encrypt the passwd."""
    ret = kmc.init(key_component_1, key_component_2, 9)
    if ret is False:
        logging.error("kmc init error.")
        return ""
    domain_id = 0
    result = kmc.encrypt(domain_id, origin_mm)
    kmc.finalize()
    return result


def validate_certificate(cert, key, origin_mm):
    """Validate the certificate."""
    flag = True
    with open(key, "r", encoding="utf-8") as f:
        key_value = f.read()
    try:
        load_privatekey(FILETYPE_PEM, key_value, passphrase=origin_mm.encode('utf-8'))
    except Exception:
        flag = False
        logging.error("Wrong PEM.")
        return flag

    # check signature algorithm
    with open(cert, "r", encoding="utf-8") as f:
        cert_value = f.read()
        cert_value = load_certificate(FILETYPE_PEM, cert_value)
    enc_algorithm = cert_value.get_signature_algorithm()
    if enc_algorithm in b'sha1WithRSAEncryption' b'md5WithRSAEncryption':
        logging.warning("Insecure encryption algorithm: %s", enc_algorithm)
    # check key length

    p1 = subprocess.Popen(["openssl", "x509", "-in", cert, "-text", "-noout"],
                          stdout=subprocess.PIPE, shell=False)
    p2 = subprocess.Popen(["grep", "RSA Public-Key"], stdin=p1.stdout, stdout=subprocess.PIPE, shell=False)
    p3 = subprocess.Popen(["tr", "-cd", "[0-9]"], stdin=p2.stdout, stdout=subprocess.PIPE, shell=False)
    RSA_key = p3.communicate()[0]
    if int(RSA_key) < 3072:
        logging.warning("Insecure key length: %d. The recommended key length is at least 3072", int(RSA_key))
    return flag


def import_certificate(args, origin_mm):
    """Load the certificate."""
    # 1.validate private key and certification, if not pass, program will exit
    ret = validate_certificate(args.cert, args.key, origin_mm)
    if not ret:
        logging.error("Validate certificate failed.")
        return 0

    # 2.encrypt private key's passwd.
    encrypt = encrypt_mm(origin_mm, args.key_component_1, args.key_component_2)
    if not encrypt:
        logging.error("kmc encrypt private key error.")
        return 0
    logging.warning(f"Encrypt sucuess. The encrypted of your input is {encrypt}")
    logging.warning(f"The key components are {args.key_component_1} and {args.key_component_2}, please keep it safe.")

    return True


def args_parse():
    """Parse the input args."""
    parser = argparse.ArgumentParser(description='Certificate import')
    parser.add_argument("--cert", default="./kmc/config/crt/sever.cert", type=str,
                        help="The path of certificate file")
    parser.add_argument("--key", default='./kmc/config/crt/sever.key', type=str,
                        help="The path of private Key file.")
    parser.add_argument("--key_component_1", default='./kmc/config/ksf/ksmaster.dat', type=str,
                        help="key material 1.")
    parser.add_argument("--key_component_2", default='./kmc/config/ksf/ksstandby.dat', type=str,
                        help="key material 2.")

    args = parser.parse_args()

    return args


def main():
    """Run the encrypt process."""
    args = args_parse()
    logging.info("process encrypt begin.")
    origin_mm = getpass.getpass("Please enter the password to be encrypted: ")
    if not check_password_rule(origin_mm):
        logging.info("You should re-generate your server cert/key with following rules:")
        logging.info("1. equals to or longer than 8 letters")
        logging.info("2. contains at least one digit letter")
        logging.info("3. contains at least one capital letter")
        logging.info("4. contains at least one lowercase letter")

    ret = import_certificate(args, origin_mm)
    if not ret:
        logging.error("Encrypt failed.")
