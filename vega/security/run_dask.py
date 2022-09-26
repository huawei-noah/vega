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

"""Run dask scheduler and worker."""
import os
import subprocess
import shutil
import logging
import socket
import random
from distributed import Client
from distributed.security import Security
from .conf import get_config
from .verify_cert import verify_cert
import ssl

sec_cfg = get_config('server')


def get_client_security(address):
    """Get client."""
    address = address.replace("tcp", "tls")
    if not verify_cert(sec_cfg.ca_cert, sec_cfg.client_cert_dask):
        logging.error(f"The cert {sec_cfg.ca_cert} and {sec_cfg.client_cert_dask} are invalid, please check.")
    sec = Security(tls_ca_file=sec_cfg.ca_cert,
                   tls_client_cert=sec_cfg.client_cert_dask,
                   tls_client_key=sec_cfg.client_secret_key_dask,
                   tls_min_version=ssl.TLSVersion.TLSv1_3,
                   require_encryption=True)
    return Client(address, security=sec)


def get_address_security(master_host, master_port):
    """Get address."""
    return "tls://{}:{}".format(master_host, master_port)


def run_scheduler_security(ip, port, tmp_file):
    """Run scheduler."""
    if not verify_cert(sec_cfg.ca_cert, sec_cfg.server_cert_dask):
        logging.error(f"The cert {sec_cfg.ca_cert} and {sec_cfg.server_cert_dask} are invalid, please check.")
    dashboard_port = _available_port(31000, 31999)
    return subprocess.Popen(
        [
            "dask-scheduler",
            "--no-dashboard",
            "--no-show",
            f"--dashboard-address=127.0.0.1:{dashboard_port}",
            f"--tls-ca-file={sec_cfg.ca_cert}",
            f"--tls-cert={sec_cfg.server_cert_dask}",
            f"--tls-key={sec_cfg.server_secret_key_dask}",
            f"--host={ip}",
            "--protocol=tls",
            f"--port={port}",
            f"--scheduler-file={tmp_file}",
        ],
        env=os.environ
    )


def _available_port(min_port, max_port) -> int:
    _sock = socket.socket()
    while True:
        port = random.randint(min_port, max_port)
        try:
            _sock.bind(('', port))
            _sock.close()
            return port
        except Exception:
            logging.debug('Failed to get available port, continue.')
            continue
    return None


def run_local_worker_security(slave_ip, address, local_dir):
    """Run dask-worker on local node."""
    address = address.replace("tcp", "tls")
    nanny_port = _available_port(30000, 30999)
    worker_port = _available_port(29000, 29999)
    dashboard_port = _available_port(31000, 31999)
    pid = subprocess.Popen(
        [
            "dask-worker",
            address,
            '--nthreads=1',
            '--nprocs=1',
            '--memory-limit=0',
            f"--dashboard-address=127.0.0.1:{dashboard_port}",
            f"--local-directory={local_dir}",
            f"--tls-ca-file={sec_cfg.ca_cert}",
            f"--tls-cert={sec_cfg.client_cert_dask}",
            f"--tls-key={sec_cfg.client_secret_key_dask}",
            "--no-dashboard",
            f"--host={slave_ip}",
            "--protocol=tls",
            f"--nanny-port={nanny_port}",
            f"--worker-port={worker_port}",
        ],
        env=os.environ
    )
    return pid


def run_remote_worker_security(slave_ip, address, local_dir):
    """Run dask-worker on remote node."""
    address = address.replace("tcp", "tls")
    nanny_port = _available_port(30000, 30999)
    worker_port = _available_port(29000, 29999)
    dashboard_port = _available_port(31000, 31999)
    pid = subprocess.Popen(
        [
            "ssh",
            slave_ip,
            shutil.which("dask-worker"),
            address,
            '--nthreads=1',
            '--nprocs=1',
            '--memory-limit=0',
            f"--dashboard-address=127.0.0.1:{dashboard_port}",
            f"--local-directory={local_dir}",
            f"--tls-ca-file={sec_cfg.ca_cert}",
            f"--tls-cert={sec_cfg.client_cert_dask}",
            f"--tls-key={sec_cfg.client_secret_key_dask}",
            "--no-dashboard",
            f"--host={slave_ip}",
            "--protocol=tls",
            f"--nanny-port={nanny_port}",
            f"--worker-port={worker_port}",
        ],
        env=os.environ
    )
    return pid
