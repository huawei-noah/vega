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
import time
from distributed import Client
from vega.common import General


def get_client(address):
    """Get dask client."""
    if not General.security:
        return Client(address)
    else:
        from vega.security.run_dask import get_client_security
        return get_client_security(address)


def get_address(master_host, master_port):
    """Get master address."""
    if not General.security:
        return "tcp://{}:{}".format(master_host, master_port)
    else:
        from vega.security.run_dask import get_address_security
        return get_address_security(master_host, master_port)


def run_scheduler(ip, port, tmp_file):
    """Run dask-scheduler."""
    if not General.security:
        return subprocess.Popen(
            [
                "dask-scheduler",
                "--no-dashboard",
                "--no-show",
                f"--host={ip}",
                f"--port={port}",
                f"--scheduler-file={tmp_file}",
            ],
            env=os.environ
        )
    else:
        from vega.security.run_dask import run_scheduler_security
        return run_scheduler_security(ip, port, tmp_file)


def run_local_worker(slave_ip, address, local_dir):
    """Run dask-worker on local node."""
    if not General.security:
        return subprocess.Popen(
            [
                "dask-worker",
                address,
                '--nthreads=1',
                '--nprocs=1',
                '--memory-limit=0',
                "--no-dashboard",
                f"--local-directory={local_dir}",
                f"--host={slave_ip}",
            ],
            env=os.environ
        )
    else:
        from vega.security.run_dask import run_local_worker_security
        time.sleep(1)
        return run_local_worker_security(slave_ip, address, local_dir)


def run_remote_worker(slave_ip, address, local_dir):
    """Run dask-worker on remove node."""
    if not General.security:
        id = subprocess.Popen(
            [
                "ssh",
                slave_ip,
                shutil.which("dask-worker"),
                address,
                '--nthreads=1',
                '--nprocs=1',
                '--memory-limit=0',
                "--no-dashboard",
                f"--local-directory={local_dir}",
                f"--host={slave_ip}",
            ],
            env=os.environ
        )
        return id
    else:
        from vega.security.run_dask import run_remote_worker_security
        time.sleep(1)
        return run_remote_worker_security(slave_ip, address, local_dir)
