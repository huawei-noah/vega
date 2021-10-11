# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run dask scheduler and worker."""
import os
import subprocess
import shutil
from distributed import Client
from vega.common.utils import get_available_port


def get_client(address):
    """Get dask client."""
    return Client(address)


def get_address(master_host, master_port):
    """Get master address."""
    return "tcp://{}:{}".format(master_host, master_port)


def run_scheduler(port):
    """Run scheduler."""
    dashboard_port = get_available_port(min_port=30000, max_port=30999)
    """Run dask-scheduler."""
    return subprocess.Popen(
        [
            "dask-scheduler",
            ""
            "--no-dashboard",
            "--no-show",
            "--host=127.0.0.1",
            port,
            f"--dashboard-address={dashboard_port}"
        ],
        env=os.environ
    )


def run_local_worker(address, local_dir):
    """Run dask-worker on local node."""
    work_port = get_available_port(min_port=31000, max_port=31999)
    dashboard_address = get_available_port(min_port=33000, max_port=33999)
    return subprocess.Popen(
        [
            "dask-worker",
            address,
            '--nthreads=1',
            '--nprocs=1',
            '--memory-limit=0',
            local_dir,
            "--no-dashboard",
            f'--listen-address=tcp://127.0.0.1:{work_port}',
            '--nanny-port=32000:32999',
            f'--dashboard-address={dashboard_address}'
        ],
        env=os.environ
    )


def run_remote_worker(slave_ip, address, local_dir):
    """Run dask-worker on remote node."""
    return subprocess.Popen(
        [
            "ssh",
            slave_ip,
            shutil.which("dask-worker"),
            address,
            '--nthreads=1',
            '--nprocs=1',
            '--memory-limit=0',
            local_dir
        ],
        env=os.environ
    )
