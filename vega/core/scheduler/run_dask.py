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


def get_client(address):
    """Get dask client."""
    return Client(address)


def get_address(master_host, master_port):
    """Get master address."""
    return "tcp://{}:{}".format(master_host, master_port)


def run_scheduler(port):
    """Run dask-scheduler."""
    id = subprocess.Popen(
        ["dask-scheduler", "--no-dashboard", "--no-show", port],
        env=os.environ
    )
    return id


def run_local_worker(address, local_dir):
    """Run dask-worker on local."""
    id = subprocess.Popen(
        [
            "dask-worker",
            address,
            '--nthreads=1',
            '--nprocs=1',
            '--memory-limit=0',
            local_dir],
        env=os.environ
    )
    return id


def run_remote_worker(slave_ip, address, local_dir):
    """Run dask-worker on remove node."""
    id = subprocess.Popen(
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
    return id
