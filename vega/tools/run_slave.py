# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Run dask worker on slave."""

import os
import time
import subprocess


def run_dask_worker(master_ip, port, num_workers):
    """Run dask worker on slave."""
    success = 0
    interval = 3    # sleep 3s
    for _ in range(60 * 60 // interval):
        try:
            subprocess.Popen(
                ["dask-worker", f"{master_ip}:{port}", '--nthreads=1', '--nprocs=1', '--memory-limit=0'],
                env=os.environ)
            success += 1
            if success == num_workers:
                break
        except Exception as e:
            print(f"Failed to start dask-worker ({e}), try again {interval}s later.")
            time.sleep(interval)
    if success != num_workers:
        raise Exception("Failed to start dask-worker. Gave up.")
    else:
        print("dask-worker running.")


if __name__ == "__main__":
    run_dask_worker()
