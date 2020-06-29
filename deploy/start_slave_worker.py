# -*- coding: utf-8 -*-
"""Deploy slave worker."""
import sys
import subprocess


def start_worker(master_ip, master_port):
    """Start worker on slave nodes."""
    try:
        subprocess.call(["dask-worker", "{}:{}".format(master_ip, master_port)])
    except Exception:
        raise ValueError("Can't start subprocess")


if __name__ == "__main__":
    start_worker(sys.argv[1], sys.argv[2])
