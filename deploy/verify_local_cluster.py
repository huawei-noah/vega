# -*- coding: utf-8 -*-
"""Verify local cluster."""
import sys
from dask.distributed import Client


def verify(master_ip):
    """Verify cluster."""
    try:
        Client("{}".format(master_ip))
    except Exception:
        raise ValueError("Client can't running")


if __name__ == "__main__":
    verify(sys.argv[1])
