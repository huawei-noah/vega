# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""
Classes for Share Memory.

Distributor Base Class, Dask Distributor Class and local Evaluator Distributor
Class. Distributor Classes are used in Master to init and maintain the cluster.
"""
import ast
import logging
from zeus.common.general import General
from zeus.common.utils import singleton


class ShareMemory(object):
    """Share Memory base class."""

    def __new__(cls, name):
        """Construct method."""
        if General._parallel:
            t_cls = ClusterShareMemory
        else:
            t_cls = LocalShareMemory
        return super(ShareMemory, cls).__new__(t_cls)


@singleton
class ShareMemoryClient():
    """Dask Client for Share Memory."""

    def __init__(self):
        if General.env.init_method is None:
            raise ValueError("Cluster master ip in Share Memory can not be None.")
        address = General.env.init_method
        logging.debug("Start Cluster Share Memory, address=%s", address)
        from dask.distributed import Client
        self._client = Client(address=address)
        logging.debug("Create Variable Cluster Share Memory, address=%s", address)

    @property
    def client(self):
        """Get Client instance."""
        return self._client

    def close(self):
        """Close Client."""
        if hasattr(self, "_client") and self._client:
            try:
                self._client.close()
            except Exception as ex:
                logging.warning("Close Share Memory client error, ex=%s", ex)
            finally:
                self._client = None


class ClusterShareMemory(ShareMemory):
    """Share Memory for dask cluster."""

    def __init__(self, name):
        from dask.distributed import Variable
        self.var = Variable(name, client=ShareMemoryClient().client)

    def put(self, value):
        """Put value into shared data."""
        self.var.set(str(value))

    def get(self):
        """Get value from shared data."""
        # TODO: block issue when var no data.
        return ast.literal_eval(self.var.get(timeout=2))

    def delete(self):
        """Delete data according to name."""
        self.var.delete()

    def close(self):
        """Close Share Memory."""
        ShareMemoryClient().close()


class LocalShareMemory(ShareMemory):
    """Local Share Memory."""

    __shared_data__ = {}

    def __init__(self, name):
        self.name = name

    def put(self, value):
        """Put value into shared data."""
        self.__shared_data__[self.name] = value

    def get(self):
        """Get value from shared data."""
        return self.__shared_data__.get(self.name)

    def delete(self):
        """Delete data according to name."""
        del self.__shared_data__[self.name]

    def close(self):
        """Close Share Memory."""
        pass
