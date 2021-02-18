# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Uni comm."""

import threading
from absl import logging
from zeus.common.util.register import Registers


class UniComm(object):
    """Uni comm."""

    def __init__(self, comm_name, **comm_info):
        super(UniComm, self).__init__()
        self.comm = Registers.comm[comm_name](comm_info)
        self.lock = threading.Lock()

    def send(self, data, name=None, block=True, **kwargs):
        """Create common send interface."""
        return self.comm.send(data, name, block, **kwargs)

    def recv(self, name=None, block=True):
        """Create common recieve interface."""
        return self.comm.recv(name, block)

    def send_bytes(self, data):
        """Create common send_bytes interface."""
        return self.comm.send_bytes(data)

    def recv_bytes(self):
        """Create common recv_bytes interface."""
        return self.comm.recv_bytes()

    def send_multipart(self, data):
        """Create common send_multipart interface."""
        return self.comm.send_multipart(data)

    def recv_multipart(self):
        """Create common recv_multipart interface."""
        return self.comm.recv_multipart()

    def delete(self, name):
        """Delete."""
        return self.comm.delete(name)

    @property
    def info(self):
        """Fetch comm info."""
        return str(self.comm)

    def close(self):
        """Close."""
        logging.debug("start close comm...")
        with self.lock:
            try:
                self.comm.close()
            except AttributeError as err:
                logging.info("call comm.close failed! with: \n{}".format(err))
