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
"""Share by plasma."""
import os
import time
from multiprocessing import Queue
from subprocess import PIPE, Popen

import lz4.frame
from pyarrow import deserialize, plasma, serialize

from zeus.common.util.register import Registers


@Registers.comm
class ShareByPlasma(object):
    """Share by plasma."""

    def __init__(self, comm_info):
        """Init plasma component."""
        super(ShareByPlasma, self).__init__()
        self.size_shared_mem = comm_info.get("size", 1000000000)
        self.path = comm_info.get("path", "/tmp/plasma" + str(os.getpid()))
        self.compress = comm_info.get("compress", True)

        self.control_q = Queue()
        self.client = {}
        self.start()

    def send(self, data, name=None, block=True, data_type="data"):
        """Send data to plasma server."""
        client = self.connect()
        if data_type == "data":
            data_buffer = lz4.frame.compress(serialize(data).to_buffer())
        else:
            data_buffer = serialize(data).to_buffer()

        object_id = client.put_raw_buffer(data_buffer)
        self.control_q.put((object_id, data_type))

        # del data
        if data["ctr_info"].get("cmd") == "train":
            keys = []
            for key in data["data"].keys():
                keys.append(key)
            for key in keys:
                del data["data"][key]
        elif data["ctr_info"].get("cmd") == "predict":
            del data["data"]

    def recv(self, name=None, block=True):
        """Receive data from plasma server."""
        object_id = self.control_q.get()
        client = self.connect()
        data = deserialize(lz4.frame.decompress(client.get_buffers([object_id])))
        client.delete([object_id])

        return data

    def send_bytes(self, data_buffer, data_type="data"):
        """Send data to plasma server without serialize."""
        client = self.connect()
        object_id = client.put_raw_buffer(data_buffer)
        self.control_q.put((object_id, data_type))

    def recv_bytes(self):
        """Receive data from plasma server without deserialize."""
        object_info = self.control_q.get()
        client = self.connect()
        object_id, _ = object_info

        data_buffer = client.get_buffers([object_id])
        # client.delete([object_id])

        return data_buffer[0], object_info

    def delete(self, object_id):
        """Delete."""
        client = self.connect()
        client.delete([object_id])

    def send_multipart(self, data_buffer):
        """Send multi-data to plasma server without serialize."""
        client = self.connect()
        self.control_q.put(len(data_buffer))
        for _buffer in data_buffer:
            objec_id = client.put_raw_buffer(_buffer)
            self.control_q.put(objec_id)

    def recv_multipart(self):
        """Recieve multi-data from plasma server without deserialize."""
        len_data = self.control_q.get()
        object_id = []
        client = self.connect()
        for _ in range(len_data):
            _object_id = self.control_q.get()
            object_id.append(_object_id)

        data_buffer = client.get_buffers(object_id)
        client.delete(object_id)

        return data_buffer

    def start(self):
        """Start plasma server."""
        try:
            plasma.connect(self.path, int_num_retries=2)
        except Exception:
            Popen(
                "plasma_store -m {} -s {}".format(self.size_shared_mem, self.path),
                shell=True,
                stderr=PIPE,
            )
            print(
                "plasma_store -m {} -s {} is acitvated!".format(
                    self.size_shared_mem, self.path
                )
            )
            time.sleep(0.1)

    def connect(self):
        """Connect to plasma server."""
        pid = os.getpid()
        if pid in self.client:
            return self.client[pid]
        else:
            self.client[pid] = plasma.connect(self.path)
            return self.client[pid]

    def close(self):
        """Close plasma server."""
        os.system("pkill -9 plasma")
