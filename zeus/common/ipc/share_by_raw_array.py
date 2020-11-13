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
"""Share by raw array."""
from ctypes import addressof, c_ubyte, memmove
from multiprocessing import Queue, RawArray
import pyarrow

import lz4.frame

from zeus.common.util.register import Registers


@Registers.comm
class ShareByRawArray(object):
    """Share by raw array."""

    def __init__(self, comm_info):
        """Initilize shared memory."""
        super(ShareByRawArray, self).__init__()

        self.size_shared_mem = comm_info.get("size", 100000000)
        self.agent_num = comm_info.get("agent_num", 4)

        self.control_q = Queue()
        self.mem = RawArray(c_ubyte, self.size_shared_mem)
        self.size_mem_agent = int(self.size_shared_mem / self.agent_num)

    def send(self, data, name=None, block=True):
        """Put data in share memory."""
        data_id, data = data
        msg = lz4.frame.compress(pyarrow.serialize(data).to_buffer())

        memmove(addressof(self.mem) + int(data_id *
                                          self.size_mem_agent), msg, len(msg))

        self.control_q.put((data_id, len(msg)))

    def recv(self, name=None):
        """Get data from share memory."""
        data_id, len_data = self.control_q.get()

        data = pyarrow.deserialize(
            lz4.frame.decompress(
                memoryview(self.mem)[
                    int(data_id * self.size_mem_agent): int(
                        data_id * self.size_mem_agent + len_data
                    )
                ]
            )
        )

        return data

    def recv_bytes(self):
        """Get data from share memory without deserialize."""
        data_id, len_data = self.control_q.get()

        return memoryview(self.mem)[
            int(data_id * self.size_mem_agent): int(
                data_id * self.size_mem_agent + len_data
            )
        ]

    def send_bytes(self, data):
        """Put data in share memory without serialize."""
        data_id, data_buffer = data
        memmove(
            addressof(self.mem) + int(data_id) * self.size_mem_agent,
            data_buffer,
            len(data_buffer),
        )

        self.control_q.put((data_id, len(data_buffer)))

    def send_multipart(self, data):
        """Put multi-data in share memory without serialize."""
        data_id, data_buffer = data
        self.control_q.put(len(data_buffer))
        for _id, _buffer in zip(data_id, data_buffer):
            memmove(
                addressof(self.mem) + int(_id) * self.size_mem_agent,
                _buffer,
                len(_buffer),
            )
            self.control_q.put((_id, len(_buffer)))

    def recv_multipart(self):
        """Get multi-data from share memory without deserialize."""
        len_data = self.control_q.get()
        data_id = []
        data_buffer = []
        for _ in range(len_data):
            _id, len_buff = self.control_q.get()
            data_id.append(_id)
            data_buffer.append(
                memoryview(self.mem)[
                    int(_id * self.size_mem_agent): int(
                        _id * self.size_mem_agent + len_buff
                    )
                ]
            )

        return data_buffer

    def close(self):
        """Close."""
        pass
