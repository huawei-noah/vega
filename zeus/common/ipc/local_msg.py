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
"""Local message."""
import queue
from zeus.common.util.register import Registers


@Registers.comm
class LocalMsg(object):
    """Create local message used for communication inner process."""

    def __init__(self, comm_info):
        """Initialize."""
        self.cmd_q = queue.Queue()
        self.data_list = list()
        self.msg_num = 0

    def send(self, data, name=None, block=True):
        """Send data."""
        self.data_list.append(data)

        self.cmd_q.put(self.msg_num, block=block)
        self.msg_num += 1
        # if "cur_state" not in data.get(0):
        #     raise NotImplementedError

    def recv(self, name=None, block=True):
        """Receive data."""
        self.cmd_q.get(block=block)
        data = self.data_list.pop(0)
        # print("locl msg", msg_num, data)

        return data
