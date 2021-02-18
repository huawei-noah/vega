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
"""Share by redis."""
import time
from subprocess import Popen
import pyarrow
import redis
from zeus.common.util.register import Registers


@Registers.comm
class ShareByRedis(object):
    """Share by redis."""

    def __init__(self, comm_info):
        """Initilize redis component."""
        super(ShareByRedis, self).__init__()
        # For master, there is no 'addr' parameter given.
        self.ip_addr = comm_info.get("addr", "127.0.0.1")
        self.port = comm_info.get("port", 6379)
        self.password = comm_info.get("password", None)
        self.strat_redis = False
        if self.ip_addr == "127.0.0.1":
            self.start()

        self.redis = redis.Redis(host=self.ip_addr, port=self.port, db=0)

    def send(self, data, name=None, block=True):
        """Send data to redis server."""
        data_buffer = pyarrow.serialize(data).to_buffer()
        self.redis.set(name, data_buffer)

    def recv(self, name=None):
        """Recieve data from redis server."""
        data_buffer = self.redis.get(name)
        data = pyarrow.deserialize(data_buffer)
        return data

    def delete(self, name):
        """Delete items in redis server."""
        self.redis.delete(name)

    def start(self):
        """Start redis server."""
        try:
            redis.Redis(host=self.ip_addr, port=self.port, db=0).ping()
        except redis.ConnectionError:
            Popen("echo save '' | setsid redis-server -", shell=True)
            self.strat_redis = True
            time.sleep(0.1)

    def close(self):
        """Shutdown redis client."""
        # self.redis.flushdb()
        self.redis.shutdown(nosave=True)
        print("redis shutdown")
