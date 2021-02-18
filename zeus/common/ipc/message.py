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
"""Message."""
from zeus.common.util.register import Registers


def message(data, **kwargs):
    """Create message."""
    ctr_info = {"broker_id": -1, "explorer_id": -1, "agent_id": -1, "cmd": "train"}

    ctr_info.update(**kwargs)
    return {"data": data, "ctr_info": ctr_info}


def get_msg_info(msg, key):
    """Get message ctr info."""
    return msg["ctr_info"].get(key)


def set_msg_info(msg, **kwargs):
    """Set message ctr info."""
    msg["ctr_info"].update(**kwargs)


def get_msg_data(msg):
    """Get message data."""
    return msg["data"]


def set_msg_data(msg, data):
    """Set message data."""
    msg.update({"data": data})


@Registers.comm
class Message(object):
    """Message."""

    def __init__(self, data, **kwargs):
        """Initialize."""
        self.msg_data = data
        self.ctr_info = {
            "actor_id": -1,
            "explorer_id": -1,
            "agent_id": -1,
            "cmd": "train",
        }
        self.ctr_info.update(**kwargs)

    def set_ctr_info(self, **kwargs):
        """Set control info."""
        self.ctr_info.update(**kwargs)

    def get_cmd(self):
        """Get command."""
        return self.ctr_info.get("cmd")

    def get_explorer_id(self):
        """Get explorer id."""
        return self.ctr_info.get("explorer_id")

    def get_actor_id(self):
        """Get actor id."""
        return self.ctr_info.get("actor_id")

    def get_msg_data(self):
        """Get message data."""
        return {"ctr_info": self.ctr_info, "data": self.msg_data}

    @staticmethod
    def load(msg_data):
        """Load."""
        return Message(msg_data["data"], **msg_data["ctr_info"])
