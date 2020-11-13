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
"""Share buffer."""
import os
import time
from absl import logging
from multiprocessing import Queue
from subprocess import PIPE, Popen
import numpy as np
from pyarrow import deserialize, plasma, serialize


class ShareBuf(object):
    """Share Buffer among broker salve."""

    def __init__(self, live, size=200000000, max_keep=20,
                 path="/tmp/plasma_share{}".format(os.getpid()), start=False):
        """Init buffer share with plasma."""
        super(ShareBuf, self).__init__()
        self.size_shared_mem = size
        self.path = path
        self.max_keep = max_keep

        self.client = dict()
        if start:
            self.start()
        else:
            time.sleep(0.1)

        # add 1 vary with once get
        self.live_info = dict()
        # ref explorer number under this broker
        self.live_threshold = live
        # vanish after register with half live times
        self.to_vanish = Queue()
        self._update_vanish_attr(live)

    def plus_one_live(self):
        """Add one live value."""
        self.live_threshold += 1
        self._update_vanish_attr(self.live_threshold)

    def update_live(self, live):
        """Update the live attribute. Without check history Yet."""
        self.live_threshold = live
        self._update_vanish_attr(live)

    def _update_vanish_attr(self, live):
        self._vanish_threshold = min(max(7, live // 2), 13)

    def get_path(self):
        """Get plasma path."""
        return self.path

    def _pre_to_vanish_obj(self):
        """Check live management information, Return non_lived object ids."""
        to_vanish_ids = list()
        for _id, _val in self.live_info.items():
            if _val > 0:
                continue

            to_vanish_ids.append(_id)

        for _id in to_vanish_ids:
            del self.live_info[_id]

        return to_vanish_ids

    def _get_over_keep_ids(self):
        """Get over keep object ids."""
        over_ids = list()
        client = self.connect()
        id_list = client.list()
        sorted_id = sorted(id_list.items(), key=lambda v: v[1]["create_time"],
                           reverse=True)

        over_ids.extend(_val[0] for _val in sorted_id[self.max_keep:])
        logging.debug("over_ids: {}".format(over_ids))
        return over_ids

    def _get_vanish_obj(self):
        """Get ready vanish object_ids."""
        to_ids = self._pre_to_vanish_obj()
        for _id in to_ids:
            self.to_vanish.put(_id)

        ready_vanish_ids = set()
        while self.to_vanish.qsize() > self._vanish_threshold:
            obj_id = self.to_vanish.get()
            ready_vanish_ids.add(plasma.ObjectID(bytes(obj_id)))

        # check store_capacity.
        over_ids = self._get_over_keep_ids()
        ready_vanish_ids.update(over_ids)

        # we re-delete the key in live_info
        for _oid in over_ids:
            if _oid.binary() in self.live_info:
                del self.live_info[_oid.binary()]

        return list(ready_vanish_ids)

    def _init_obj(self, object_id, special_live=None):
        """Put New object."""
        self.live_info.update({object_id: special_live or self.live_threshold})

    def reduce_once(self, object_id):
        """Reduce one times of this object."""
        if object_id not in self.live_info:
            logging.debug("obj_id: {} is deleted yet".format(object_id))
        else:
            self.live_info[object_id] -= 1

    def put(self, data_buffer, special_live=None):
        """Put data buffer for share."""
        client = self.connect()
        object_id = client.put_raw_buffer(data_buffer)
        logging.debug("put buffer with id: {}".format(object_id))
        self._init_obj(object_id.binary(), special_live)

        # del data within the vanish Queue
        ready_vanish_ids = self._get_vanish_obj()
        if ready_vanish_ids:
            logging.debug("delete: {}, vanish queue.size: {}, odd: {}".format(
                ready_vanish_ids, self.to_vanish.qsize(), len(self.live_info)))
            client.delete(ready_vanish_ids)
        else:
            logging.debug("odd:{}".format(len(self.live_info)))
        # print("list plasma: ", client.list())
        return object_id.binary()

    def _get_buf(self, obj_id, retry=5):
        object_id = plasma.ObjectID(bytes(obj_id))
        logging.debug("get buffer: {}".format(object_id))
        buf = None
        # may instability.
        for _t in range(retry):
            try:
                client = self.connect()
                buf = client.get_buffers([object_id], timeout_ms=10)[0]
            except BaseException as error:
                logging.info("try-{} to get buffer except: {}".format(_t, error))
                pid = os.getpid()
                del self.client[pid]
                continue
            else:
                break

        data = deserialize(buf) if buf else {"data": None}
        # data = deserialize(client.get_buffers([object_id])[0])
        return data

    def get(self, object_id_byte):
        """Get a object data from plasma server with id."""
        return self._get_buf(object_id_byte)

    def get_with_live_consume(self, object_id_byte):
        """Get a object data from plasma server with id, and reduce live count."""
        data = self._get_buf(object_id_byte)

        self.reduce_once(object_id_byte)
        return data

    def delete(self, object_id):
        """Delete a object within plasma."""
        client = self.connect()
        client.delete([object_id])

    def start(self):
        """Start plasma server."""
        try:
            client = plasma.connect(self.path, int_num_retries=3)
        except:
            Popen(
                "plasma_store -m {} -s {}".format(self.size_shared_mem, self.path),
                shell=True,
                stderr=PIPE,
            )
            logging.info(
                "Share buf: plasma_store -m {} -s {} is activated!".format(
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
            # logging.debug("self.path", self.path)
            return self.client[pid]

    def close(self):
        """Close plasma server."""
        os.system("pkill -9 plasma")


def test_buf_get_live():
    """Test share buf live count."""
    live_count = 10
    logging.set_verbosity(logging.DEBUG)
    share_buf = ShareBuf(live=live_count, size=20000000, start=True)
    data = {"d{}".format(i): np.array(np.arange(i)) for i in range(5, 8)}
    ds = serialize(data).to_buffer()
    b_id = share_buf.put(data_buffer=ds)
    for _ in range(live_count // 2):
        ret = share_buf.get_with_live_consume(b_id)
        print(share_buf.live_info)

    b_id2 = share_buf.put(data_buffer=ds)

    for _ in range(live_count // 2):
        ret = share_buf.get_with_live_consume(b_id)
        print(share_buf.live_info)
    assert share_buf.live_info[b_id] < 1

    # to remove the first one item
    b_id3 = share_buf.put(data_buffer=ds)
    assert b_id not in share_buf.live_info
    assert share_buf.live_info[b_id2] == live_count
    # print(share_buf.live_info)


def test_share_buf_io():
    """Test share buf io-out."""
    logging.set_verbosity(logging.DEBUG)
    share_buf = ShareBuf(live=10, size=20000000, start=True)
    data = {"d{}".format(i): np.array(np.arange(i)) for i in range(5, 8)}
    print(data)
    ds = serialize(data).to_buffer()
    b_id = share_buf.put(data_buffer=ds)
    print("b_id", b_id)
    ret = share_buf.get(b_id)
    print("ret:", ret)
    check_equal_dict(data, ret)


def check_equal_dict(d1: dict, d2: dict):
    """Check dict if equal."""
    assert d1.keys() == d2.keys()
    for _k, val in d1.items():
        if isinstance(val, np.ndarray):
            assert (val == d2[_k]).all(), "{} vs {}".format(val, d2[_k])
        else:
            assert val == d2[_k], "{} vs {}".format(val, d2[_k])


if __name__ == "__main__":
    test_share_buf_io()
    test_buf_get_live()
