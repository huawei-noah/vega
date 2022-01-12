# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class of list dict."""

import os
from collections import OrderedDict
import pandas as pd


class ListDict:
    """Class of list dict.

    :param data: data
    :type data: list
    """

    def __init__(self, data=None, **kwargs):
        if data is None:
            data = []
        self.data = data
        self.kwargs = kwargs

    def __len__(self):
        """Get the length of data."""
        return len(self.data)

    def __getitem__(self, key: (int, slice, str, tuple, list)):
        """Get item."""
        if isinstance(key, str):
            return [p[key] for p in self.data]
        elif isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return self.__class__(data=self.data[key], **self.kwargs)
        elif isinstance(key, (tuple, list)):
            records = []
            for key_ in key:
                records.append(self[key_])
            if isinstance(records[-1], (dict, OrderedDict)):
                return self.__class__(data=records, **self.kwargs)
            else:
                return list(zip(*records))
        else:
            raise TypeError('Key must be str or list')

    def __str__(self):
        """Str."""
        s = []
        for i in self.data:
            s.append(str(i))
        return '\n'.join(s)

    @property
    def header(self):
        """Get the header of the data."""
        if len(self.data) > 0:
            return list(self.data[0].keys())
        else:
            return None

    def get(self, key, default=None):
        """Get value for key."""
        try:
            return self[key]
        except BaseException:
            return default

    def append(self, data):
        """Append data."""
        if isinstance(data, ListDict):
            if len(data) != 0:
                raise Exception('data len must be 0')
            data = data.data[0]
        if isinstance(data, (dict, OrderedDict)):
            self.data.append(data)
        else:
            raise TypeError(
                'Method append does support for type {}'.format(
                    type(data)))

    def extend(self, data):
        """Extend data."""
        if isinstance(data, ListDict):
            data = data.data
        if isinstance(data, list):
            self.data.extend(data)
        else:
            raise TypeError(
                'Method extend does support for type {}'.format(
                    type(data)))

    def insert(self, idx, data):
        """Insert an item."""
        if isinstance(data, ListDict):
            if len(data) != 0:
                raise Exception('data len must be 0')
            data = data.data[0]
        if isinstance(data, (dict, OrderedDict)):
            self.data.insert(idx, data)
        else:
            raise TypeError(
                'Method insert does support for type {}'.format(
                    type(data)))

    def pop(self, idx):
        """Pop an item."""
        return self.data.pop(idx)

    def to_dataframe(self):
        """Dump to DataFrame."""
        return pd.DataFrame(self.data)

    def to_csv(self, path, index=False, **kwargs):
        """Dump to csv file."""
        df = self.to_dataframe()
        df.to_csv(path, columns=self.header, index=index, **kwargs)

    @classmethod
    def load_csv(cls, path, **kwargs):
        """Load csv file."""
        if not os.path.isfile(path):
            raise FileExistsError('{} does not exist.'.format(path))
        df = pd.read_csv(path)
        data = df.to_dict('records')
        return cls(data=data, **kwargs)
