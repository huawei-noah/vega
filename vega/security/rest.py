# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Rest operation."""

import requests
from vega.common import General


def post(host, files, data):
    """Post a REST requstion."""
    if General.security_setting.get("security").get("enable"):
        pem_file = General.security_setting.get("https").get("cert_pem_file")
        if not pem_file:
            print("CERT file ({}) is not existed.".format(pem_file))
        result = requests.post(host, files=files, data=data, proxies={"https": None}, verify=pem_file)
    else:
        result = requests.post(host, files=files, data=data, proxies={"http": None})
    data = result.json()
    return data
