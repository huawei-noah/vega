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

"""Security config.

~/.vega/server.ini

[security]
    ca_cert=<~/.vega/car.crt>
    server_cert_dask=<~/.vega/server_dask.crt>
    server_secret_key_dask=<~/.vega/server_dask.key>
    client_cert_dask=<~/.vega/client_dask.crt>
    client_secret_key_dask=<~/.vega/ client_dask.key>

~/.vega/client.ini

[security]
    ca_cert=<~/.vega/car.crt>
    client_cert=<~/.vega/client.crt>
    client_secret_key=<~/.vega/client.key>
    encrypted_password=<encrypted client key password>
    key_component_1=<~/.vega/ksmaster_client.dat>
    key_component_2=<~/.vega/ksstandby_client.dat>

"""

import os
import logging
import configparser
from .verify_config import check_risky_files


class Config():
    """Security Config."""

    def load(self) -> bool:
        """Load from config file."""
        if not check_risky_files([self.file_name]):
            return False
        config = configparser.ConfigParser()
        try:
            config.read(self.file_name)
        except Exception:
            logging.error(f"Failed to read setting from {self.file_name}")
            return False
        if "security" not in config.sections():
            return False
        keys = []
        pass_check_keys = ["encrypted_password", "white_list", "ciphers"]
        for key in config["security"]:
            if key not in self.keys:
                return False
            setattr(self, key, config.get("security", key))
            if key not in pass_check_keys and not check_risky_files([config.get("security", key)]):
                return False
            keys.append(key)
        if len(keys) != len(self.keys):
            missing_keys = list(set(self.keys) - set(keys))
            if missing_keys != ["ciphers"]:
                logging.error(f"setting items {missing_keys} are missing in {self.file_name}")
                return False
        return True


class ServerConfig(Config):
    """Security Config."""

    def __init__(self):
        """Initialize."""
        self.ca_cert = None
        self.server_cert_dask = None
        self.server_secret_key_dask = None
        self.client_cert_dask = None
        self.client_secret_key_dask = None
        self.file_name = os.path.expanduser("~/.vega/server.ini")
        self.keys = ["ca_cert", "server_cert_dask", "server_secret_key_dask", "client_cert_dask",
                     "client_secret_key_dask"]


class ClientConfig(Config):
    """Security Config."""

    def __init__(self):
        """Initialize."""
        self.ca_cert = None
        self.client_cert = None
        self.client_secret_key = None
        self.encrypted_password = None
        self.key_component_1 = None
        self.key_component_2 = None
        self.ciphers = None
        self.white_list = []
        self.file_name = os.path.expanduser("~/.vega/client.ini")
        self.keys = [
            "ca_cert", "client_cert", "client_secret_key", "encrypted_password",
            "key_component_1", "key_component_2", "ciphers"]


_server_config = ServerConfig()
_client_config = ClientConfig()


def load_config(_type: str) -> bool:
    """Load security config."""
    if _type not in ["all", "server", "client"]:
        logging.error(f"not support security config type: {_type}")
        return False
    if _type in ["server", "all"]:
        global _server_config
        if not _server_config.load():
            logging.error("load server security config fail.")
            return False
    if _type in ["client", "all"]:
        global _client_config
        if not _client_config.load():
            logging.error("load client security config fail.")
            return False
    return True


def get_config(_type: str) -> Config:
    """Get config."""
    if _type not in ["server", "client"]:
        logging.error(f"not support security config type: {_type}")
        return False
    if _type == "server":
        return _server_config
    else:
        return _client_config
