# -*- coding: utf-8 -*-
"""The config of the evaluate service."""
ip_address = "192.168.0.1"
listen_port = 8888
optional_params = {
    "davinci_environment_type": "ATLAS300",
    # if your environment_type is ATLAS200DK, the following parameters should be configed, if not, just ignore it
    "ddk_user_name": "user",
    "atlas_host_ip": "192.168.0.2"
}
clean_interval = 1 * 24 * 3600  # one day
