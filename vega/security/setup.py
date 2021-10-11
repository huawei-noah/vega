# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Setuptools of vega."""

import os
import setuptools
import sys
from setuptools.command.install import install as _install

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported.")

with open("RELEASE.md", "r") as fh:
    long_desc = fh.read()


def _post_install():
    vega_dir = os.path.join(os.getenv("HOME"), ".vega")
    os.makedirs(vega_dir, exist_ok=True)
    vega_config_file = os.path.join(vega_dir, "vega.ini")
    if os.path.exists(vega_config_file):
        return

    with open(vega_config_file, "w") as wf:
        wf.write("[security]\n")
        wf.write("enable=True\n")
        wf.write("\n")
        wf.write("[https]\n")
        wf.write("cert_pem_file=\n")
        wf.write("secret_key_file=\n")
        wf.write("\n")
        wf.write("[limit]\n")
        wf.write("request_frequency_limit=100/minute\n")
        wf.write("max_content_length=1000000000\n")
        wf.write("#white_list=0.0.0.0,127.0.0.1\n")


class install(_install):
    """Post installation."""

    def run(self):
        """Run."""
        _install.run(self)
        self.execute(_post_install, (), msg="Running pre install task")


cmd_class = dict(install=install)

setuptools.setup(
    name="noah-vega",
    cmdclass=cmd_class,
    version="1.8.0.mindstudio",
    packages=["vega", "evaluate_service"],
    include_package_data=True,
    python_requires=">=3.6",
    author="Huawei Noah's Ark Lab",
    author_email="",
    description="AutoML Toolkit",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/huawei-noah/vega",
    # packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "pyzmq",
    ],
    entry_points="""
        [console_scripts]
        vega=vega.tools.run_pipeline:run_pipeline
        vega-security-config=vega.tools.config_op:vega_config_operate
        vega-kill=vega.tools.kill:_kill
        vega-verify-cluster=vega.tools.verify_cluster:_verify_cluster
        vega-fine-tune=vega.tools.fine_tune:_fine_tune
        vega-progress=vega.tools.query_progress:print_progress
        vega-process=vega.tools.query_process:print_processes
        vega-evaluate-service=evaluate_service.main:run
      """,
)
