# -*- coding: utf-8 -*-

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

"""Setuptools of evaluate service."""

import sys
import os
import shutil
import setuptools
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported.")


with open("RELEASE.md", "r") as fh:
    long_desc = fh.read()


def _copy_security_folder():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(cur_path, "../vega/security")
    dst_path = os.path.join(cur_path, "evaluate_service/security")
    if not os.path.exists(dst_path):
        shutil.copytree(src_path, dst_path)


class custom_build_py(build_py):
    """Custom build_py."""

    def run(self):
        """Copy security folder before run."""
        _copy_security_folder()
        setuptools.command.build_py.build_py.run(self)


class custom_sdist(sdist):
    """Custom sdist."""

    def run(self):
        """Copy security folder before run."""
        _copy_security_folder()
        setuptools.command.sdist.sdist.run(self)


setuptools.setup(
    name="evaluate-service",
    version="1.8.5",
    packages=["evaluate_service"],
    include_package_data=True,
    python_requires=">=3.6",
    author="Huawei Noah's Ark Lab",
    author_email="",
    description="AutoML Toolkit",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/huawei-noah/vega/evaluate-service",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "Flask-RESTful",
        "Flask-Limiter",
        "gevent",
        "PyYAML",
    ],
    cmdclass={
        "build_py": custom_build_py,
        "sdist": custom_sdist,
    },
    entry_points="""
        [console_scripts]
        vega-evaluate-service=evaluate_service.main:run
        vega-encrypt_key=evaluate_service.security.kmc.encrypt_key:main
    """,
)
