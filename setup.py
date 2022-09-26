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

"""Setuptools of vega."""
import sys
import setuptools


if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported.")


with open("RELEASE.md", "r") as fh:
    long_desc = fh.read()


setuptools.setup(
    name="noah-vega",
    version="1.8.5",
    packages=["vega"],
    include_package_data=True,
    python_requires=">=3.6",
    author="Huawei Noah's Ark Lab",
    author_email="",
    description="AutoML Toolkit",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/huawei-noah/vega",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "thop",
        "psutil",
        "pillow",
        "pyzmq",
        "pandas",
        "distributed",
        "click",
        "PyYAML",
        "numpy",
        "scipy",
        "dill",
        "scikit-learn",
        "opencv-python",
    ],
    entry_points="""
        [console_scripts]
        vega=vega.tools.run_pipeline:main
        vega-inference=vega.tools.inference:main
        vega-inference-det=vega.tools.detection_inference:main
        vega-kill=vega.tools.kill:main
        vega-progress=vega.tools.query_progress:main
        vega-process=vega.tools.query_process:main
        vega-encrypt_key=vega.security.kmc.encrypt_key:main
    """,
)
