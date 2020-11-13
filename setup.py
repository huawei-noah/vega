# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Setuptools of vega."""
import setuptools
import sys


if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported.")


with open("RELEASE.md", "r") as fh:
    long_desc = fh.read()


setuptools.setup(
    name="noah-vega",
    version="1.2.0",
    packages=["vega", "zeus"],
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
        "PyYAML==5.1.2",
        "pandas==0.25.2",
        "pareto==1.1.1.post3",
        "scipy==1.3.3",
        "matplotlib==3.3.0",
        "py-dag==3.0.1",
        "psutil==5.6.3",
        "distributed==2.18.0",
    ],
)
