# -*- coding:utf-8 -*-
"""Setuptools of vega."""
import setuptools
import sys


if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported.")


setuptools.setup(
    name="vega",
    version="0.9.3",
    packages=["vega"],
    include_package_data=True,
    python_requires=">=3.6",
    author="Huawei Noah's Ark Lab",
    author_email="",
    description="AutoML Toolkit",
    license="MIT",
    url="https://github.com/huawei-noah/vega",
)
