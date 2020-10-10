# -*- coding:utf-8 -*-
"""Setuptools of vega."""
import setuptools
import sys


if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported.")


with open("RELEASE.md", "r") as fh:
    long_desc = fh.read()


setuptools.setup(
    name="noah-vega",
    version="1.0.0",
    packages=["vega"],
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
        "Operating System :: OS Independent",
    ],
)
