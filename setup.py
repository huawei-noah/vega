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
    version="1.6.0",
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
        "py-dag",
        "pareto",
        "thop",
        "psutil",
        "pillow",
        "pyzmq",
        "tf-slim",
        "pandas==0.25.2",
        "distributed==2.18.0",
        "click==7.1.2",
        "PyYAML==5.1.2",
        "numpy==1.18.5",
        "scipy==1.5.3",
        "scikit-learn==0.21.3",
        "opencv-python-headless==4.3.0.38",
        "tensorboardX==1.9",
        "tf-models-official==0.0.3.dev1",
        "torch==1.3.0",
        "torchvision==0.4.1",
        "tensorflow-gpu>=1.14.0,<2.0",
        # "onnx-simplifier"
    ],
    entry_points="""
        [console_scripts]
        vega=vega.tools.run_pipeline:run_pipeline
        vega-kill=vega.tools.kill:_kill
        vega-verify-cluster=vega.tools.verify_cluster:_verify_cluster
        vega-fine-tune=vega.tools.fine_tune:_fine_tune
        vega-progress=vega.tools.query_progress:print_progress
        vega-process=vega.tools.query_process:print_processes
        vega-evaluate-service=evaluate_service.main:run
      """,
)
