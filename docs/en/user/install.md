# Installation

## 1. Preparations Before Installation

The host where the Vega is installed has a GPU and meets the following requirements:

1. Ubuntu 18.04 or later (other Linux distributions are not fully tested)
2. CUDA 10.0
3. Python 3.7
4. pip

Before installing Vega, you need to install MMDetection and pycocotools in addition to some mandatory software packages through pip.

## 2. Installing Vega

Execute the following command to install:

```bash
pip3 install --user noah-vega
```

Vega's dependent packages are not installed during the Vega installation process. After completing the Vega installation, you need to execute the following commands to install the dependent packages:

```bash
python3 -m vega.tools.install_pkgs
```

## 3. Installing the MMDetection

If the detection algorithm (SP-NAS) is required, install MMDetection and download `mmdetection-1.0rc1.zip`. Install the software according to the installation instructions on the home page of the software.

## 4. Mixing precision and SyncBNN are supported.

To support mixed precision and SyncBN, install Apex, download the Apex source code `apex-master.zip`, and install the software according to the installation instructions on the home page of the software.
