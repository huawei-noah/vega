# Installation

## 1. Preparations Before Installation

The host where the Vega is installed has a GPU and meets the following requirements:

1. Ubuntu 18.04 or EulerOS 2.0 SP8
2. CUDA 10.0 or CANN 20.1
3. Python 3.7
4. pip3

## 2. Installing Vega

Execute the following command to install:

```bash
pip3 install --user --upgrade noah-vega
```

**Important:**

1. Before installing the VGA, run the following command to upgrade the PIP:

    ```bash
    pip3 install --user --upgrade pip
    ```

2. After the installation is complete, if `~/.local/bin` is not in the `$PATH` environment variable, you need to log in again to make the environment variable take effect.

If the training is performed on the Atlas 900, please contact us.
