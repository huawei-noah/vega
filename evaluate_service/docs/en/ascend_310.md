# Deploy the Ascend environment.

Deploy the Ascend environment by referring to the Ascend official document. The following installation guide 
is a key step during the installation. If an error occurs during the installation, refer to the official document.
Before the deployment, download the installation package from the official website.

## 1 Checking the Installed Driver and CANN Versions

For a new Ascend host, check whether the `/usr/local/HiAi` directory exists. If yes, 
run the following command as user root to uninstall the directory:

```bash
/usr/local/HiAi/uninstall.sh
```

Run the following commands as a non-root user to create the `Ascend` directory and make the
directory accessible to the `HwHiAiUser` user:

```bash
mkdir /usr/local/Ascend/
sudo chown -R :HwHiAiUser /usr/local/Ascend/
sudo chmod -R 750 /usr/local/Ascend/
```

If `/usr/local/Ascend/` exists, check if the old Driver and CANN packages have been installed 
before the installation. 
Run the following command to query the version number of each component:

```bash
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/nnae/latest/ascend_nnae_install.info
cat /usr/local/Ascend/ascend-toolkit/latest/arm64-linux/ascend_toolkit_install.info
cat /usr/local/Ascend/nnrt/latest/arm64-linux/ascend_nnrt_install.info
cat /usr/local/Ascend/tfplugin/latest/ascend_tfplugin_install.info
```


If the version is older than expected, uninstall it as by root user.

```bash
/usr/local/Ascend/driver/script/uninstall.sh
/usr/local/Ascend/nnae/latest/script/uninstall.sh
/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/script/uninstall.sh
/usr/local/Ascend/nnrt/latest/arm64-linux/script/uninstall.sh
/usr/local/Ascend/tfplugin/latest/script/uninstall.sh
```

If the platform is x86, replace `arm64-linux` in the directory contained in the preceding command with `x86_64-linux`.

If nnae, ascend-toolkit, nnrt, and tfplugin are not installed by the root user, uninstall them as the user.

## 2 Installing the Driver and CANN

Run the following command as the root user to install the software. The following version number is for reference only:

```bash
chmod +x *.run
./A300-3000-3010-npu-driver_21.0.2_linux-aarch64.run --full
```

Run the following command to check whether the installation is successful:

```bash
npu-smi info
```

Before installing other packages as a non-root user, set this user to the same group as `HwHiAiUser`.

```bash
usermod -a -G HwHiAiUser <username>
` ` `

```bash
./Ascend-cann-nnae_5.0.T306_linux-aarch64.run --install
./Ascend-cann-nnrt_5.0.T306_linux-aarch64.run --install
./Ascend-cann-tfplugin_5.0.T306_linux-aarch64.run --install
./Ascend-cann-toolkit_5.0.T306_linux-aarch64.run --install
```

After the installation is complete, restart the host as prompted.

## 3 Setting Environment Variables

Set the following environment variables:

```bash
export ASCEND_HOME=/usr/local/Ascend
export HOME_DIR=/home/<username>
export PATH=$HOME_DIR/.local/bin:$PATH
source /usr/local/Ascend/nnae/set_env.sh
source /usr/local/Ascend/nnrt/set_env.sh
source /usr/local/Ascend/tfplugin/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/atc/lib64
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$PYTHONPATH
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/fwkacllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME_DIR/.local/lib/python3.7/site-packages/evaluate_service/security:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME_DIR/.local/lib/python3.7/site-packages/evaluate_service/security/kmc/aarch64:$LD_LIBRARY_PATH
```

In the preceding command, `<username>` indicates the user directory, 
and `$NPU_HOST_LIB` indicates the path of `libascendcl.so`. 
Set this variable based on the actual location of `libascendcl.so`.