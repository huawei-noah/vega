# vega 安全配置

Vega的安全配置，包括如下步骤：

1. 安装OpenSSL
2. 生成CA根证书
3. 生成评估服务用的证书
4. 生成Dask用的证书
5. 加密私钥口令
6. 配置安全相关的配置文件
7. 配置评估服务守护服务
8. 安装dask和distributed
9. 配置HCCL白名单
10. 注意事项

## 1.安装OpenSSL

首先要安装OpenSSL 1.1.1，从源码编译安装，或者直接安装编译后的发行包。

然后安装OpenSSL的python接口，如下：

```shell
pip3 install --user pyOpenSSL==19.0.0
```

## 2.生成CA证书

执行如下命令生成CA证书：

```shell
openssl genrsa -out ca.key 4096 
openssl req -new -x509 -key ca.key -out ca.crt -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
```

注意：

1. 以上`<country>`、`<province>`、`<city>`、`<organization>`、`<group>`、`<cn>`根据实际情况填写，去掉符号`<>`，本文后面的配置也是同样的。并且CA的配置需要和其他的不同。
2. RSA密钥长度建议在3072位及以上，如本例中使用4096长度。

## 3. 生成评估服务使用的证书

评估服务支持加密证书和普通证书：

1. 若使用加密证书，需要安装华为公司的KMC安全组件，参考`生成加密证书`章节
2. 若使用普通证书，参考`生成普通证书`章节

### 3.1 生成加密证书

执行如下脚本，生成评估服务器所使用的证书的加密私钥，执行该命令时，会提示输入加密密码，密码的强度要求如下：

1. 密码长度大于等于8位
2. 必须包含至少1位大写字母
3. 必须包含至少1位小写字母
4. 必须包含至少1位数字

```shell
openssl genrsa -aes-256-ofb -out server.key 4096
```

然后再执行如下命令，生成证书，并删除临时文件：

```shell
openssl req -new -key server.key -out server.csr -extensions v3_ca -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt
rm server.csr
```

执行如下脚本生成评估服务客户端所使用的证书的加密私钥，执行该命令时，会提示输入加密密码，密码的强度要求如服务器端私钥，且和服务器段私钥密码不同，请记录好改密码，后继还需使用：

```shell
openssl genrsa -aes-256-ofb -out client.key 4096
```

然后再执行如下命令，生成证书，并删除临时文件：

```shell
openssl req -new -key client.key -out client.csr -extensions v3_ca -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt
rm client.csr
```

### 3.2 生成普通证书

执行如下脚本，生成评估服务器端和客户端使用的证书的私钥和证书：

```shell
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr -extensions v3_ca  -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt
rm server.csr

openssl genrsa -out client.key 4096
openssl req -new -key client.key -out client.csr -extensions v3_ca  -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt
rm client.csr
```

## 4. 生成Dask使用的证书

执行如下脚本，生成Dask服务器端和客户端使用的证书的私钥和证书：

```shell
openssl genrsa -out server_dask.key 4096
openssl req -new -key server_dask.key -out server_dask.csr -extensions v3_ca  -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in server_dask.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server_dask.crt
rm server_dask.csr

openssl genrsa -out client_dask.key 4096
openssl req -new -key client_dask.key -out client_dask.csr -extensions v3_ca  -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in client_dask.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client_dask.crt
rm client_dask.csr
```

删除CA私钥：

```shell
rm ca.key
```

## 5. 加密私钥口令

若加密服务器使用加密证书，则需要执行本章节余下步骤，若使用普通证书，则跳过该章节。

加密生成评估服务的服务器端和客户端的私钥口令，需要安装华为公司KMC安全组件，并将该安全组件动态链接库所在的目录添加到`LD_LIBRARY_PATH`中。

```shell
export LD_LIBRARY_PATH=<Directory where the KMC dynamic link library is located>:$LD_LIBRARY_PATH
```

接下来安装Vega，使用Vega的密码加密工具调用KMC安全组件对密码加密。
在执行如下命令时，请输入在生成私钥时输入的口令，该命令会生成加密后的口令，请注意保存，在配置文件中会使用到这两个加密后的口令：

```shell
vega-encrypt_key --cert=server.crt --key=server.key --key_component_1=ksmaster_server.dat --key_component_2=ksstandby_server.dat
vega-encrypt_key --cert=client.crt --key=client.key --key_component_1=ksmaster_client.dat --key_component_2=ksstandby_client.dat
```

## 6. 配置安全配置文件

请在当前用户的主目录下创建`.vega`目录，并将如上生成的秘钥、证书、加密材料等，拷贝到该目录下，并改变权限：

```shell
mkdir ~/.vega
mv * ~/.vega/
chmod 600 ~/.vega/*
```

说明：

1. 如上的秘钥、证书、加密材料也可以放到其他目录位置，注意访问权限要设置为`600`，并在后继的配置文件中同步修改该文件的位置，需要使用绝对路径。
2. 在训练集群上，需要保留`ca.crt`、`client.key`、`client.crt`、`ksmaster_client.dat`、`ksstandby_client.dat`、`server_dask.key`、`server_dask.crt`、`client_dask.key`、`client_dask.crt`，并删除其他文件。
3. 评估服务上，需要保留`ca.crt`、`server.key`、`server.crt`、`ksmaster_server.dat`、`ksstandby_server.dat`，并删除其他文件。

在`~/.vega`目录下创建`server.ini`和`client.ini`。

在训练集群中，需要配置`~/.vega/server.ini`和`~/.vega/client.ini`：

server.ini:

```ini
[security]  # 以下文件路径需要修改为绝对路径
    ca_cert=<~/.vega/ca.crt>
    server_cert_dask=<~/.vega/server_dask.crt>
    server_secret_key_dask=<~/.vega/server_dask.key>
    client_cert_dask=<~/.vega/client_dask.crt>
    client_secret_key_dask=<~/.vega/client_dask.key>
```

client.ini:

```ini
[security]  # 以下文件路径需要修改为绝对路径
    ca_cert=<~/.vega/ca.crt>
    client_cert=<~/.vega/client.crt>
    client_secret_key=<~/.vega/client.key>
    encrypted_password=<加密后的client端的口令>  # 如果使用普通证书， 此项配置为空
    key_component_1=<~/.vega/ksmaster_client.dat>  # 如果使用普通证书， 此项配置为空
    key_component_2=<~/.vega/ksstandby_client.dat>  # 如果使用普通证书， 此项配置为空
```

在评估服务器上，需要配置`~/.vega/vega.ini`：

```ini
[security]  # 以下文件路径需要修改为绝对路径
    ca_cert=<~/.vega/ca.crt>
    server_cert=<~/.vega/server.crt>
    server_secret_key=<~/.vega/server.key>
    encrypted_password=<加密后的server端的口令>  # 如果使用普通证书， 此项配置为空
    key_component_1=<~/.vega/ksmaster_server.dat>  # 如果使用普通证书， 此项配置为空
    key_component_2=<~/.vega/ksstandby_server.dat>  # 如果使用普通证书， 此项配置为空
```

## 7. 配置评估服务守护服务

使用systemctl管理评估服务器进程，当进程出现异常时自动重启，保证评估服务器连续性。

首先创建一个启动评估服务的脚本`run_evaluate_service.sh`，内容如下，注意替换`<ip>`、`<path>`为真实IP和目录：

```shell
vega-evaluate_service-service -i <ip> -w <path>
```

然后再创建一个守护服务的文件`evaluate-service.service`，脚本内容如下，注意替换为真实的脚本位置：

```ini
[Unit]
    Description=Vega Evaluate Service Daemon
[Service]
    Type=forking
    ExecStart=/<your_run_script_path>/run.sh
    Restart=always
    RestartSec=60
[Install]
    WantedBy=multi-user.target
```

然后将`evaluate-service.service`拷贝到目录`/usr/lib/systemd/system`中，并启动该服务：

```shell
sudo cp evaluate-service.service /usr/lib/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start evaluate-service
```

## 8. 安装Dask和distributed

安装Vega时，会自动安装Dask和Distributed的最新版本，我们发现在当前版本中Distributed关闭dash board时存在bug，需要执行如下命令，安装如下版本的这两个组件：

```shell
pip3 install --user dask==2.11.0
pip3 install --user distributed==2.11.0
```

## 9. 配置HCCL白名单

请参考Ascend提供的[配置指导](https://support.huawei.com/enterprise/zh/doc/EDOC1100206668/8e964064)。

## 10. 注意事项

### 10.1 模型风险

对于AI框架来说，模型就是程序，模型可能会读写文件、发送网络数据。例如Tensorflow提供了本地操作API tf.read_file, tf.write_file，返回值是一个operation，可以被Tensorflow直接执行。
因此对于未知来源的模型，请谨慎使用，使用前应该排查该模型是否存在恶意操作，消除安全隐患。

### 10.2 运行脚本风险

Vega提供的script_runner功能可以调用外部脚本进行超参优化，请确认脚本来源，确保不存在恶意操作，谨慎运行未知来源脚本。

### 10.3 KMC组件不支持多个用户同时使用

若使用KMC组件对私钥密码加密，需要注意KMC组件不支持不同的用户同时使用KMC组件。若需要切换用户，需要在root用户下，使用如下命令查询当前信号量：

```bash
ipcs
```

然后删除查询到的当前所有的信号量：

```bash
ipcrm -S '<信号量>'
```
