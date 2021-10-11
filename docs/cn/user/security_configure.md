# vega 安全配置
## 用户数据保护
用户用于训练的模型脚本/文件、预训练模型以及数据集属于比较重要的数据文件，需要做好安全保护，可以通过设置正确的文件权限来提升其安全性。可以通过如下命令来设置正确的文件权限
```shell
chmod 640 -R "file_path"
```

## 安全配置文件
vega在启动时会尝试读取```~/.vega/vega.ini```配置文件中的内容，如果该文件不存在或者文件中的配置不正确，那么vega会报错并自动退出。

用户在安装vega之后，可以通过命令```vega-security-config -i```初始化该文件，初始化之后该文件内容如下：
```ini
[security]
enable = True

[https]
cert_pem_file =
secret_key_file =
```
```[security] -> enable```的默认配置为True，此时用户还需要配置```[https]```段落下的```cert_pem_file```与```secret_key_file。```关于如何生成这2个文件请参考下面的章节，生成文件之后用户可以直接编辑vega.ini配置这2项内容，也可以通过如下命令来配置
```shell
vega-security-config -m https -c "cert_file_path" -k "key_file_path"
# 替换“cert_file_path”与“key_file_path"为真实的文件路径
```

> 注意：用户也可以选择关闭安全配置，通过运行命令```vega-security-config -s 0```来实现。关闭安全配置之后，训练服务器与推理服务器之间的通信将不再使用https而是https协议，无法保证通信安全。
> 
> 用户在关闭安全配置后，可以通过命令```vega-security-config -s 1```来重新开启安全配置。
> 

vega-security-config提供的操作vega.ini文件的命令总览如下：
```shell
# 1. 初始化vega.ini文件
vega-security-config -i
# 2. 关闭安全配置
vega-security-config -s 0
# 3. 打开安全配置
vega-security-config -s 1
# 4. 查询当前的安全配置开关是否打开
vega-security-config -q sec
# 5. 查询https的证书与密钥配置
vega-security-config -q https
# 6. 配置https的证书与密钥文件路径
vega-security-config -m https -c "cert_file_path" -k "key_file_path"
# 7. 只配置https的证书路径(在训练服务器上)
vega-security-config -m https -c "cert_file_path"
```

## 评估服务器
### 评估服务器 https 安全配置
#### 生成评估服务器密钥和证书

在评估服务器上执行以下操作

1.将/etc/pki/tls/openssl.cnf或者/etc/ssl/openssl.cnf拷贝到当前文件夹

2.修改当前目录下的openssl.cnf文件内容，在[ v3_ca ]段落中添加内容
```ini
subjectAltName = IP:xx.xx.xx.xx
```
> 注意：xx.xx.xx.xx修改为推理服务器的IP地址
> 
3.生成服务器密钥
```shell
openssl genrsa -aes-256-ofb -out example_key.pem 4096
```
> 注意：在这个阶段需要用户输入保护密钥的密码，此密码由用户自己记住，并且输入的密码强度需满足需求，具体的密码强度需求见下面的启动评估服务器章节
> 
4.生成证书请求文件
```shell
openssl req -new -key example_key.pem -out example.csr -extensions v3_ca \
-config openssl.cnf
```
5.生成自签名证书
```shell
openssl x509 -req -days 365 -in example.csr -signkey example_key.pem \
-out example_crt.pem -extensions v3_ca -extfile openssl.cnf
```
6.设置密钥/证书权限
为了确保系统安全，需要正确配置密钥/证书文件的权限，用户可以使用如下命令进行配置
```shell
sudo chmod 600 example_key.pem example_crt.pem
```

#### 评估服务器配置https密钥和证书
将example_key.pem和example_crt.pem拷贝到```~/.vega```文件夹下

修改配置文件`~/.vega/vega.ini` 配置密钥和证书
```ini
[security]
enable = True  # 需要配置成True才能启用https加密通信

[https]
cert_pem_file = /home/<username>/.vega/example_crt.pem  # 修改username和证书文件名
secret_key_file = /home/<username>/.vega/example_key.pem  # 修改username和密钥文件名
```


#### 评估服务器配置访问频率
配置文件`~/.vega/vega.ini` 配置访问频率,默认限制每分钟最大100次访问
```ini
[limit]
request_frequency_limit=5/minute # 配置为每分钟最大5次访问
```

#### 评估服务器配置请求大小限制
配置文件`~/.vega/vega.ini` 配置请求大小限制，可以控制上传文件大小，默认配置 1G
```ini
[limit]
max_content_length=100000 # 配置请求大小最大100K 
```

#### 评估服务器配置白名单，仅可信的服务器连接评估服务器
1. linux 白名单配置
    * 配置白名单：
        ```
        sudo iptables -I INPUT -p tcp --dport 评估端口 -j DROP
        sudo iptables -I INPUT -s 白名单IP地址1 -p tcp --dport 评估端口 -j ACCEPT
        sudo iptables -I INPUT -s 白名单IP地址2 -p tcp --dport 评估端口 -j ACCEPT
        sudo iptables -I INPUT -s 白名单IP地址3 -p tcp --dport 评估端口 -j ACCEPT
        sudo iptables -I INPUT -s 白名单IP地址4 -p tcp --dport 评估端口 -j ACCEPT
       ```
    * 如果需要从白名单中删除某一项
        1. 查询白名单 ```sudo iptables -L -n --line-number```
        2. 删除白名单 ```sudo iptables -D INPUT 查询的对应行编号```

2. 配置文件 `.vega/vega.ini` 配置白名单
    * 在配置中的 limit.white_list 中配置白名单，用逗号分隔
    ```ini
    [limit]
    white_list=127.0.0.1,10.174.183.95
    ```

#### 启动评估服务器
在配置了以上安全配置项之后，用户用以下命令启动评估服务器

```vega-evaluate_service-service -i {your_ip_adress} -w {your_work_path}```

其中`-i`参数指定当前使用的服务器的ip地址， 
`-w`参数指定工作路径， 程序运行时的中间文件将存储在该目录下，请使用绝对路径。 其他可选参数的设置可查看该命令的帮助信息， 一般情况下建议采用默认值。

在评估服务器启动时需要用户输入服务器密钥对应的密码（在生成密钥时输入的密码），系统会检查用户密码的强度，如果密码强度不符合需求，将会提示用户并自动退出。密码强度要求如下：
```
1. 密码长度大于等于8位
2. 必须包含至少1位大写字母
3. 必须包含至少1位小写字母
4. 必须包含至少1位数字
```

## 训练服务器
### 训练服务器安全配置
训练服务器需要配置推理服务器的证书信息，才能正常向推理服务器发送请求进行推理。用户可以按照如下方法进行配置：

修改配置文件`~/.vega/vega.ini` 配置密钥和证书
```ini
[security]
enable = True  # 需要配置成True才能启用https加密通信

[https]
cert_pem_file = /home/<username>/.vega/example_crt.pem  # 修改username和证书文件名
```
> 注意：这里的example_crt.pem为上面的步骤中生成的证书文件，用户需要手动将该证书文件拷贝到训练节点的对应目录下。

### 训练服务器防火墙设置
训练节点在进行多卡训练时需要启动dask和zmq服务，这些服务会随机监听本地127.0.0.1的27000 - 34000 端口。为了保护用户的服务不被恶意攻击，可以通过如下方式配置防火墙保护这些端口:

```shell
iptables -I OUTPUT -p tcp -m owner --uid-owner "user_id" -d 127.0.0.1 --match multiport --dports 27000:34000 -j ACCEPT
iptables -A OUTPUT -p tcp --match multiport -d 127.0.0.1 --dports 27000:34000 -j DROP
```
其中```"user_id"```需要用户执行命令```id "username"```查看用户的id并镜像替换。
> 注意：该配置限制了所有其他用户对端口27000-34000的访问，在多用户环境下如果其他用户也需要运行vega训练任务，需要使用其他用户的id去运行第一条命令，以便使该用户添加到防火墙的白名单中。
> 

