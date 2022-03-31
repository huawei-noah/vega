# VEGA security configuration

The security configuration of the Vega includes the following steps:

1. Install OpenSSL
2. Generate the CA root certificate
3. Generate the certificate for evaluate_services
4. Generate the certificate for dask
5. Encrypt the private key password
6. Configure security-related configuration files
7. Configure the evaluation service daemon service
8. Install dask and distributed
9. Configuring the HCCL trustlist
10. Precautions

## 1. Install OpenSSL

You need to install OpenSSL 1.1.1, compile and install from the source code, or directly install the compiled release package.

Install the Python interface of the OpenSSL as follows:

```shell
pip3 install --user pyOpenSSL==19.0.0
```

## 2. Generate the CA Certificate

Run the following command to generate a CA certificate:
Note: The length of the RSA key must be 3072 bits or more. The following RSA key length configuration also requires the same.

```shell
openssl genrsa -out ca.key 4096
openssl req -new -x509 -key ca.key -out ca.crt -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
```

Note:

1. `<country>`, `<province>`, `<city>`, `<organization>`, `<group>`, and `<cn>` should be set based on the situation. The values do not contain `< >'. In addition, the CA configuration must be different from other configurations.
2. It is recommended that the length of the RSA key be 3072 bits or more.

## 3. Generate the Certificate for Evaluate_service

The evaluation service supports encryption certificates and common certificates.

1. If an encryption certificate is used, install Huawei KMC security components. For details, see section "Generating an Encryption Certificate."
2. If a common certificate is used, see section "Generating a Common Certificate."

### 3.1 Generating the Encryption Certificate

Run the following commands to generate the encryption private key for the server of evaluate_service. When you run this command, the system prompts you to enter the encryption password. The password strength requirements are as follows:

1. The password contains at least eight characters.
2. The value must contain at least one uppercase letter.
3. The value must contain at least one lowercase letter.
4. The value must contain at least one digit.

```shell
openssl genrsa -aes-256-ofb -out server.key 4096
```

Run the following commands to generate a certificate and delete the temporary file:

```shell
openssl req -new -key server.key -out server.csr -extensions v3_ca -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt
rm server.csr
```

Run the following commands to generate the encryption private key of the certificate for the client of evaluate_service. When you run this command, the system prompts you to enter the encryption password. The password strength must be the same as that of the server private key and is different from that of the server private key. Record the password and use it later.

```shell
openssl genrsa -aes-256-ofb -out client.key 4096
```

Run the following commands to generate a certificate and delete the temporary file:

```shell
openssl req -new -key client.key -out client.csr -extensions v3_ca -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt
rm client.csr
```

### 3.2 Generating the Common Certificate

Run the following commands to generate the private key and certificate for server and client of evaluate_service:

```shell
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr -extensions v3_ca -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt
rm server.csr

openssl genrsa -out client.key 4096
openssl req -new -key client.key -out client.csr -extensions v3_ca -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt
rm client.csr
```

## 4. Generate the Certificate for Dask

Run the following commands to generate the private key and certificate for server and client of dask:

```shell
openssl genrsa -out server_dask.key 4096
openssl req -new -key server_dask.key -out server_dask.csr -extensions v3_ca -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in server_dask.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server_dask.crt
rm server_dask.csr

openssl genrsa -out client_dask.key 4096
openssl req -new -key client_dask.key -out client_dask.csr -extensions v3_ca -subj "/C=<country>/ST=<province>/L=<city>/O=<organization>/OU=<group>/CN=<cn>"
openssl x509 -req -in client_dask.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client_dask.crt
rm client_dask.csr
```

Run the following command to delete the CA private key:

```shell
rm ca.key
```

## 5. Encrypting the Private Key Password

If the encryption certificate is used, perform the rest of this section. If the common certificate is used, skip this section.

To encrypt the private key passwords of the server and client for evaluate_service, you need to install Huawei KMC security component and add the directory where the dynamic link library of the security component is located to `LD_LIBRARY_PATH`.

```shell
export LD_LIBRARY_PATH=<Directory where the KMC dynamic link library is located>:$LD_LIBRARY_PATH
```

Install Vega and use the password encryption tool to encrypt the password.
When running the following command, enter the password entered during private key generation. This command will generate an encrypted password. Save the two encrypted passwords in the configuration file:

```shell
vega-encrypt_key --cert=server.crt --key=server.key --key_component_1=ksmaster_server.dat --key_component_2=ksstandby_server.dat
vega-encrypt_key --cert=client.crt --key=client.key --key_component_1=ksmaster_client.dat --key_component_2=ksstandby_client.dat
```

## 6.Configure Security-related Configuration Files

Create the `.vega` directory in the home directory of the current user, copy the generated keys, certificates, and encryption materials to this directory, and change the permission.

```shell
mkdir ~/.vega
mv * ~/.vega/
chmod 600 ~/.vega/*
```

Description:

1. The preceding keys, certificates, and encryption materials can also be stored in other directories. The access permission must be set to 600, and the file location must be changed to an absolute path in subsequent configuration files.
2. In the train cluster, reserve `ca.crt`, `client.key`, `client.crt`, `ksmaster_client.dat`, `ksstandby_client.dat`, and `server_dask.key`, `server_dask.crt`, `client_dask.key`, `client_dask.crt`, and delete other files.
3. In the evaluate service, reserve `ca.crt`, `server.key`, `server.crt`, `ksmaster_server.dat`, and `ksstandby_server.dat` files, and delete other files.

Create `server.ini` and `client.ini` in the `~/.vega` directory.

In the train cluster, configure `~/.vega/server.ini` and `~/.vega/client.ini`.

server.ini:

```ini
[security]  # The following file paths need to be changed to absolute paths.
    ca_cert=<~/.vega/ca.crt>
    server_cert_dask=<~/.vega/server_dask.crt>
    server_secret_key_dask=<~/.vega/server_dask.key>
    client_cert_dask=<~/.vega/client_dask.crt>
    client_secret_key_dask=<~/.vega/ client_dask.key>
```

client.ini:

```ini
[security]  # The following file paths need to be changed to absolute paths.
    ca_cert=<~/.vega/ca.crt>
    client_cert=<~/.vega/client.crt>
    client_secret_key=<~/.vega/client.key>
    encrypted_password=<Encrypted client password>  # If a common certificate is used, leave this parameter blank.
    key_component_1=<~/.vega/ksmaster_client.dat>  # If a common certificate is used, leave this parameter blank.
    key_component_2=<~/.vega/ksstandby_client.dat>  # If a common certificate is used, leave this parameter blank.
```

On the evaluation server, configure `~/.vega/vega.ini`.

```ini
[security]   # The following file paths need to be changed to absolute paths.
    ca_cert=<~/.vega/ca.crt>
    server_cert=<~/.vega/server.crt>
    server_secret_key=<~/.vega/server.key>
    encrypted_password=<Encrypted server password>  # If a common certificate is used, leave this parameter blank.
    key_component_1=<~/.vega/ksmaster_server.dat>  # uses a common certificate, leave this parameter blank.
    key_component_2=<~/.vega/ksstandby_server.dat>  # uses a common certificate, leave this parameter blank.
```

## 7. Configuring the Evaluation Service Daemon Service

The systemctl is used to manage the evaluation server process. When the process is abnormal, the systemctl automatically restarts to ensure the continuity of the evaluation server.

Create a script `run_evaluate_service.sh` for starting the evaluation service. Replace `<ip>` and `<path>` with the actual IP address and directory.

```shell
vega-evaluate_service-service -i <ip> -w <path>
```

Create a daemon service file `evaluate-service.service`. The script content is as follows. Replace it with the actual script location.

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

Copy `evaluate-service.service` to the `/usr/lib/systemd/system` directory and start the service.

```shell
sudo cp evaluate-service.service /usr/lib/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start evaluate-service
```

## 8. Install Dask and Distributed

When Vega is installed, the latest versions of the Dashboard and Distributed are automatically installed. In the current version, a bug exists when the Dashboard is disabled in Distributed. You need to run the following commands to install the two components of the following versions:

```shell
pip3 install --user dask==2.11.0
pip3 install --user distributed==2.11.0
```

## 9. Configuring the HCCL Trustlist

For details, see the [Configuration Guide](https://support.huawei.com/enterprise/en/doc/EDOC1100206669/8e964064) provided by the Ascend.

## 10. Precautions

### 10.1 Model Risks

For an AI framework, a model is a program. A model may read and write files and send network data. For example, TensorFlow provides the local operation API tf.read_file, tf.write_file. The return value is an operation that can be directly executed by TensorFlow.
Therefore, exercise caution when using a model with unknown sources. Before using the model, check whether malicious operations exist in the model to eliminate security risks.

### 10.2 Risks of Running Scripts

The script_runner function provided by Vega can invoke external scripts to perform hyperparameter optimization. Check the script source and ensure that no malicious operation exists. Exercise caution when running scripts from unknown sources.

### 10.3 Do Not Use KMC Components By Different Users At The Same Time

If the KMC component is used to encrypt the private key password, note that different users cannot use the KMC component at the same time.
To switch user, run the following command as the root user to query the current semaphore:

```bash
ipcs
```

Run the following command to delete all the semaphores:

```bash
ipcrm -S '<semaphore>'
```
