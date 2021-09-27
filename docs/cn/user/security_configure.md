# vega 安全配置

## 评估服务器
### 评估服务器 https 安全配置
待补充
### 评估服务器 其他安全配置建议
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

#### 评估服务器配置访问频率
配置文件`.vega/vega.ini` 配置访问频率,默认限制每分钟最大100次访问
```ini
[limit]
request_frequency_limit=5/minute # 配置为每分钟最大5次访问
```

#### 评估服务器配置请求大小限制
配置文件`.vega/vega.ini` 配置请求大小限制，可以控制上传文件大小，默认配置 1G
```ini
[limit]
max_content_length=100000 # 配置请求大小最大100K 
```

