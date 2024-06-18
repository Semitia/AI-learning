# ML


## Links
实现了巨多的深度学习模型：
https://github.com/lucidrains  
一行行进行代码解读：
https://nn.labml.ai/
小土堆
https://www.bilibili.com/video/BV1hE411t7RN/

## Env Configuration
### ValueError: check_hostname requires server_hostname
使用的clash打开system proxy后就会出现这个报错，但是如果没有加速的话，下载异常缓慢。所以给clash指定代理端口，可以避免错误并提高速度。
``` bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --proxy http://127.0.0.1:7890
```
也有可能遇到`ReadTimeoutError`，可以尝试增加超时时间
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --proxy http://127.0.0.1:7890 --timeout 1000
```
