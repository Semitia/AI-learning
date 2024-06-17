### .ipynb->.py
```bash
jupyter nbconvert --to script your_notebook.ipynb
```
在相同目录下生成一个名为 your_notebook.py 的文件。

### tensorboard 常用配置
```bash
tensorboard --logdir=logs/ant_bee
```