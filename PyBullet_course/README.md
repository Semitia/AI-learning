# PyBullet Notes
## 引言
### 目标：
- 机械臂工作台场景搭建
- 机械臂反馈与控制
- 物体状态与交互
- 机械臂夹取物体


## 开发环境搭建
#### 安装 MicroSoft C++ Build Tools
> https://blog.csdn.net/Oona_01/article/details/139567000
> 
> https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/

之后重启

#### 安装PyBullet
```bash
 pip install pybullet==3.2.5
```
指定安装3.2.5版本的，3.2.6似乎有问题

## 基础概念
- pybullet服务器常用操作
- URDF基本语法
- Transform
- 机器人结构
![img.png](img.png)
- 关节初步控制
- 机械臂运动学
- 碰撞检测
https://blog.csdn.net/weixin_44350205/article/details/109705590
- 抓取任务

### 强化学习示例
#### 环境搭建
##### 安装stable-baselines3
```bash
pip install stable-baselines3
```
##### 安装gym
```bash
pip install gym
```
不过Gym已经停止维护，为了适配stable-baselines3，需要安装兼容库

```bash
pip install 'shimmy>=2.0'
```
PyBullet适配Gym版本<=0.21
需要手动修改一下PyBullet的库文件（或者新建一个虚拟环境，感觉也比较麻烦）

```python
# Lib/site-packages/pybullet_envs/__init__.py
# 第一处
if id in registry.env_specs:
# 修改为
if id in gym.envs.registry:
    
# 第二处
btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
# 修改后
btenvs = ['- ' + env_id for env_id in gym.envs.registry.keys() if 'Bullet' in env_id]
```
#### 常见参数解释
```bash
-----------------------------------
| eval/               |           |
|    mean_ep_length   | 1e+03     |
|    mean_reward      | -4.65e+03 |
| rollout/            |           |
|    exploration_rate | 0.02      |
| time/               |           |
|    total_timesteps  | 110000    |
| train/              |           |
|    learning_rate    | 0.001     |
|    loss             | 0.0145    |
|    n_updates        | 27474     |
-----------------------------------
```
- mean_ep_length：平均episode长度
- mean_reward：平均reward
- exploration_rate：探索率
- total_timesteps：总训练步数
- learning_rate：学习率
- loss：损失
- n_updates：网络参数更新次数
