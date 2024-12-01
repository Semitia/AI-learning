import gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv


def main():
    # 创建环境（渲染模式开启）
    env = KukaGymEnv(renders=True, isDiscrete=True)
    # 加载训练好的模型
    model = DQN.load("kuka_dqn_model", env=env)

    # 重置环境
    obs = env.reset()
    done = False
    total_reward = 0

    print("开始测试模型...")

    while not done:
        # 使用模型预测动作（确定性策略）
        action, _states = model.predict(obs, deterministic=True)

        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # 渲染效果（环境自动渲染，不需要手动调用 render）

    print(f"总奖励: {total_reward}")


if __name__ == "__main__":
    main()
