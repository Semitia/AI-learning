from KukaGraspEnv import KukaGraspEnv
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


def main():
    # 创建环境
    env = KukaGraspEnv(renders=False, is_discrete=True)
    policy_kwargs = dict(net_arch=[64])  # 单隐藏层，64 个神经元
    # DQN 模型
    # model = DQN("MlpPolicy",
    #             env,
    #             learning_rate=1e-3,
    #             buffer_size=50000,
    #             exploration_fraction=0.1,
    #             exploration_final_eps=0.02,
    #             verbose=1,
    #             policy_kwargs=policy_kwargs)
    # model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
    #             gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, verbose=1)
    # SAC
    policy_kwargs = dict(net_arch=[256, 256])  # 两层，每层 256 个神经元
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=32,
        gamma=0.99,
        tau=0.005,
        gradient_steps=1,
        train_freq=1,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    # 评估回调
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./logs2/',  # 保存最佳模型的路径
        log_path='./logs2/',              # 日志路径
        eval_freq=25000,                  # 每 25000 步评估一次
        deterministic=True,               # 使用确定性动作进行评估
        render=False                      # 不渲染环境
    )

    # 开始训练模型
    model.learn(total_timesteps=1500000, callback=eval_callback)
    # 保存最终模型
    model.save("kuka_dqn_model2")
    print("Training completed and model saved.")


if __name__ == '__main__':
    main()
