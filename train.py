"""
UAV-RIS 系统训练脚本
使用 PPO 算法训练 UAV 轨迹优化模型
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime

from main import UAVEnv

class TrainingCallback(BaseCallback):
    """训练过程中的回调函数，用于记录和打印训练进度"""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 每 check_freq 步打印一次信息
        if self.n_calls % self.check_freq == 0:
            if len(self.locals.get('infos', [])) > 0:
                info = self.locals['infos'][0]
                if 'total_time' in info:
                    print(f"Steps: {self.num_timesteps}, "
                          f"Delay: {info['total_time']:.4f}s, "
                          f"Jain: {info['Jain_step']:.4f}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """在每个 rollout 结束时调用"""
        if self.verbose > 0 and self.num_timesteps % 10000 < 2048:
            print(f"\n=== Rollout {self.num_timesteps // 2048} completed ===")


def train_model(
    total_timesteps: int = 200000,
    save_path: str = "./models",
    model_name: str = "ppo_uav_improved"
):
    """
    训练 PPO 模型
    
    Args:
        total_timesteps: 总训练步数
        save_path: 模型保存路径
        model_name: 模型名称
    """
    print("=" * 80)
    print("开始训练 UAV-RIS 轨迹优化模型")
    print("=" * 80)
    
    # 创建环境
    print("\n1. 创建训练环境...")
    env = UAVEnv()
    print(f"   ✓ 环境创建成功")
    print(f"   观察空间: {env.observation_space.shape}")
    print(f"   动作空间: {env.action_space}")
    
    # 创建模型
    print("\n2. 创建 PPO 模型...")
    import torch
    
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu"
    )
    print(f"   ✓ 模型创建成功")
    
    # 创建回调
    print("\n3. 开始训练...")
    callback = TrainingCallback(check_freq=2048, verbose=1)
    
    # 训练模型
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 保存模型
    print("\n4. 保存模型...")
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{model_name}_{timestamp}"
    model_file = os.path.join(save_path, full_model_name)
    model.save(model_file)
    print(f"   ✓ 模型已保存到: {model_file}.zip")
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    
    # 简单测试
    print("\n5. 快速测试模型...")
    obs, _ = env.reset()
    total_reward = 0
    
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"   测试总奖励: {total_reward:.2f}")
    print(f"   最终平均时延: {info['total_time']:.4f}s")
    print(f"   最终 Jain 指数: {info['Jain_step']:.4f}")
    
    return model_file + ".zip"


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 开始训练
    model_path = train_model(
        total_timesteps=200000,  # 可以调整训练步数
        save_path="./models",
        model_name="ppo_uav_improved"
    )
    
    print(f"\n训练完成！模型保存在: {model_path}")
    print(f"\n可以使用以下命令测试模型:")
    print(f"  python model_evaluation.py --model {model_path}")
