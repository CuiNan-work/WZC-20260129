"""
UAV-RIS 卸载系统 - 模型测试与可视化
测试训练好的 PPO 模型，绘制：
1. UAV 轨迹图
2. 时延变化曲线
3. UAV 负载均衡曲线
4. 性能指标对比
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stable_baselines3 import PPO
import os
from datetime import datetime
import gymnasium as gym

# 导入训练时的环境和参数
# 注意：需要根据你的实际训练脚本文件名修改
try:
    from main import UAVEnv, num_uavs, num_users, ris_pos, uav_H
except ImportError:
    print("错误：无法导入训练环境！")
    print("请确保训练脚本文件名正确，并修改上面的导入语句")
    print("例如：from your_training_file import UAVEnv, num_uavs, num_users, ris_pos, uav_H")
    exit(1)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModelTester:
    def __init__(self, model_path: str, num_episodes: int = 5):
        """
        初始化测试器
        Args:
            model_path: 训练好的模型路径
            num_episodes: 测试的 episode 数量
        """
        self.model_path = model_path
        self.num_episodes = num_episodes

        # 先创建环境
        print("创建测试环境...")
        self.env = UAVEnv()

        # 加载模型（提供环境信息）
        print(f"从 {model_path} 加载模型...")
        try:
            # 方法1：使用 custom_objects（推荐）
            custom_objects = {
                "observation_space": self.env.observation_space,
                "action_space": self.env.action_space,
            }
            self.model = PPO.load(model_path, env=self.env, custom_objects=custom_objects)
            print("✓ 模型加载成功！")
        except Exception as e1:
            print(f"方法1失败: {e1}")
            try:
                # 方法2：只提供环境
                self.model = PPO.load(model_path, env=self.env)
                print("✓ 模型加载成功！(方法2)")
            except Exception as e2:
                print(f"✗ 模型加载失败: {e2}")
                print("\n尝试方法3：重新创建模型并加载权重...")
                try:
                    # 方法3：重新创建模型结构
                    import torch
                    policy_kwargs = dict(
                        net_arch=dict(pi=[256, 256], vf=[256, 256]),
                        activation_fn=torch.nn.ReLU
                    )

                    self.model = PPO(
                        "MlpPolicy",
                        self.env,
                        policy_kwargs=policy_kwargs,
                        device="cpu"
                    )

                    # 加载权重
                    self.model.set_parameters(model_path)
                    print("✓ 模型加载成功！(方法3)")
                except Exception as e3:
                    print(f"✗ 所有加载方法都失败了")
                    print(f"错误信息: {e3}")
                    print("\n请检查:")
                    print("1. 模型文件是否存在？")
                    print("2. 模型是否是用相同版本的 stable-baselines3 训练的？")
                    print("3. 环境定义是否与训练时一致？")
                    exit(1)

        # 存储测试数据
        self.test_data = {
            'uav_trajectories': [],  # 每个episode的UAV轨迹
            'user_positions': [],  # 每个episode的用户位置
            'total_delays': [],  # 每个episode每步的总时延
            'comm_delays': [],  # 通信时延
            'comp_delays': [],  # 计算时延
            'uav_loads': [],  # UAV负载
            'jain_indices': [],  # Jain公平性指数
            'rewards': [],  # 奖励
            'user_decisions': [],  # 用户决策历史
            'channel_gains': [],  # 信道增益
            'unload_rates': [],  # 卸载速率
        }

        # 基准方法对比数据
        self.baseline_data = {
            'random': {'delays': [], 'jains': [], 'rewards': []},
            'nearest': {'delays': [], 'jains': [], 'rewards': []},
            'round_robin': {'delays': [], 'jains': [], 'rewards': []}
        }

    def run_test(self):
        """运行测试"""
        print("\n" + "=" * 80)
        print("开始模型测试")
        print("=" * 80)

        for episode in range(self.num_episodes):
            print(f"\n--- Episode {episode + 1}/{self.num_episodes} ---")

            # 重置环境
            obs, info = self.env.reset()

            # 存储当前episode的数据
            episode_uav_traj = [self.env.uav_positions.copy()]
            episode_user_pos = self.env.user_positions.copy()
            episode_delays = []
            episode_comm_delays = []
            episode_comp_delays = []
            episode_loads = []
            episode_jains = []
            episode_rewards = []
            episode_decisions = []
            episode_channels = []
            episode_rates = []

            done = False
            step = 0
            max_steps =20

            while not done:
                # 使用模型预测动作
                action, _states = self.model.predict(obs, deterministic=True)

                # 执行动作
                obs, reward, done, truncated, info = self.env.step(action)

                # 记录数据
                episode_uav_traj.append(self.env.uav_positions.copy())
                episode_delays.append(info['total_time'])
                episode_comm_delays.append(np.sum(info['comm_delay']))
                episode_comp_delays.append(np.sum(info['comp_delay']))
                episode_loads.append(info['uav_load'].copy())
                episode_jains.append(info['Jain_step'])
                episode_rewards.append(reward)
                episode_decisions.append(info['user_decisions'].copy())
                episode_channels.append(info['composite_channel'].copy())
                episode_rates.append(info['unload_rate'].copy())

                step += 1

                if step % 500 == 0:
                    print(f"  Step {step}: Delay={info['total_time']:.4f}s, "
                          f"Jain={info['Jain_step']:.4f}, Reward={reward:.4f}")

            # 保存episode数据
            self.test_data['uav_trajectories'].append(episode_uav_traj)
            self.test_data['user_positions'].append(episode_user_pos)
            self.test_data['total_delays'].append(episode_delays)
            self.test_data['comm_delays'].append(episode_comm_delays)
            self.test_data['comp_delays'].append(episode_comp_delays)
            self.test_data['uav_loads'].append(episode_loads)
            self.test_data['jain_indices'].append(episode_jains)
            self.test_data['rewards'].append(episode_rewards)
            self.test_data['user_decisions'].append(episode_decisions)
            self.test_data['channel_gains'].append(episode_channels)
            self.test_data['unload_rates'].append(episode_rates)

            # 运行基准方法对比
            print("  运行基准方法对比...")
            self._run_baseline_comparison(episode_user_pos)

            print(f"  ✓ Episode完成: 平均时延={np.mean(episode_delays):.4f}s, "
                  f"平均Jain={np.mean(episode_jains):.4f}, "
                  f"总奖励={np.sum(episode_rewards):.2f}")

        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80 + "\n")

    """运行基准方法对比"""
    def _run_baseline_comparison(self, user_positions):


        # 1. 随机策略
        obs, _ = self.env.reset()
        self.env.user_positions = user_positions.copy()

        total_delay_random = 0
        total_jain_random = 0
        total_reward_random = 0
        steps = 0
        done = False

        while not done:
            action = self.env.action_space.sample()
            obs, reward, done, truncated, info = self.env.step(action)
            total_delay_random += info['total_time']
            total_jain_random += info['Jain_step']
            total_reward_random += reward
            steps += 1

        self.baseline_data['random']['delays'].append(total_delay_random / steps)
        self.baseline_data['random']['jains'].append(total_jain_random / steps)
        self.baseline_data['random']['rewards'].append(total_reward_random)

        # 2. 最近邻策略
        obs, _ = self.env.reset()
        self.env.user_positions = user_positions.copy()

        total_delay_nearest = 0
        total_jain_nearest = 0
        total_reward_nearest = 0
        steps = 0
        done = False

        while not done:
            # UAV不移动，用户选择最近的UAV
            uav_actions = [0] * num_uavs  # 不移动
            user_decisions = []

            for k in range(num_users):
                distances = np.linalg.norm(
                    self.env.uav_positions - self.env.user_positions[k], axis=1
                )
                nearest_uav = np.argmin(distances) + 1  # 1-indexed
                user_decisions.append(nearest_uav)

            action = uav_actions + user_decisions
            obs, reward, done, truncated, info = self.env.step(action)
            total_delay_nearest += info['total_time']
            total_jain_nearest += info['Jain_step']
            total_reward_nearest += reward
            steps += 1

        self.baseline_data['nearest']['delays'].append(total_delay_nearest / steps)
        self.baseline_data['nearest']['jains'].append(total_jain_nearest / steps)
        self.baseline_data['nearest']['rewards'].append(total_reward_nearest)

        # 3. 轮询策略
        obs, _ = self.env.reset()
        self.env.user_positions = user_positions.copy()

        total_delay_rr = 0
        total_jain_rr = 0
        total_reward_rr = 0
        steps = 0
        done = False

        while not done:
            uav_actions = [0] * num_uavs
            user_decisions = [(k % num_uavs) + 1 for k in range(num_users)]

            action = uav_actions + user_decisions
            obs, reward, done, truncated, info = self.env.step(action)
            total_delay_rr += info['total_time']
            total_jain_rr += info['Jain_step']
            total_reward_rr += reward
            steps += 1

        self.baseline_data['round_robin']['delays'].append(total_delay_rr / steps)
        self.baseline_data['round_robin']['jains'].append(total_jain_rr / steps)
        self.baseline_data['round_robin']['rewards'].append(total_reward_rr)

    """绘制UAV轨迹图"""
    def plot_uav_trajectories(self, save_dir='./test_results'):

        os.makedirs(save_dir, exist_ok=True)

        for episode_idx in range(self.num_episodes):
            fig, ax = plt.subplots(figsize=(12, 12), dpi=150)

            # 绘制区域边界
            ax.plot([-400, 400, 400, -400, -400],
                    [-400, -400, 400, 400, -400],
                    'k--', linewidth=2, label='区域边界')

            # 绘制RIS位置
            ax.scatter(ris_pos[0], ris_pos[1], s=300, c='gold',
                       marker='*', edgecolors='black', linewidth=2,
                       label=f'RIS (h={ris_pos[2]}m)', zorder=10)

            # 绘制用户位置
            user_pos = self.test_data['user_positions'][episode_idx]
            ax.scatter(user_pos[:, 0], user_pos[:, 1], s=100, c='red',
                       marker='o', alpha=0.7, label='GT用户', zorder=5)

            # 为每个用户标注编号
            for k, pos in enumerate(user_pos):
                ax.annotate(f'GT{k}', xy=(pos[0], pos[1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, color='darkred')

            # 绘制UAV轨迹
            trajectories = self.test_data['uav_trajectories'][episode_idx]
            trajectories = np.array(trajectories)

            colors = ['blue', 'green', 'purple']
            markers = ['s', '^', 'D']

            for uav_idx in range(num_uavs):
                traj = trajectories[:, uav_idx, :]

                # 轨迹线
                ax.plot(traj[:, 0], traj[:, 1],
                        color=colors[uav_idx], linewidth=2,
                        alpha=0.6, label=f'UAV{uav_idx + 1} 轨迹')

                # 起点
                ax.scatter(traj[0, 0], traj[0, 1], s=200,
                           c=colors[uav_idx], marker=markers[uav_idx],
                           edgecolors='black', linewidth=2,
                           label=f'UAV{uav_idx + 1} 起点', zorder=8)

                # 终点
                ax.scatter(traj[-1, 0], traj[-1, 1], s=250,
                           c=colors[uav_idx], marker=markers[uav_idx],
                           edgecolors='red', linewidth=3,
                           label=f'UAV{uav_idx + 1} 终点', zorder=9)

                # 每隔一定步数标注位置
                step_interval = max(len(traj) // 5, 1)
                for i in range(0, len(traj), step_interval):
                    if i > 0 and i < len(traj) - 1:
                        ax.scatter(traj[i, 0], traj[i, 1], s=50,
                                   c=colors[uav_idx], marker='o',
                                   alpha=0.5, zorder=7)

            ax.set_xlabel('X 坐标 (m)', fontsize=12)
            ax.set_ylabel('Y 坐标 (m)', fontsize=12)
            ax.set_title(f'UAV轨迹图 - Episode {episode_idx + 1}\n'
                         f'(UAV高度: {uav_H}m)', fontsize=14, pad=20)
            ax.legend(loc='upper right', fontsize=10, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_xlim(-450, 450)
            ax.set_ylim(-450, 450)

            save_path = os.path.join(save_dir, f'uav_trajectory_ep{episode_idx + 1}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ 轨迹图已保存: {save_path}")

    """绘制时延分析图"""
    def plot_delay_analysis(self, save_dir='./test_results'):

        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

        # 1. 总时延随步数变化
        ax = axes[0, 0]
        for ep_idx in range(self.num_episodes):
            steps = np.arange(len(self.test_data['total_delays'][ep_idx]))
            ax.plot(steps, self.test_data['total_delays'][ep_idx],
                    linewidth=2, alpha=0.7, label=f'Episode {ep_idx + 1}')

        ax.set_xlabel('步数', fontsize=12)
        ax.set_ylabel('总时延 (s)', fontsize=12)
        ax.set_title('总时延变化曲线', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 通信与计算时延对比
        ax = axes[0, 1]
        avg_comm = np.mean([np.mean(d) for d in self.test_data['comm_delays']])
        avg_comp = np.mean([np.mean(d) for d in self.test_data['comp_delays']])

        categories = ['通信时延', '计算时延']
        values = [avg_comm, avg_comp]
        colors_bar = ['skyblue', 'lightcoral']

        bars = ax.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=2)
        ax.set_ylabel('平均时延 (s)', fontsize=12)
        ax.set_title('通信 vs 计算时延', fontsize=14)
        ax.grid(axis='y', alpha=0.3)

        # 在柱状图上标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}s', ha='center', va='bottom', fontsize=11)

        # 3. 时延分布箱线图
        ax = axes[1, 0]
        all_delays = [self.test_data['total_delays'][i]
                      for i in range(self.num_episodes)]

        bp = ax.boxplot(all_delays, labels=[f'Ep{i + 1}' for i in range(self.num_episodes)],
                        patch_artist=True, showmeans=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('总时延 (s)', fontsize=12)
        ax.set_title('时延分布箱线图', fontsize=14)
        ax.grid(axis='y', alpha=0.3)

        # 4. 与基准方法对比
        ax = axes[1, 1]

        ppo_avg = np.mean([np.mean(d) for d in self.test_data['total_delays']])
        random_avg = np.mean(self.baseline_data['random']['delays'])
        nearest_avg = np.mean(self.baseline_data['nearest']['delays'])
        rr_avg = np.mean(self.baseline_data['round_robin']['delays'])

        methods = ['PPO\n(本文)', '随机策略', '最近邻', '轮询']
        delays = [ppo_avg, random_avg, nearest_avg, rr_avg]
        colors_cmp = ['green', 'gray', 'orange', 'brown']

        bars = ax.bar(methods, delays, color=colors_cmp, edgecolor='black', linewidth=2)
        ax.set_ylabel('平均总时延 (s)', fontsize=12)
        ax.set_title('不同方法时延对比', fontsize=14)
        ax.grid(axis='y', alpha=0.3)

        # 标注改进百分比
        for i, bar in enumerate(bars[1:], 1):
            if delays[i] > 0:
                improvement = (delays[i] - ppo_avg) / delays[i] * 100
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'-{improvement:.1f}%', ha='center', va='bottom',
                        fontsize=10, color='red', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'delay_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 时延分析图已保存: {save_path}")

    """绘制UAV负载分析图"""
    def plot_uav_load_analysis(self, save_dir='./test_results'):

        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

        # 1. UAV负载随时间变化（堆叠面积图）
        ax = axes[0, 0]

        # 选择第一个episode的数据
        loads = np.array(self.test_data['uav_loads'][0])  # shape: (steps, num_uavs)
        steps = np.arange(loads.shape[0])

        colors_load = ['#3498db', '#2ecc71', '#9b59b6']
        labels = [f'UAV{i + 1}' for i in range(num_uavs)]

        ax.stackplot(steps, loads.T, labels=labels, colors=colors_load, alpha=0.7)
        ax.set_xlabel('步数', fontsize=12)
        ax.set_ylabel('负载 (Mbits)', fontsize=12)
        ax.set_title('UAV负载随时间变化 (Episode 1)', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 2. Jain公平性指数变化
        ax = axes[0, 1]
        for ep_idx in range(self.num_episodes):
            steps = np.arange(len(self.test_data['jain_indices'][ep_idx]))
            ax.plot(steps, self.test_data['jain_indices'][ep_idx],
                    linewidth=2, alpha=0.7, label=f'Episode {ep_idx + 1}')

        ax.axhline(y=1 / num_uavs, color='red', linestyle='--',
                   linewidth=2, label=f'最差公平性 (1/{num_uavs})')
        ax.axhline(y=1.0, color='green', linestyle='--',
                   linewidth=2, label='完美公平性 (1.0)')

        ax.set_xlabel('步数', fontsize=12)
        ax.set_ylabel('Jain公平性指数', fontsize=12)
        ax.set_title('负载均衡公平性变化', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])

        # 3. 平均负载对比
        ax = axes[1, 0]

        avg_loads = np.zeros(num_uavs)
        for ep_loads in self.test_data['uav_loads']:
            ep_loads_array = np.array(ep_loads)
            avg_loads += np.mean(ep_loads_array, axis=0)
        avg_loads /= self.num_episodes

        uav_labels = [f'UAV{i + 1}' for i in range(num_uavs)]
        bars = ax.bar(uav_labels, avg_loads, color=colors_load,
                      edgecolor='black', linewidth=2)

        # 添加平均线
        mean_load = np.mean(avg_loads)
        ax.axhline(y=mean_load, color='red', linestyle='--',
                   linewidth=2, label=f'平均负载: {mean_load:.2f}')

        ax.set_ylabel('平均负载 (Mbits)', fontsize=12)
        ax.set_title('各UAV平均负载对比', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=11)

        # 4. Jain指数对比（基准方法）
        ax = axes[1, 1]

        ppo_jain = np.mean([np.mean(j) for j in self.test_data['jain_indices']])
        random_jain = np.mean(self.baseline_data['random']['jains'])
        nearest_jain = np.mean(self.baseline_data['nearest']['jains'])
        rr_jain = np.mean(self.baseline_data['round_robin']['jains'])

        methods = ['PPO\n(本文)', '随机策略', '最近邻', '轮询']
        jains = [ppo_jain, random_jain, nearest_jain, rr_jain]
        colors_jain = ['green', 'gray', 'orange', 'brown']

        bars = ax.bar(methods, jains, color=colors_jain,
                      edgecolor='black', linewidth=2)
        ax.set_ylabel('平均Jain公平性指数', fontsize=12)
        ax.set_title('不同方法公平性对比', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])

        # 标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'uav_load_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ UAV负载分析图已保存: {save_path}")

    """绘制综合性能指标"""
    def plot_performance_metrics(self, save_dir='./test_results'):

        os.makedirs(save_dir, exist_ok=True)

        fig = plt.figure(figsize=(16, 10), dpi=150)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 奖励曲线
        ax1 = fig.add_subplot(gs[0, :])
        for ep_idx in range(self.num_episodes):
            steps = np.arange(len(self.test_data['rewards'][ep_idx]))
            rewards = self.test_data['rewards'][ep_idx]
            ax1.plot(steps, rewards, linewidth=2, alpha=0.7,
                     label=f'Episode {ep_idx + 1}')

        ax1.set_xlabel('步数', fontsize=12)
        ax1.set_ylabel('奖励', fontsize=12)
        ax1.set_title('奖励变化曲线', fontsize=14)
        ax1.legend(ncol=self.num_episodes)
        ax1.grid(True, alpha=0.3)

        # 2. 用户决策分布
        ax2 = fig.add_subplot(gs[1, 0])

        # 统计所有episode中用户选择各UAV的次数
        decision_counts = np.zeros(num_uavs + 1)  # [本地, UAV1, UAV2, UAV3]
        total_decisions = 0

        for ep_decisions in self.test_data['user_decisions']:
            for step_decisions in ep_decisions:
                for decision in step_decisions:
                    decision_counts[decision] += 1
                    total_decisions += 1

        decision_ratios = decision_counts / total_decisions * 100

        labels = ['本地计算'] + [f'UAV{i + 1}' for i in range(num_uavs)]
        colors_pie = ['lightgray', '#3498db', '#2ecc71', '#9b59b6']

        wedges, texts, autotexts = ax2.pie(decision_ratios, labels=labels,
                                           colors=colors_pie, autopct='%1.1f%%',
                                           startangle=90, textprops={'fontsize': 11})
        ax2.set_title('用户卸载决策分布', fontsize=14)

        # 3. 信道增益热力图（取第一个episode的平均值）
        ax3 = fig.add_subplot(gs[1, 1])

        avg_channel = np.mean(self.test_data['channel_gains'][0], axis=0)  # (num_uavs, num_users)
        avg_channel_db = 10 * np.log10(avg_channel + 1e-12)

        im = ax3.imshow(avg_channel_db, cmap='hot', aspect='auto')
        ax3.set_xlabel('GT用户', fontsize=12)
        ax3.set_ylabel('UAV', fontsize=12)
        ax3.set_title('平均信道增益 (dB) - Ep1', fontsize=14)
        ax3.set_xticks(range(num_users))
        ax3.set_xticklabels([f'GT{i}' for i in range(num_users)])
        ax3.set_yticks(range(num_uavs))
        ax3.set_yticklabels([f'UAV{i + 1}' for i in range(num_uavs)])

        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('信道增益 (dB)', fontsize=11)

        # 4. 卸载速率热力图
        ax4 = fig.add_subplot(gs[1, 2])

        avg_rate = np.mean(self.test_data['unload_rates'][0], axis=0)  # (num_uavs, num_users)

        im = ax4.imshow(avg_rate, cmap='viridis', aspect='auto')
        ax4.set_xlabel('GT用户', fontsize=12)
        ax4.set_ylabel('UAV', fontsize=12)
        ax4.set_title('平均卸载速率 (Mbps) - Ep1', fontsize=14)
        ax4.set_xticks(range(num_users))
        ax4.set_xticklabels([f'GT{i}' for i in range(num_users)])
        ax4.set_yticks(range(num_uavs))
        ax4.set_yticklabels([f'UAV{i + 1}' for i in range(num_uavs)])

        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('速率 (Mbps)', fontsize=11)

        # 5. 总奖励对比
        ax5 = fig.add_subplot(gs[2, 0])

        ppo_reward = np.mean([np.sum(r) for r in self.test_data['rewards']])
        random_reward = np.mean(self.baseline_data['random']['rewards'])
        nearest_reward = np.mean(self.baseline_data['nearest']['rewards'])
        rr_reward = np.mean(self.baseline_data['round_robin']['rewards'])

        methods = ['PPO', '随机', '最近邻', '轮询']
        rewards_cmp = [ppo_reward, random_reward, nearest_reward, rr_reward]
        colors_cmp = ['green', 'gray', 'orange', 'brown']

        bars = ax5.bar(methods, rewards_cmp, color=colors_cmp,
                       edgecolor='black', linewidth=2)
        ax5.set_ylabel('平均总奖励', fontsize=12)
        ax5.set_title('不同方法奖励对比', fontsize=14)
        ax5.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=11)

        # 6. 性能雷达图
        ax6 = fig.add_subplot(gs[2, 1:], projection='polar')

        # 归一化指标（越高越好）
        ppo_avg = np.mean([np.mean(d) for d in self.test_data['total_delays']])
        random_avg = np.mean(self.baseline_data['random']['delays'])
        nearest_avg = np.mean(self.baseline_data['nearest']['delays'])
        rr_avg = np.mean(self.baseline_data['round_robin']['delays'])

        ppo_jain = np.mean([np.mean(j) for j in self.test_data['jain_indices']])
        random_jain = np.mean(self.baseline_data['random']['jains'])

        max_delay = max(ppo_avg, random_avg, nearest_avg, rr_avg)
        ppo_delay_norm = 1 - (ppo_avg / max_delay) if max_delay > 0 else 0
        random_delay_norm = 1 - (random_avg / max_delay) if max_delay > 0 else 0

        ppo_jain_norm = ppo_jain
        random_jain_norm = random_jain

        max_reward = max(abs(ppo_reward), abs(random_reward), abs(nearest_reward), abs(rr_reward))
        ppo_reward_norm = ppo_reward / max_reward if max_reward > 0 else 0
        random_reward_norm = random_reward / max_reward if max_reward > 0 else 0

        categories = ['时延性能', 'Jain公平性', '总奖励']
        ppo_values = [ppo_delay_norm, ppo_jain_norm, ppo_reward_norm]
        random_values = [random_delay_norm, random_jain_norm, random_reward_norm]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        ppo_values += ppo_values[:1]
        random_values += random_values[:1]
        angles += angles[:1]

        ax6.plot(angles, ppo_values, 'o-', linewidth=2, label='PPO', color='green')
        ax6.fill(angles, ppo_values, alpha=0.25, color='green')

        ax6.plot(angles, random_values, 'o-', linewidth=2, label='随机策略', color='gray')
        ax6.fill(angles, random_values, alpha=0.25, color='gray')

        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories, fontsize=11)
        ax6.set_ylim(0, 1)
        ax6.set_title('综合性能雷达图', fontsize=14, pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax6.grid(True)

        save_path = os.path.join(save_dir, 'performance_metrics.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 性能指标图已保存: {save_path}")

    """生成测试报告"""
    def generate_report(self, save_dir='./test_results'):

        os.makedirs(save_dir, exist_ok=True)

        report_path = os.path.join(save_dir, 'test_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("UAV-RIS 卸载系统 - 测试报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"测试Episode数: {self.num_episodes}\n\n")

            f.write("=" * 80 + "\n")
            f.write("一、PPO模型性能\n")
            f.write("=" * 80 + "\n\n")

            # 时延统计
            avg_delays = [np.mean(d) for d in self.test_data['total_delays']]
            f.write(f"1. 时延性能:\n")
            f.write(f"   平均总时延: {np.mean(avg_delays):.4f} ± {np.std(avg_delays):.4f} s\n")
            f.write(f"   最小时延: {np.min([np.min(d) for d in self.test_data['total_delays']]):.4f} s\n")
            f.write(f"   最大时延: {np.max([np.max(d) for d in self.test_data['total_delays']]):.4f} s\n\n")

            avg_comm = [np.mean(d) for d in self.test_data['comm_delays']]
            avg_comp = [np.mean(d) for d in self.test_data['comp_delays']]
            f.write(f"   平均通信时延: {np.mean(avg_comm):.4f} ± {np.std(avg_comm):.4f} s\n")
            f.write(f"   平均计算时延: {np.mean(avg_comp):.4f} ± {np.std(avg_comp):.4f} s\n\n")

            # Jain指数
            avg_jains = [np.mean(j) for j in self.test_data['jain_indices']]
            f.write(f"2. 负载均衡性能:\n")
            f.write(f"   平均Jain指数: {np.mean(avg_jains):.4f} ± {np.std(avg_jains):.4f}\n")
            f.write(f"   最小Jain指数: {np.min([np.min(j) for j in self.test_data['jain_indices']]):.4f}\n")
            f.write(f"   最大Jain指数: {np.max([np.max(j) for j in self.test_data['jain_indices']]):.4f}\n\n")

            # 奖励
            total_rewards = [np.sum(r) for r in self.test_data['rewards']]
            f.write(f"3. 奖励性能:\n")
            f.write(f"   平均总奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}\n")
            f.write(f"   最大总奖励: {np.max(total_rewards):.2f}\n")
            f.write(f"   最小总奖励: {np.min(total_rewards):.2f}\n\n")

            f.write("=" * 80 + "\n")
            f.write("二、与基准方法对比\n")
            f.write("=" * 80 + "\n\n")

            ppo_delay = np.mean(avg_delays)
            random_delay = np.mean(self.baseline_data['random']['delays'])
            nearest_delay = np.mean(self.baseline_data['nearest']['delays'])
            rr_delay = np.mean(self.baseline_data['round_robin']['delays'])

            f.write(f"1. 时延对比:\n")
            f.write(f"   PPO:      {ppo_delay:.4f} s\n")
            if random_delay > 0:
                f.write(
                    f"   随机策略: {random_delay:.4f} s  (改进: {(random_delay - ppo_delay) / random_delay * 100:.1f}%)\n")
            if nearest_delay > 0:
                f.write(
                    f"   最近邻:   {nearest_delay:.4f} s  (改进: {(nearest_delay - ppo_delay) / nearest_delay * 100:.1f}%)\n")
            if rr_delay > 0:
                f.write(f"   轮询:     {rr_delay:.4f} s  (改进: {(rr_delay - ppo_delay) / rr_delay * 100:.1f}%)\n\n")

            ppo_jain = np.mean(avg_jains)
            random_jain = np.mean(self.baseline_data['random']['jains'])
            nearest_jain = np.mean(self.baseline_data['nearest']['jains'])
            rr_jain = np.mean(self.baseline_data['round_robin']['jains'])

            f.write(f"2. Jain指数对比:\n")
            f.write(f"   PPO:      {ppo_jain:.4f}\n")
            if random_jain > 0:
                f.write(
                    f"   随机策略: {random_jain:.4f}  (改进: {(ppo_jain - random_jain) / random_jain * 100:.1f}%)\n")
            if nearest_jain > 0:
                f.write(
                    f"   最近邻:   {nearest_jain:.4f}  (改进: {(ppo_jain - nearest_jain) / nearest_jain * 100:.1f}%)\n")
            if rr_jain > 0:
                f.write(f"   轮询:     {rr_jain:.4f}  (改进: {(ppo_jain - rr_jain) / rr_jain * 100:.1f}%)\n\n")

            ppo_reward = np.mean(total_rewards)
            random_reward = np.mean(self.baseline_data['random']['rewards'])
            nearest_reward = np.mean(self.baseline_data['nearest']['rewards'])
            rr_reward = np.mean(self.baseline_data['round_robin']['rewards'])

            f.write(f"3. 总奖励对比:\n")
            f.write(f"   PPO:      {ppo_reward:.2f}\n")
            if abs(random_reward) > 0:
                f.write(
                    f"   随机策略: {random_reward:.2f}  (改进: {(ppo_reward - random_reward) / abs(random_reward) * 100:.1f}%)\n")
            if abs(nearest_reward) > 0:
                f.write(
                    f"   最近邻:   {nearest_reward:.2f}  (改进: {(ppo_reward - nearest_reward) / abs(nearest_reward) * 100:.1f}%)\n")
            if abs(rr_reward) > 0:
                f.write(
                    f"   轮询:     {rr_reward:.2f}  (改进: {(ppo_reward - rr_reward) / abs(rr_reward) * 100:.1f}%)\n\n")

            f.write("=" * 80 + "\n")
            f.write("三、UAV负载分析\n")
            f.write("=" * 80 + "\n\n")

            avg_loads = np.zeros(num_uavs)
            for ep_loads in self.test_data['uav_loads']:
                ep_loads_array = np.array(ep_loads)
                avg_loads += np.mean(ep_loads_array, axis=0)
            avg_loads /= self.num_episodes

            f.write(f"各UAV平均负载 (Mbits):\n")
            for i in range(num_uavs):
                f.write(f"   UAV{i + 1}: {avg_loads[i]:.3f}\n")
            f.write(f"   负载标准差: {np.std(avg_loads):.3f}\n\n")

            f.write("=" * 80 + "\n")
            f.write("测试完成！\n")
            f.write("=" * 80 + "\n")

        print(f"✓ 测试报告已保存: {report_path}")

    def run_complete_test(self):
        """运行完整测试流程"""
        # 1. 运行测试
        self.run_test()

        # 2. 生成所有图表
        print("\n生成可视化图表...")
        self.plot_uav_trajectories()
        self.plot_delay_analysis()
        self.plot_uav_load_analysis()
        self.plot_performance_metrics()

        # 3. 生成报告
        print("\n生成测试报告...")
        self.generate_report()

        print("\n" + "=" * 80)
        print("所有测试完成！结果已保存至 ./test_results/ 目录")
        print("=" * 80)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("UAV-RIS 卸载系统 - 模型测试程序")
    print("=" * 80 + "\n")

    # 配置参数
    MODEL_PATH = "ppo_uav_ris_10"
    NUM_EPISODES = 5

    # 创建测试器
    tester = ModelTester(
        model_path=MODEL_PATH,
        num_episodes=NUM_EPISODES
    )

    # 运行完整测试
    tester.run_complete_test()


if __name__ == "__main__":
    main()