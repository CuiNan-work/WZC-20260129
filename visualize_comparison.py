"""
UAV 轨迹对比可视化
对比改进前后的 UAV 行为差异
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from main import UAVEnv, num_uavs, num_users, area_size, uav_H

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def simulate_trajectory(env, strategy='improved', steps=20, seed=42):
    """
    模拟 UAV 轨迹
    
    Args:
        env: UAV 环境
        strategy: 'improved' (改进后), 'boundary' (边界策略), 'random' (随机)
        steps: 模拟步数
        seed: 随机种子
    
    Returns:
        trajectories: UAV 轨迹列表
        user_positions: 用户位置
        rewards: 每步奖励
        delays: 每步时延
    """
    # 常量定义
    MIN_TARGET_DISTANCE = 20.0  # UAV到目标的最小距离（米）
    BOUNDARY_MARGIN = 100.0      # 边界区域宽度（米）
    DIRECTION_THRESHOLD = 5.0    # 方向判断阈值（米）
    
    np.random.seed(seed)
    obs, _ = env.reset()
    
    trajectories = [env.uav_positions.copy()]
    user_positions = env.user_positions.copy()
    rewards = []
    delays = []
    
    for step in range(steps):
        if strategy == 'improved':
            # 改进策略：向用户靠近
            action = []
            
            # UAV 移动：向最近的用户群中心移动
            for i in range(num_uavs):
                # 计算到所有用户的距离
                distances = np.linalg.norm(
                    env.user_positions - env.uav_positions[i], 
                    axis=1
                )
                
                # 找到最近的用户
                nearest_user_idx = np.argmin(distances)
                target = env.user_positions[nearest_user_idx]
                
                # 计算方向
                direction = target - env.uav_positions[i]
                
                # 选择移动动作
                if np.linalg.norm(direction) < MIN_TARGET_DISTANCE:  # 已经很近了
                    move_action = 0  # 不动
                else:
                    # 根据方向选择动作
                    dx, dy = direction
                    
                    if abs(dx) < DIRECTION_THRESHOLD and abs(dy) < DIRECTION_THRESHOLD:
                        move_action = 0  # 不动
                    elif abs(dx) > abs(dy):
                        if dx > 0:
                            move_action = 4  # 右
                        else:
                            move_action = 3  # 左
                    else:
                        if dy > 0:
                            move_action = 1  # 上
                        else:
                            move_action = 2  # 下
                
                action.append(move_action)
            
            # 用户决策：选择最近的 UAV
            for k in range(num_users):
                distances = np.linalg.norm(
                    env.uav_positions - env.user_positions[k], 
                    axis=1
                )
                nearest_uav = np.argmin(distances) + 1  # 1-indexed
                action.append(nearest_uav)
        
        elif strategy == 'boundary':
            # 边界策略：保持在边界附近
            action = []
            
            # UAV 不移动或向边界移动
            for i in range(num_uavs):
                x, y = env.uav_positions[i]
                
                # 如果已经在边界，不动
                if x < BOUNDARY_MARGIN or x > area_size - BOUNDARY_MARGIN or \
                   y < BOUNDARY_MARGIN or y > area_size - BOUNDARY_MARGIN:
                    move_action = 0
                else:
                    # 向最近的边界移动
                    to_left = x
                    to_right = area_size - x
                    to_bottom = y
                    to_top = area_size - y
                    
                    min_dist = min(to_left, to_right, to_bottom, to_top)
                    
                    if min_dist == to_left:
                        move_action = 3  # 左
                    elif min_dist == to_right:
                        move_action = 4  # 右
                    elif min_dist == to_bottom:
                        move_action = 2  # 下
                    else:
                        move_action = 1  # 上
                
                action.append(move_action)
            
            # 用户随机选择（包括本地计算选项）
            for k in range(num_users):
                action.append(np.random.randint(0, num_uavs + 1))
        
        else:  # random
            action = env.action_space.sample()
        
        # 执行动作
        obs, reward, done, truncated, info = env.step(action)
        
        # 记录数据
        trajectories.append(env.uav_positions.copy())
        rewards.append(reward)
        delays.append(info['total_time'])
        
        if done:
            break
    
    return trajectories, user_positions, rewards, delays


def visualize_comparison(save_path='trajectory_comparison.png'):
    """可视化不同策略的轨迹对比"""
    
    # 可视化常量
    USER_COVERAGE_RADIUS = 50.0  # 用户覆盖半径（米）
    
    # 创建三个环境实例
    env_improved = UAVEnv()
    env_boundary = UAVEnv()
    env_random = UAVEnv()
    
    # 模拟轨迹
    print("模拟改进策略...")
    traj_improved, users, rewards_imp, delays_imp = simulate_trajectory(
        env_improved, 'improved', steps=15, seed=42
    )
    
    print("模拟边界策略...")
    traj_boundary, _, rewards_bnd, delays_bnd = simulate_trajectory(
        env_boundary, 'boundary', steps=15, seed=42
    )
    
    print("模拟随机策略...")
    traj_random, _, rewards_rnd, delays_rnd = simulate_trajectory(
        env_random, 'random', steps=15, seed=42
    )
    
    # 创建图表
    fig = plt.figure(figsize=(18, 10))
    
    # 绘制轨迹对比
    strategies = [
        ('改进策略（向用户靠近）', traj_improved, rewards_imp, delays_imp),
        ('边界策略（问题行为）', traj_boundary, rewards_bnd, delays_bnd),
        ('随机策略（基线）', traj_random, rewards_rnd, delays_rnd)
    ]
    
    colors_uav = ['red', 'blue', 'green']
    
    for idx, (title, trajectories, rewards, delays) in enumerate(strategies):
        # 轨迹图
        ax = plt.subplot(2, 3, idx + 1)
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        
        # 绘制边界区域（浅灰色）
        boundary_margin = area_size * 0.1
        ax.add_patch(plt.Rectangle((0, 0), boundary_margin, area_size, 
                                   color='gray', alpha=0.1))
        ax.add_patch(plt.Rectangle((area_size - boundary_margin, 0), 
                                   boundary_margin, area_size, 
                                   color='gray', alpha=0.1))
        ax.add_patch(plt.Rectangle((0, 0), area_size, boundary_margin, 
                                   color='gray', alpha=0.1))
        ax.add_patch(plt.Rectangle((0, area_size - boundary_margin), 
                                   area_size, boundary_margin, 
                                   color='gray', alpha=0.1))
        
        # 绘制用户位置
        for k in range(num_users):
            ax.plot(users[k, 0], users[k, 1], 
                   marker='^', markersize=12, color='orange',
                   markeredgecolor='black', markeredgewidth=1.5)
            ax.add_patch(Circle(users[k], USER_COVERAGE_RADIUS, 
                               color='orange', alpha=0.15, linewidth=0))
        
        # 绘制 UAV 轨迹
        for i in range(num_uavs):
            # 提取该 UAV 的轨迹
            traj = np.array([t[i] for t in trajectories])
            
            # 绘制轨迹线
            ax.plot(traj[:, 0], traj[:, 1], 
                   color=colors_uav[i], linewidth=2, 
                   alpha=0.6, linestyle='--')
            
            # 标记起点
            ax.plot(traj[0, 0], traj[0, 1], 
                   marker='o', markersize=10, color=colors_uav[i],
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'UAV{i+1} 起点')
            
            # 标记终点
            ax.plot(traj[-1, 0], traj[-1, 1], 
                   marker='s', markersize=10, color=colors_uav[i],
                   markeredgecolor='black', markeredgewidth=1.5)
        
        # 图例（只在第一个子图显示）
        if idx == 0:
            handles = [
                plt.Line2D([0], [0], marker='^', color='w', 
                          markerfacecolor='orange', markersize=10,
                          markeredgecolor='black', label='用户'),
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='gray', markersize=10,
                          markeredgecolor='black', label='UAV起点'),
                plt.Line2D([0], [0], marker='s', color='w', 
                          markerfacecolor='gray', markersize=10,
                          markeredgecolor='black', label='UAV终点'),
            ]
            ax.legend(handles=handles, loc='upper right', fontsize=10)
        
        # 性能指标图
        ax2 = plt.subplot(2, 3, idx + 4)
        
        steps = range(len(rewards))
        ax2_twin = ax2.twinx()
        
        # 绘制时延
        line1 = ax2.plot(steps, delays, color='blue', linewidth=2, 
                        marker='o', label='总时延')
        ax2.set_xlabel('步数', fontsize=11)
        ax2.set_ylabel('总时延 (s)', color='blue', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.grid(True, alpha=0.3)
        
        # 绘制奖励
        line2 = ax2_twin.plot(steps, rewards, color='red', linewidth=2,
                             marker='s', label='奖励')
        ax2_twin.set_ylabel('奖励', color='red', fontsize=11)
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # 统一图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='best', fontsize=10)
        
        # 添加性能统计
        avg_delay = np.mean(delays)
        total_reward = np.sum(rewards)
        ax2.text(0.02, 0.98, 
                f'平均时延: {avg_delay:.3f}s\n总奖励: {total_reward:.1f}',
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.suptitle('UAV 轨迹优化策略对比\n改进后奖励函数增强了距离敏感度，使UAV主动靠近用户', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 对比图已保存到: {save_path}")
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("性能对比统计")
    print("=" * 80)
    print(f"{'策略':<20} {'平均时延(s)':<15} {'总奖励':<15} {'平均奖励':<15}")
    print("-" * 80)
    
    for title, _, rewards, delays in strategies:
        avg_delay = np.mean(delays)
        total_reward = np.sum(rewards)
        avg_reward = np.mean(rewards)
        print(f"{title:<20} {avg_delay:<15.4f} {total_reward:<15.2f} {avg_reward:<15.2f}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("UAV 轨迹对比可视化")
    print("=" * 80)
    
    visualize_comparison('trajectory_comparison.png')
    
    print("\n完成！")
