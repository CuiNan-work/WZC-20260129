"""
UAV-RIS 卸载系统环境定义
基于 Gymnasium 的强化学习环境，用于优化 UAV 轨迹和用户卸载决策
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ==================== 系统参数 ====================
# UAV 参数
num_uavs = 3  # UAV 数量
uav_H = 100.0  # UAV 飞行高度 (m)
uav_max_speed = 10.0  # UAV 最大速度 (m/s)
uav_computing_capacity = 10e9  # UAV 计算能力 (10 GHz)

# 用户参数
num_users = 6  # 地面用户数量
user_task_size = 2.0  # 用户任务大小 (Mbits)
user_computing_capacity = 1e9  # 用户本地计算能力 (1 GHz)
task_cycles = 1000  # 每 bit 需要的 CPU 周期数

# 区域参数
area_size = 1000.0  # 仿真区域大小 (m × m)

# RIS 参数
ris_pos = np.array([area_size / 2, area_size / 2, 50.0])  # RIS 位置 [x, y, z]
ris_elements = 100  # RIS 反射元素数量

# 通信参数
bandwidth = 10e6  # 带宽 (10 MHz)
noise_power = 1e-13  # 噪声功率 (W)
tx_power = 0.1  # 发射功率 (W) = 20 dBm
path_loss_exponent = 2.5  # 路径损耗指数（加大以增强距离敏感度）
reference_distance = 1.0  # 参考距离 (m)
reference_path_loss_db = 30.0  # 参考路径损耗 (dB)

# 时间参数
time_slot = 1.0  # 时隙长度 (s)


# ==================== UAV 环境类 ====================
class UAVEnv(gym.Env):
    """UAV-RIS 辅助的边缘计算卸载环境"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        super(UAVEnv, self).__init__()
        
        # 动作空间：[UAV1移动方向, UAV2移动方向, UAV3移动方向, 
        #          用户1选择, 用户2选择, ..., 用户N选择]
        # UAV 移动：0=不动, 1=上, 2=下, 3=左, 4=右, 5=左上, 6=右上, 7=左下, 8=右下
        # 用户选择：0=本地计算, 1=UAV1, 2=UAV2, 3=UAV3
        self.action_space = spaces.MultiDiscrete([9] * num_uavs + [num_uavs + 1] * num_users)
        
        # 观察空间：[UAV位置(6), 用户位置(12), UAV负载(3), 历史时延(1)]
        # 总共 22 维
        obs_dim = num_uavs * 2 + num_users * 2 + num_uavs + 1
        self.observation_space = spaces.Box(
            low=0.0, 
            high=area_size, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # 初始化环境状态
        self.uav_positions = None
        self.user_positions = None
        self.uav_load = None
        self.step_count = 0
        self.max_steps = 20
        self.prev_total_delay = 0.0
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机初始化 UAV 位置（在区域内）
        self.uav_positions = np.random.uniform(
            low=area_size * 0.2, 
            high=area_size * 0.8, 
            size=(num_uavs, 2)
        )
        
        # 随机初始化用户位置
        self.user_positions = np.random.uniform(
            low=area_size * 0.1, 
            high=area_size * 0.9, 
            size=(num_users, 2)
        )
        
        # 初始化 UAV 负载
        self.uav_load = np.zeros(num_uavs)
        
        # 重置计数器
        self.step_count = 0
        self.prev_total_delay = 0.0
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """执行一步动作"""
        self.step_count += 1
        
        # 解析动作
        uav_actions = action[:num_uavs]
        user_decisions = action[num_uavs:]
        
        # 1. 更新 UAV 位置
        self._update_uav_positions(uav_actions)
        
        # 2. 计算通信和计算时延
        comm_delays, comp_delays, channel_gains, unload_rates = self._calculate_delays(user_decisions)
        
        # 3. 更新 UAV 负载
        self.uav_load = np.zeros(num_uavs)
        for k, decision in enumerate(user_decisions):
            if decision > 0:  # 卸载到 UAV
                uav_idx = decision - 1
                self.uav_load[uav_idx] += user_task_size
        
        # 4. 计算总时延
        total_delays = comm_delays + comp_delays
        total_time = np.sum(total_delays)
        
        # 5. 计算 Jain 公平性指数
        jain_index = self._calculate_jain_index(total_delays)
        
        # 6. 计算奖励（改进的奖励函数）
        reward = self._calculate_reward(total_delays, user_decisions, jain_index)
        
        # 7. 检查是否结束
        done = self.step_count >= self.max_steps
        truncated = False
        
        # 8. 构建 info 字典
        info = {
            'total_time': total_time,
            'comm_delay': comm_delays,
            'comp_delay': comp_delays,
            'uav_load': self.uav_load.copy(),
            'Jain_step': jain_index,
            'user_decisions': user_decisions.copy(),
            'composite_channel': channel_gains,
            'unload_rate': unload_rates
        }
        
        # 9. 获取新观察
        obs = self._get_observation()
        
        self.prev_total_delay = total_time
        
        return obs, reward, done, truncated, info
    
    def _update_uav_positions(self, uav_actions):
        """根据动作更新 UAV 位置"""
        # 定义 9 个移动方向：0=不动, 1-8=八个方向
        move_vectors = {
            0: [0, 0],           # 不动
            1: [0, 1],           # 上
            2: [0, -1],          # 下
            3: [-1, 0],          # 左
            4: [1, 0],           # 右
            5: [-1, 1],          # 左上
            6: [1, 1],           # 右上
            7: [-1, -1],         # 左下
            8: [1, -1]           # 右下
        }
        
        for i, action in enumerate(uav_actions):
            move = np.array(move_vectors[action]) * uav_max_speed * time_slot
            self.uav_positions[i] += move
            
            # 限制在区域内
            self.uav_positions[i] = np.clip(
                self.uav_positions[i], 
                0, 
                area_size
            )
    
    def _calculate_delays(self, user_decisions):
        """计算通信和计算时延"""
        comm_delays = np.zeros(num_users)
        comp_delays = np.zeros(num_users)
        channel_gains = np.zeros((num_uavs, num_users))
        unload_rates = np.zeros((num_uavs, num_users))
        
        for k in range(num_users):
            decision = user_decisions[k]
            
            if decision == 0:  # 本地计算
                comm_delays[k] = 0.0
                comp_delays[k] = (user_task_size * 1e6 * task_cycles) / user_computing_capacity
            else:  # 卸载到 UAV
                uav_idx = decision - 1
                
                # 计算 3D 距离
                uav_pos_3d = np.array([
                    self.uav_positions[uav_idx, 0],
                    self.uav_positions[uav_idx, 1],
                    uav_H
                ])
                user_pos_3d = np.array([
                    self.user_positions[k, 0],
                    self.user_positions[k, 1],
                    0.0
                ])
                distance = np.linalg.norm(uav_pos_3d - user_pos_3d)
                
                # 计算路径损耗（增强距离敏感度）
                if distance < reference_distance:
                    distance = reference_distance
                
                path_loss_db = reference_path_loss_db + 10 * path_loss_exponent * np.log10(
                    distance / reference_distance
                )
                path_loss = 10 ** (path_loss_db / 10)
                
                # 考虑 RIS 辅助的复合信道增益
                # 直达链路增益
                channel_gain_direct = tx_power / path_loss
                
                # RIS 辅助链路增益（简化模型）
                dist_to_ris = np.linalg.norm(user_pos_3d - ris_pos)
                dist_ris_to_uav = np.linalg.norm(uav_pos_3d - ris_pos)
                
                path_loss_to_ris_db = reference_path_loss_db + 10 * path_loss_exponent * np.log10(
                    max(dist_to_ris, reference_distance) / reference_distance
                )
                path_loss_ris_to_uav_db = reference_path_loss_db + 10 * path_loss_exponent * np.log10(
                    max(dist_ris_to_uav, reference_distance) / reference_distance
                )
                
                path_loss_ris = 10 ** ((path_loss_to_ris_db + path_loss_ris_to_uav_db) / 10)
                channel_gain_ris = (tx_power * ris_elements) / path_loss_ris
                
                # 复合信道增益
                composite_gain = channel_gain_direct + channel_gain_ris
                channel_gains[uav_idx, k] = composite_gain
                
                # 计算信噪比和传输速率
                snr = composite_gain / noise_power
                rate = bandwidth * np.log2(1 + snr)  # bps
                unload_rates[uav_idx, k] = rate / 1e6  # Mbps
                
                # 通信时延（上传时延）
                comm_delays[k] = (user_task_size * 1e6) / rate  # seconds
                
                # 计算时延（在 UAV 上计算）
                comp_delays[k] = (user_task_size * 1e6 * task_cycles) / uav_computing_capacity
        
        return comm_delays, comp_delays, channel_gains, unload_rates
    
    def _calculate_jain_index(self, delays):
        """计算 Jain 公平性指数"""
        if np.sum(delays) == 0:
            return 1.0
        
        n = len(delays)
        numerator = (np.sum(delays)) ** 2
        denominator = n * np.sum(delays ** 2)
        
        if denominator == 0:
            return 1.0
        
        return numerator / denominator
    
    def _calculate_reward(self, total_delays, user_decisions, jain_index):
        """
        改进的奖励函数 - 增强对 UAV-用户距离的敏感度
        
        关键改进：
        1. 增加对 UAV-用户平均距离的直接惩罚
        2. 增加通信时延的权重
        3. 对边界附近的 UAV 位置增加额外惩罚
        """
        # 1. 时延惩罚（主要奖励信号）
        avg_delay = np.mean(total_delays)
        delay_penalty = avg_delay * 10.0  # 增加权重
        
        # 2. 距离惩罚（新增 - 直接惩罚 UAV 到用户的距离）
        distance_penalty = 0.0
        for k in range(num_users):
            # 计算该用户到最近 UAV 的距离
            min_dist = float('inf')
            for i in range(num_uavs):
                dist_2d = np.linalg.norm(
                    self.uav_positions[i] - self.user_positions[k]
                )
                # 考虑高度差异，使用3D距离
                dist_3d = np.sqrt(dist_2d ** 2 + uav_H ** 2)
                min_dist = min(min_dist, dist_3d)
            
            # 距离惩罚：使用二次方增加敏感度
            distance_penalty += (min_dist / 100.0) ** 2  # 归一化后平方
        
        distance_penalty *= 5.0  # 距离惩罚权重
        
        # 3. 边界惩罚（防止 UAV 停留在边界）
        boundary_penalty = 0.0
        boundary_margin = area_size * 0.1  # 边界区域宽度
        
        for i in range(num_uavs):
            x, y = self.uav_positions[i]
            
            # 计算到边界的距离
            dist_to_boundary = min(
                x,  # 左边界
                area_size - x,  # 右边界
                y,  # 下边界
                area_size - y  # 上边界
            )
            
            # 如果在边界区域内，增加惩罚
            if dist_to_boundary < boundary_margin:
                boundary_penalty += (1.0 - dist_to_boundary / boundary_margin) ** 2
        
        boundary_penalty *= 3.0  # 边界惩罚权重
        
        # 4. 负载均衡奖励（基于 Jain 指数）
        fairness_reward = jain_index * 2.0
        
        # 5. UAV 负载均衡惩罚
        load_imbalance = np.std(self.uav_load) if np.sum(self.uav_load) > 0 else 0.0
        load_penalty = load_imbalance * 0.5
        
        # 6. 组合奖励
        reward = (
            -delay_penalty          # 时延惩罚（主要）
            - distance_penalty      # 距离惩罚（新增，重要！）
            - boundary_penalty      # 边界惩罚（新增）
            + fairness_reward       # 公平性奖励
            - load_penalty          # 负载均衡惩罚
        )
        
        return reward
    
    def _get_observation(self):
        """获取当前观察"""
        obs = np.concatenate([
            self.uav_positions.flatten(),      # UAV 位置 (6)
            self.user_positions.flatten(),     # 用户位置 (12)
            self.uav_load,                     # UAV 负载 (3)
            [self.prev_total_delay]            # 历史时延 (1)
        ]).astype(np.float32)
        
        return obs
    
    def render(self, mode='human'):
        """可视化（可选）"""
        pass


# ==================== 训练代码（如果需要） ====================
if __name__ == "__main__":
    print("=" * 80)
    print("UAV-RIS 环境测试")
    print("=" * 80)
    
    # 创建环境
    env = UAVEnv()
    print(f"✓ 环境创建成功")
    print(f"  观察空间维度: {env.observation_space.shape}")
    print(f"  动作空间: {env.action_space}")
    
    # 测试环境
    print("\n测试环境运行...")
    obs, info = env.reset()
    print(f"  初始观察: {obs[:6]}... (显示前6个元素)")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {step+1}: Reward={reward:.4f}, Delay={info['total_time']:.4f}s, Jain={info['Jain_step']:.4f}")
        
        if done:
            break
    
    print("\n✓ 环境测试完成！")
    print("=" * 80)
