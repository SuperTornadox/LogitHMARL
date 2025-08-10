"""
HMARL (Hierarchical Multi-Agent Reinforcement Learning) 模型实现
使用Softmax策略的强化学习任务分配
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """策略网络：输入状态，输出动作概率分布"""
    
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """价值网络：评估状态价值"""
    
    def __init__(self, state_dim: int, hidden_dim: int):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        value = self.fc3(x)
        return value


class HMARLModel:
    """
    分层多智能体强化学习模型
    - 上层：决定任务类型（urgent/regular, forklift/normal）
    - 下层：选择具体拣货员
    """
    
    def __init__(self, num_pickers: int = 6, state_dim: int = 15, 
                 hidden_dim: int = 128, learning_rate: float = 1e-3):
        self.num_pickers = num_pickers
        self.picker_ids = [f"P{i}" for i in range(num_pickers)]
        
        # 状态维度：订单特征(5) + 每个拣货员特征(5) * num_pickers
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # 上层网络：选择任务类型（4种组合）
        self.high_level_policy = PolicyNetwork(state_dim, hidden_dim, 4)
        self.high_level_value = ValueNetwork(state_dim, hidden_dim)
        
        # 下层网络：选择拣货员
        self.low_level_policy = PolicyNetwork(state_dim + 4, hidden_dim, num_pickers)
        self.low_level_value = ValueNetwork(state_dim + 4, hidden_dim)
        
        # 优化器
        self.high_policy_optimizer = optim.Adam(self.high_level_policy.parameters(), lr=learning_rate)
        self.high_value_optimizer = optim.Adam(self.high_level_value.parameters(), lr=learning_rate)
        self.low_policy_optimizer = optim.Adam(self.low_level_policy.parameters(), lr=learning_rate)
        self.low_value_optimizer = optim.Adam(self.low_level_value.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # 训练参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def extract_state(self, order_features: Dict, picker_features: Dict[str, Dict]) -> np.ndarray:
        """从订单和拣货员特征中提取状态向量"""
        state = []
        
        # 订单特征
        state.extend([
            order_features['priority_score'] / 3.0,  # 归一化
            order_features['num_items'] / 10.0,
            order_features['pickup_x'] / 32.0,
            order_features['pickup_y'] / 20.0,
            float(order_features['need_forklift'])
        ])
        
        # 拣货员特征（按ID排序以保证顺序一致）
        for picker_id in self.picker_ids:
            if picker_id in picker_features:
                pf = picker_features[picker_id]
                state.extend([
                    pf['distance'] / 50.0,  # 归一化距离
                    pf['workload'] / 20.0,   # 归一化工作负载
                    float(pf['has_forklift']),
                    float(pf['equipment_match']),
                    pf['idle_time'] / 3600.0  # 归一化空闲时间（小时）
                ])
            else:
                state.extend([0, 0, 0, 0, 0])  # 默认值
                
        return np.array(state, dtype=np.float32)
        
    def get_task_type_encoding(self, task_type: int) -> np.ndarray:
        """任务类型的one-hot编码"""
        encoding = np.zeros(4)
        encoding[task_type] = 1
        return encoding
        
    def choose_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, int, Dict]:
        """选择动作（任务类型和拣货员）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 上层决策：选择任务类型
        if training and random.random() < self.epsilon:
            task_type = random.randint(0, 3)
        else:
            with torch.no_grad():
                high_probs = self.high_level_policy(state_tensor)
                task_type = torch.multinomial(high_probs, 1).item()
                
        # 准备下层状态（原始状态 + 任务类型编码）
        task_encoding = self.get_task_type_encoding(task_type)
        low_state = np.concatenate([state, task_encoding])
        low_state_tensor = torch.FloatTensor(low_state).unsqueeze(0)
        
        # 下层决策：选择拣货员
        if training and random.random() < self.epsilon:
            picker_idx = random.randint(0, self.num_pickers - 1)
        else:
            with torch.no_grad():
                low_probs = self.low_level_policy(low_state_tensor)
                picker_idx = torch.multinomial(low_probs, 1).item()
                
        # 计算动作概率（用于分析）
        with torch.no_grad():
            high_probs = self.high_level_policy(state_tensor).squeeze().numpy()
            low_probs = self.low_level_policy(low_state_tensor).squeeze().numpy()
            
        info = {
            'task_type': task_type,
            'task_type_probs': high_probs,
            'picker_probs': low_probs,
            'picker_id': self.picker_ids[picker_idx]
        }
        
        return task_type, picker_idx, info
        
    def remember(self, state: np.ndarray, task_type: int, picker_idx: int, 
                 reward: float, next_state: np.ndarray, done: bool):
        """存储经验"""
        self.memory.append((state, task_type, picker_idx, reward, next_state, done))
        
    def compute_reward(self, order_features: Dict, picker_features: Dict, 
                      outcome: Dict) -> float:
        """计算奖励函数"""
        # 基础奖励
        reward = 0
        
        # 完成时间奖励（越快越好）
        time_reward = 300 / outcome['completion_time']  # 基准300秒
        reward += time_reward
        
        # 距离惩罚
        distance_penalty = -outcome['travel_distance'] * 0.1
        reward += distance_penalty
        
        # 优先级匹配奖励
        if order_features['priority'] == 'high':
            if outcome['completion_time'] < 180:  # 3分钟内完成
                reward += 2
                
        # 设备匹配奖励
        if order_features['need_forklift'] and picker_features['has_forklift']:
            reward += 1
            
        # 负载均衡奖励（工作负载低的拣货员获得额外奖励）
        if picker_features['workload'] < 5:
            reward += 0.5
            
        return reward
        
    def train(self, epochs: int = 100):
        """训练模型"""
        if len(self.memory) < self.batch_size:
            return
            
        for epoch in range(epochs):
            # 采样批次
            batch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor([e[0] for e in batch])
            task_types = torch.LongTensor([e[1] for e in batch])
            picker_indices = torch.LongTensor([e[2] for e in batch])
            rewards = torch.FloatTensor([e[3] for e in batch])
            next_states = torch.FloatTensor([e[4] for e in batch])
            dones = torch.FloatTensor([e[5] for e in batch])
            
            # 训练上层网络
            high_values = self.high_level_value(states).squeeze()
            next_high_values = self.high_level_value(next_states).squeeze()
            high_targets = rewards + self.gamma * next_high_values * (1 - dones)
            high_value_loss = F.mse_loss(high_values, high_targets.detach())
            
            self.high_value_optimizer.zero_grad()
            high_value_loss.backward()
            self.high_value_optimizer.step()
            
            # 策略梯度更新
            high_probs = self.high_level_policy(states)
            high_log_probs = torch.log(high_probs.gather(1, task_types.unsqueeze(1)).squeeze())
            high_advantages = high_targets - high_values.detach()
            high_policy_loss = -(high_log_probs * high_advantages).mean()
            
            self.high_policy_optimizer.zero_grad()
            high_policy_loss.backward()
            self.high_policy_optimizer.step()
            
            # 训练下层网络
            # 构建下层状态
            task_encodings = torch.zeros(self.batch_size, 4)
            for i, tt in enumerate(task_types):
                task_encodings[i, tt] = 1
            low_states = torch.cat([states, task_encodings], dim=1)
            
            low_values = self.low_level_value(low_states).squeeze()
            # 简化：使用相同的目标
            low_targets = high_targets
            low_value_loss = F.mse_loss(low_values, low_targets.detach())
            
            self.low_value_optimizer.zero_grad()
            low_value_loss.backward()
            self.low_value_optimizer.step()
            
            # 下层策略更新
            low_probs = self.low_level_policy(low_states)
            low_log_probs = torch.log(low_probs.gather(1, picker_indices.unsqueeze(1)).squeeze())
            low_advantages = low_targets - low_values.detach()
            low_policy_loss = -(low_log_probs * low_advantages).mean()
            
            self.low_policy_optimizer.zero_grad()
            low_policy_loss.backward()
            self.low_policy_optimizer.step()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: High Policy Loss: {high_policy_loss.item():.4f}, "
                           f"Low Policy Loss: {low_policy_loss.item():.4f}")
                
        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def predict_probabilities(self, order_features: Dict, picker_features: Dict[str, Dict]) -> Dict[str, float]:
        """预测每个拣货员的选择概率"""
        state = self.extract_state(order_features, picker_features)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            # 获取任务类型概率
            high_probs = self.high_level_policy(state_tensor).squeeze().numpy()
            
            # 对每种任务类型，计算拣货员概率
            picker_probs = np.zeros(self.num_pickers)
            
            for task_type in range(4):
                task_encoding = self.get_task_type_encoding(task_type)
                low_state = np.concatenate([state, task_encoding])
                low_state_tensor = torch.FloatTensor(low_state).unsqueeze(0)
                
                low_probs = self.low_level_policy(low_state_tensor).squeeze().numpy()
                
                # 加权平均
                picker_probs += high_probs[task_type] * low_probs
                
        return {self.picker_ids[i]: picker_probs[i] for i in range(self.num_pickers)}
        
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'high_policy_state': self.high_level_policy.state_dict(),
            'high_value_state': self.high_level_value.state_dict(),
            'low_policy_state': self.low_level_policy.state_dict(),
            'low_value_state': self.low_level_value.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.high_level_policy.load_state_dict(checkpoint['high_policy_state'])
        self.high_level_value.load_state_dict(checkpoint['high_value_state'])
        self.low_level_policy.load_state_dict(checkpoint['low_policy_state'])
        self.low_level_value.load_state_dict(checkpoint['low_value_state'])
        self.epsilon = checkpoint['epsilon']