#!/usr/bin/env python3
"""
优先经验回放缓冲区实现
基于TD误差的优先级采样，提高重要经验的学习效率
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
import random


class SumTree:
    """
    用于高效采样的求和树数据结构
    支持O(log n)的更新和采样操作
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        
    def _propagate(self, idx: int, change: float):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """根据累积和查找叶子节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """返回所有优先级的总和"""
        return self.tree[0]
    
    def add(self, priority: float, data: object):
        """添加新的经验"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """根据累积和获取经验"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    
    特点：
    1. 基于TD误差的优先级采样
    2. 重要性采样权重校正
    3. 支持批量更新优先级
    """
    
    def __init__(self, 
                 capacity: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 0.01):
        """
        参数:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0=均匀采样, 1=完全按优先级)
            beta: 重要性采样权重的初始值
            beta_increment: beta的增长速度
            epsilon: 避免优先级为0的小常数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
    def __len__(self):
        return self.tree.n_entries
    
    def add(self, state, action, reward, next_state, done, td_error: Optional[float] = None):
        """添加新经验"""
        experience = (state, action, reward, next_state, done)
        
        # 如果提供了TD误差，使用它计算优先级
        if td_error is not None:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            # 否则使用最大优先级，确保新经验至少被采样一次
            priority = self.max_priority
        
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        采样一批经验
        
        返回:
            batch: 经验列表
            weights: 重要性采样权重
            indices: 树中的索引（用于更新优先级）
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            if data is not None:
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)
        
        # 计算重要性采样权重
        if len(priorities) > 0:
            priorities = np.array(priorities)
            sampling_probs = priorities / self.tree.total()
            weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
            weights /= weights.max()  # 归一化
        else:
            weights = np.ones(batch_size)
        
        return batch, weights, np.array(indices)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """批量更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def get_statistics(self) -> dict:
        """获取缓冲区统计信息"""
        if self.tree.n_entries == 0:
            return {
                'size': 0,
                'avg_priority': 0,
                'max_priority': 0,
                'min_priority': 0,
                'beta': self.beta
            }
        
        priorities = []
        for i in range(self.tree.n_entries):
            idx = i + self.capacity - 1
            if self.tree.tree[idx] > 0:
                priorities.append(self.tree.tree[idx])
        
        if priorities:
            return {
                'size': self.tree.n_entries,
                'avg_priority': np.mean(priorities),
                'max_priority': np.max(priorities),
                'min_priority': np.min(priorities),
                'beta': self.beta
            }
        else:
            return {
                'size': self.tree.n_entries,
                'avg_priority': 0,
                'max_priority': 0,
                'min_priority': 0,
                'beta': self.beta
            }


class PrioritizedExperienceReplay:
    """
    简化的优先经验回放接口
    专门用于仓库环境的PICK/DROP任务
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, 
                 beta: float = 0.4, beta_increment: float = 0.001):
        self.buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=alpha,
            beta=beta,
            beta_increment=beta_increment
        )
        
        # 专门存储成功经验
        self.success_buffer = []
        self.max_success_buffer = 1000
    
    def __len__(self):
        """返回缓冲区大小"""
        return len(self.buffer)
        
    def add_experience(self, state, action, reward, next_state, done, 
                      is_success: bool = False, td_error: Optional[float] = None):
        """
        添加经验，成功的经验会被特殊处理
        
        参数:
            is_success: 是否是成功的PICK或DROP
            td_error: TD误差，用于计算优先级
        """
        # 添加到主缓冲区
        self.buffer.add(state, action, reward, next_state, done, td_error)
        
        # 成功经验额外存储
        if is_success:
            exp = (state, action, reward, next_state, done)
            self.success_buffer.append(exp)
            if len(self.success_buffer) > self.max_success_buffer:
                self.success_buffer.pop(0)
    
    def sample_mixed(self, batch_size: int, success_ratio: float = 0.3):
        """
        混合采样：一部分来自优先回放，一部分来自成功经验
        
        参数:
            batch_size: 批次大小
            success_ratio: 成功经验的比例
        """
        success_batch_size = int(batch_size * success_ratio)
        priority_batch_size = batch_size - success_batch_size
        
        # 从优先缓冲区采样
        if priority_batch_size > 0 and len(self.buffer) > 0:
            priority_batch, weights, indices = self.buffer.sample(priority_batch_size)
        else:
            priority_batch, weights, indices = [], np.array([]), np.array([])
        
        # 从成功经验采样
        success_batch = []
        success_weights = []
        if success_batch_size > 0 and self.success_buffer:
            n_success = min(success_batch_size, len(self.success_buffer))
            success_batch = random.sample(self.success_buffer, n_success)
            success_weights = np.ones(n_success)  # 成功经验权重为1
        
        # 合并批次
        combined_batch = list(priority_batch) + success_batch
        combined_weights = np.concatenate([weights, success_weights]) if len(weights) > 0 else success_weights
        
        return combined_batch, combined_weights, indices
    
    def update_td_errors(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新TD误差对应的优先级"""
        if len(indices) > 0:
            self.buffer.update_priorities(indices, td_errors)


def test_prioritized_buffer():
    """测试优先经验回放缓冲区"""
    print("测试优先经验回放缓冲区...")
    
    buffer = PrioritizedExperienceReplay(capacity=1000)
    
    # 添加一些经验
    for i in range(100):
        state = np.random.randn(10)
        action = np.random.randint(0, 7)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        
        # 模拟一些成功经验
        is_success = (action == 5 and np.random.random() < 0.3) or \
                    (action == 6 and np.random.random() < 0.1)
        
        td_error = abs(reward) * np.random.random()  # 模拟TD误差
        
        buffer.add_experience(state, action, reward, next_state, done, 
                            is_success, td_error)
    
    # 测试采样
    batch, weights, indices = buffer.sample_mixed(32, success_ratio=0.3)
    
    print(f"缓冲区大小: {len(buffer.buffer)}")
    print(f"成功经验数: {len(buffer.success_buffer)}")
    print(f"采样批次大小: {len(batch)}")
    print(f"权重形状: {weights.shape}")
    
    # 获取统计信息
    stats = buffer.buffer.get_statistics()
    print(f"统计信息: {stats}")
    
    print("测试通过！")


if __name__ == "__main__":
    test_prioritized_buffer()