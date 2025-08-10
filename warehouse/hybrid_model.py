"""
混合模型：Nested Logit（任务分配） + 强化学习（拣货员行为）
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


class NestedLogitManager:
    """
    第一层：使用Nested Logit进行任务分配
    负责决定哪个拣货员执行哪个任务
    """
    
    def __init__(self, num_pickers: int = 6):
        self.num_pickers = num_pickers
        self.picker_ids = [f"P{i}" for i in range(num_pickers)]
        
        # Nested Logit参数（可学习或手动调整）
        self.params = {
            'w_distance': -0.5,      # 距离权重
            'w_workload': -0.3,      # 工作负荷权重  
            'w_priority': 1.0,       # 优先级权重
            'w_equipment': 2.0,      # 设备匹配权重
            'w_idle_time': 0.5,      # 空闲时间权重
            'lambda_urgent': 0.7,    # 紧急任务nest的lambda
            'lambda_regular': 0.5,   # 常规任务nest的lambda
            'nest_constant_forklift': 0.5  # 叉车nest常数
        }
        
    def compute_utility(self, order_features: Dict, picker_features: Dict[str, Dict]) -> Dict[str, float]:
        """计算每个拣货员的效用值"""
        utilities = {}
        
        for picker_id, features in picker_features.items():
            utility = (
                self.params['w_distance'] * (features['distance'] / 50.0) +
                self.params['w_workload'] * (features['workload'] / 20.0) +
                self.params['w_priority'] * (order_features['priority_score'] / 3.0) +
                self.params['w_equipment'] * features['equipment_match'] +
                self.params['w_idle_time'] * (features['idle_time'] / 3600.0)
            )
            utilities[picker_id] = utility
            
        return utilities
        
    def assign_task(self, order_features: Dict, picker_features: Dict[str, Dict], 
                   picker_states: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        分配任务给拣货员
        
        返回:
        - chosen_picker: 选中的拣货员ID
        - info: 包含概率分布等信息的字典
        """
        # 计算效用
        utilities = self.compute_utility(order_features, picker_features)
        
        # 确定nest结构
        nests = self._determine_nests(order_features, picker_features)
        
        # 计算nest层概率
        nest_probs = self._compute_nest_probabilities(utilities, nests)
        
        # 计算最终的拣货员选择概率
        picker_probs = self._compute_picker_probabilities(utilities, nests, nest_probs)
        
        # 如果提供了拣货员状态（来自强化学习层），可以调整概率
        if picker_states:
            picker_probs = self._adjust_probabilities_with_rl_states(
                picker_probs, picker_states, order_features
            )
        
        # 选择拣货员
        chosen_picker = self._sample_picker(picker_probs)
        
        info = {
            'utilities': utilities,
            'nest_probs': nest_probs,
            'picker_probs': picker_probs,
            'nests': nests
        }
        
        return chosen_picker, info
        
    def _determine_nests(self, order_features: Dict, picker_features: Dict[str, Dict]) -> Dict[str, List[str]]:
        """确定nest结构"""
        nests = {
            'urgent_normal': [],
            'urgent_forklift': [],
            'regular_normal': [],
            'regular_forklift': []
        }
        
        for picker_id, features in picker_features.items():
            if order_features['priority_score'] >= 2:  # urgent
                if order_features['need_forklift'] and features['has_forklift']:
                    nests['urgent_forklift'].append(picker_id)
                else:
                    nests['urgent_normal'].append(picker_id)
            else:  # regular
                if order_features['need_forklift'] and features['has_forklift']:
                    nests['regular_forklift'].append(picker_id)
                else:
                    nests['regular_normal'].append(picker_id)
                    
        return nests
        
    def _compute_nest_probabilities(self, utilities: Dict[str, float], 
                                   nests: Dict[str, List[str]]) -> Dict[str, float]:
        """计算nest层的选择概率"""
        nest_values = {}
        
        for nest_name, picker_ids in nests.items():
            if not picker_ids:
                continue
                
            # 选择合适的lambda
            if 'urgent' in nest_name:
                lambda_val = self.params['lambda_urgent']
            else:
                lambda_val = self.params['lambda_regular']
                
            # 计算inclusive value
            nest_utilities = [utilities[pid] for pid in picker_ids]
            if 'forklift' in nest_name:
                nest_utilities = [u + self.params['nest_constant_forklift'] for u in nest_utilities]
                
            iv = lambda_val * np.log(sum(np.exp(u / lambda_val) for u in nest_utilities))
            nest_values[nest_name] = iv
            
        # 计算nest概率
        if not nest_values:
            return {}
            
        max_iv = max(nest_values.values())
        exp_values = {n: np.exp(v - max_iv) for n, v in nest_values.items()}
        total = sum(exp_values.values())
        
        return {n: v/total for n, v in exp_values.items()}
        
    def _compute_picker_probabilities(self, utilities: Dict[str, float], 
                                    nests: Dict[str, List[str]], 
                                    nest_probs: Dict[str, float]) -> Dict[str, float]:
        """计算每个拣货员的最终选择概率"""
        picker_probs = {pid: 0.0 for pid in utilities.keys()}
        
        for nest_name, nest_prob in nest_probs.items():
            picker_ids = nests[nest_name]
            if not picker_ids:
                continue
                
            # 在nest内的条件概率
            lambda_val = (self.params['lambda_urgent'] if 'urgent' in nest_name 
                         else self.params['lambda_regular'])
            
            nest_utilities = {pid: utilities[pid] for pid in picker_ids}
            if 'forklift' in nest_name:
                nest_utilities = {pid: u + self.params['nest_constant_forklift'] 
                                for pid, u in nest_utilities.items()}
                
            # Softmax within nest
            max_u = max(nest_utilities.values())
            exp_utilities = {pid: np.exp((u - max_u) / lambda_val) 
                           for pid, u in nest_utilities.items()}
            total = sum(exp_utilities.values())
            
            for pid, exp_u in exp_utilities.items():
                picker_probs[pid] = nest_prob * (exp_u / total)
                
        return picker_probs
        
    def _adjust_probabilities_with_rl_states(self, picker_probs: Dict[str, float], 
                                           picker_states: Dict[str, Dict],
                                           order_features: Dict) -> Dict[str, float]:
        """根据强化学习层的状态信息调整概率"""
        # 这里可以根据拣货员的RL状态（如当前策略价值、历史表现等）来微调概率
        adjusted_probs = picker_probs.copy()
        
        for picker_id, state in picker_states.items():
            if picker_id in adjusted_probs:
                # 根据RL层的价值估计调整
                if 'value_estimate' in state:
                    adjustment = 1 + 0.1 * np.tanh(state['value_estimate'])
                    adjusted_probs[picker_id] *= adjustment
                    
        # 重新归一化
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {pid: p/total for pid, p in adjusted_probs.items()}
            
        return adjusted_probs
        
    def _sample_picker(self, picker_probs: Dict[str, float]) -> str:
        """根据概率分布选择拣货员"""
        pickers = list(picker_probs.keys())
        probs = list(picker_probs.values())
        
        # 确保概率和为1
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        return np.random.choice(pickers, p=probs)


class RLPicker(nn.Module):
    """
    第二层：拣货员的强化学习智能体
    负责学习最优的拣货策略（路径规划、动作选择等）
    """
    
    def __init__(self, state_dim: int = 10, action_dim: int = 5, hidden_dim: int = 64):
        super(RLPicker, self).__init__()
        
        # Actor网络（策略）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络（价值函数）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """前向传播"""
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
        
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.forward(state_tensor)
        
        # 采样动作
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value


class HybridWarehouseSystem:
    """
    混合仓库系统：结合Nested Logit任务分配和RL拣货员
    """
    
    def __init__(self, num_pickers: int = 6):
        self.num_pickers = num_pickers
        
        # 第一层：Nested Logit管理器
        self.manager = NestedLogitManager(num_pickers)
        
        # 第二层：每个拣货员的RL智能体
        self.pickers = {}
        for i in range(num_pickers):
            picker_id = f"P{i}"
            self.pickers[picker_id] = {
                'model': RLPicker(state_dim=10, action_dim=5),
                'optimizer': optim.Adam(RLPicker(state_dim=10, action_dim=5).parameters(), lr=1e-3),
                'memory': deque(maxlen=1000),
                'position': (1, 0),
                'workload': 0,
                'total_reward': 0
            }
            
    def process_order(self, order_features: Dict, picker_features: Dict[str, Dict], 
                     warehouse_state: Dict) -> Dict:
        """
        处理订单：任务分配 + 执行
        """
        # 获取每个拣货员的RL状态估值
        picker_states = {}
        for picker_id, picker_data in self.pickers.items():
            # 构建拣货员当前状态
            state = self._build_picker_state(picker_id, warehouse_state)
            _, value = picker_data['model'].forward(torch.FloatTensor(state).unsqueeze(0))
            picker_states[picker_id] = {
                'value_estimate': value.item(),
                'current_workload': picker_data['workload']
            }
            
        # 第一层：使用Nested Logit分配任务
        chosen_picker, assignment_info = self.manager.assign_task(
            order_features, picker_features, picker_states
        )
        
        # 第二层：选中的拣货员执行任务
        execution_result = self._execute_picking_task(
            chosen_picker, order_features, warehouse_state
        )
        
        return {
            'chosen_picker': chosen_picker,
            'assignment_info': assignment_info,
            'execution_result': execution_result
        }
        
    def _build_picker_state(self, picker_id: str, warehouse_state: Dict) -> np.ndarray:
        """构建拣货员的状态向量"""
        picker_data = self.pickers[picker_id]
        
        state = [
            picker_data['position'][0] / 32.0,  # 归一化x坐标
            picker_data['position'][1] / 20.0,  # 归一化y坐标
            picker_data['workload'] / 20.0,     # 归一化工作负载
            warehouse_state.get('congestion', 0),  # 拥堵程度
            warehouse_state.get('time_pressure', 0),  # 时间压力
            # 可以添加更多状态特征
        ]
        
        # 填充到固定维度
        while len(state) < 10:
            state.append(0)
            
        return np.array(state[:10])
        
    def _execute_picking_task(self, picker_id: str, order_features: Dict, 
                            warehouse_state: Dict) -> Dict:
        """
        拣货员执行任务（使用RL策略）
        """
        picker_data = self.pickers[picker_id]
        trajectory = []
        total_reward = 0
        
        # 模拟任务执行过程
        current_pos = picker_data['position']
        target_pos = order_features['pickup_location']
        
        steps = 0
        while current_pos != target_pos and steps < 100:
            # 构建状态
            state = self._build_picker_state(picker_id, warehouse_state)
            
            # RL智能体选择动作
            action, log_prob, value = picker_data['model'].select_action(state)
            
            # 执行动作（简化的移动逻辑）
            next_pos = self._move(current_pos, action, target_pos)
            
            # 计算奖励
            reward = self._calculate_step_reward(current_pos, next_pos, target_pos)
            
            # 记录轨迹
            trajectory.append({
                'position': current_pos,
                'action': action,
                'reward': reward
            })
            
            # 存储经验
            next_state = self._build_picker_state(picker_id, warehouse_state)
            picker_data['memory'].append((state, action, reward, next_state, False))
            
            current_pos = next_pos
            total_reward += reward
            steps += 1
            
        # 更新拣货员状态
        picker_data['position'] = target_pos
        picker_data['workload'] += 1
        picker_data['total_reward'] += total_reward
        
        return {
            'steps': steps,
            'total_reward': total_reward,
            'trajectory': trajectory,
            'success': current_pos == target_pos
        }
        
    def _move(self, current_pos: Tuple[int, int], action: int, 
             target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """根据动作移动（简化版本）"""
        x, y = current_pos
        tx, ty = target_pos
        
        # 动作映射：0=停留, 1=上, 2=下, 3=左, 4=右
        if action == 0:
            return current_pos
        elif action == 1 and y > 0:
            return (x, y - 1)
        elif action == 2 and y < 19:
            return (x, y + 1)
        elif action == 3 and x > 0:
            return (x - 1, y)
        elif action == 4 and x < 31:
            return (x + 1, y)
        else:
            # 贪心移动向目标
            if abs(tx - x) > abs(ty - y):
                return (x + np.sign(tx - x), y)
            else:
                return (x, y + np.sign(ty - y))
                
    def _calculate_step_reward(self, current_pos: Tuple[int, int], 
                             next_pos: Tuple[int, int], 
                             target_pos: Tuple[int, int]) -> float:
        """计算单步奖励"""
        # 当前距离
        current_dist = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])
        # 下一步距离
        next_dist = abs(next_pos[0] - target_pos[0]) + abs(next_pos[1] - target_pos[1])
        
        # 基础奖励：向目标靠近得正奖励，远离得负奖励
        reward = current_dist - next_dist
        
        # 到达目标的额外奖励
        if next_pos == target_pos:
            reward += 10
            
        return reward
        
    def train_pickers(self, batch_size: int = 32):
        """训练所有拣货员的RL模型"""
        for picker_id, picker_data in self.pickers.items():
            if len(picker_data['memory']) < batch_size:
                continue
                
            # 采样批次
            # 如果内存较大，使用更大的batch
            actual_batch_size = min(batch_size * 2, len(picker_data['memory']))
            batch = random.sample(picker_data['memory'], actual_batch_size)
            states = torch.FloatTensor([e[0] for e in batch])
            actions = torch.LongTensor([e[1] for e in batch])
            rewards = torch.FloatTensor([e[2] for e in batch])
            next_states = torch.FloatTensor([e[3] for e in batch])
            dones = torch.FloatTensor([e[4] for e in batch])
            
            # 计算损失并更新
            # 这里使用简化的Actor-Critic更新
            action_probs, values = picker_data['model'](states)
            _, next_values = picker_data['model'](next_states)
            
            # TD误差
            td_targets = rewards + 0.99 * next_values.squeeze() * (1 - dones)
            td_errors = td_targets - values.squeeze()
            
            # Actor损失
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
            actor_loss = -(log_probs * td_errors.detach()).mean()
            
            # Critic损失  
            critic_loss = td_errors.pow(2).mean()
            
            # 总损失
            loss = actor_loss + critic_loss
            
            picker_data['optimizer'].zero_grad()
            loss.backward()
            picker_data['optimizer'].step()


# 使用示例
if __name__ == "__main__":
    # 创建混合系统
    system = HybridWarehouseSystem(num_pickers=6)
    
    # 模拟订单处理
    order = {
        'priority_score': 3,
        'num_items': 5,
        'pickup_location': (15, 10),
        'need_forklift': 0
    }
    
    picker_features = {
        f"P{i}": {
            'distance': np.random.randint(5, 30),
            'workload': np.random.randint(0, 10),
            'has_forklift': 1 if i == 3 else 0,
            'equipment_match': 1 if i == 3 else 0,
            'idle_time': np.random.randint(0, 300)
        }
        for i in range(6)
    }
    
    warehouse_state = {
        'congestion': 0.3,
        'time_pressure': 0.7
    }
    
    # 处理订单
    result = system.process_order(order, picker_features, warehouse_state)
    
    print(f"分配给: {result['chosen_picker']}")
    print(f"分配概率: {result['assignment_info']['picker_probs']}")
    print(f"执行结果: 步数={result['execution_result']['steps']}, "
          f"奖励={result['execution_result']['total_reward']:.2f}")