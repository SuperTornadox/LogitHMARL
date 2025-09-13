import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque, namedtuple
import random
import sys
import os

# 添加utils路径以导入优先经验回放
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.prioritized_replay_buffer import PrioritizedExperienceReplay
except ImportError:
    # 如果导入失败，定义为None，后续会使用普通缓冲区
    PrioritizedExperienceReplay = None

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class FlatMARLDQN:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int = 64,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: int = 100000,
                 batch_size: int = 512,
                 buffer_size: int = 1000000,
                 target_update_freq: int = 1000,
                 grad_clip: float = 1.0,
                 use_double_dqn: bool = True,
                 use_dueling: bool = True,
                 use_prioritized_replay: bool = False,
                 per_alpha: float = 0.6,
                 per_beta: float = 0.4,
                 per_beta_increment: float = 0.001,
                 success_ratio: float = 0.3,
                 device: str = 'cpu'):
        
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self.use_double_dqn = use_double_dqn
        self.device = torch.device(device)
        
        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.steps_done = 0
        
        # Networks
        if use_dueling:
            self.q_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_network = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer (shared across agents for parameter sharing)
        self.use_prioritized_replay = use_prioritized_replay and PrioritizedExperienceReplay is not None
        self.success_ratio = success_ratio
        
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedExperienceReplay(
                capacity=buffer_size,
                alpha=per_alpha,
                beta=per_beta,
                beta_increment=per_beta_increment
            )
            self.last_td_errors = {}  # 存储最近的TD误差
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Metrics
        self.losses = []
        self.q_values = []
        
    def select_action(self, state: np.ndarray, deterministic: bool = False, action_mask: np.ndarray = None) -> int:
        # Update epsilon only during training (not deterministic)
        if not deterministic:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                          np.exp(-self.steps_done / self.epsilon_decay)
            self.steps_done += 1
        
        if not deterministic and random.random() < self.epsilon:
            # If action mask provided, only sample from valid actions
            if action_mask is not None:
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                # Apply action mask if provided
                if action_mask is not None:
                    q_numpy = q_values.cpu().numpy()[0]
                    # Set invalid actions to very low value
                    q_numpy[action_mask == 0] = -float('inf')
                    return np.argmax(q_numpy)
                    
                return q_values.argmax(dim=1).item()
                
    def store_transition(self, state, action, reward, next_state, done, is_success=False):
        """存储经验，支持优先经验回放"""
        if self.use_prioritized_replay:
            # 如果有之前计算的TD误差，使用它；否则使用None让缓冲区使用最大优先级
            td_error = self.last_td_errors.get((state.tobytes(), action), None)
            self.replay_buffer.add_experience(state, action, reward, next_state, done, is_success, td_error)
        else:
            self.replay_buffer.push(state, action, reward, next_state, done)
        
    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # Sample batch
        if self.use_prioritized_replay:
            # 优先经验采样：使用配置的成功经验比例
            experiences, weights, indices = self.replay_buffer.sample_mixed(
                self.batch_size, success_ratio=self.success_ratio
            )
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                next_actions = self.q_network(next_state_batch).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_state_batch).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_state_batch).max(dim=1)[0]
                
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            
        # Compute loss
        if self.use_prioritized_replay:
            # 计算TD误差用于更新优先级
            td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
            
            # 加权损失
            loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
            
            # 更新优先级
            if indices is not None and len(indices) > 0:
                # 只更新了优先缓冲区采样的部分
                valid_indices = indices[:len(td_errors)]
                valid_td_errors = td_errors[:len(valid_indices)]
                self.replay_buffer.update_td_errors(valid_indices, valid_td_errors)
            
            # 存储最近的TD误差以便下次添加经验时使用
            for i, (s, a) in enumerate(zip(batch.state, batch.action)):
                if i < len(td_errors):
                    self.last_td_errors[(s.tobytes(), a)] = abs(td_errors[i])
            
            # 限制存储的TD误差数量
            if len(self.last_td_errors) > 10000:
                # 删除一些旧的
                keys_to_remove = list(self.last_td_errors.keys())[:5000]
                for key in keys_to_remove:
                    del self.last_td_errors[key]
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
            
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Store metrics
        self.losses.append(loss.item())
        self.q_values.append(current_q_values.mean().item())
        
        return loss.item()
        
    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']

class DuelingDQN(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize with bias towards movement actions
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with bias towards movement actions.
        采用环境动作索引: [UP=0, DOWN=1, LEFT=2, RIGHT=3, IDLE=4, PICK=5, DROP=6]
        """
        # Initialize the final layer of advantage stream
        final_layer = self.advantage_stream[-1]
        
        # Small random weights
        nn.init.uniform_(final_layer.weight, -0.01, 0.01)
        
        # Bias initialization: movement actions (0..3) slightly higher, PICK/DROP lower
        # Action indices aligned with environment: [UP=0, DOWN=1, LEFT=2, RIGHT=3, IDLE=4, PICK=5, DROP=6]
        with torch.no_grad():
            if final_layer.out_features == 7:  # Ensure it's the action dimension
                # Movement actions get positive bias, IDLE neutral, PICK/DROP negative
                final_layer.bias.data = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.0, -0.1, -0.1])
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages (Wang et al., 2016)
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values

class MultiAgentDQN:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int = 64,
                 share_params: bool = True,
                 **kwargs):
        
        self.n_agents = n_agents
        self.share_params = share_params
        
        if share_params:
            # Single DQN shared across all agents
            self.dqn = FlatMARLDQN(state_dim, action_dim, n_agents, **kwargs)
        else:
            # Separate DQN for each agent
            self.dqns = [
                FlatMARLDQN(state_dim, action_dim, 1, **kwargs)
                for _ in range(n_agents)
            ]
            
    def select_actions(self, states: Dict[int, np.ndarray], deterministic: bool = False) -> Dict[int, int]:
        actions = {}
        
        if self.share_params:
            for agent_id, state in states.items():
                actions[agent_id] = self.dqn.select_action(state, deterministic)
        else:
            for agent_id, state in states.items():
                if agent_id < self.n_agents:
                    actions[agent_id] = self.dqns[agent_id].select_action(state, deterministic)
                    
        return actions
        
    def store_transitions(self, transitions: List[Tuple]):
        if self.share_params:
            for trans in transitions:
                # Skip agent_id, use only the last 5 elements
                agent_id, state, action, reward, next_state, done = trans
                self.dqn.store_transition(state, action, reward, next_state, done)
        else:
            for agent_id, state, action, reward, next_state, done in transitions:
                if agent_id < self.n_agents:
                    self.dqns[agent_id].store_transition(state, action, reward, next_state, done)
                    
    def train_step(self) -> List[Optional[float]]:
        if self.share_params:
            return [self.dqn.train_step()]
        else:
            return [dqn.train_step() for dqn in self.dqns]
