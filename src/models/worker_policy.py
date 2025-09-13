import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class WorkerPolicy(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 use_lstm: bool = True):
        super().__init__()
        
        self.action_dim = action_dim
        self.use_lstm = use_lstm
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.hidden_dim = hidden_dim
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, obs: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(obs)
        
        if self.use_lstm:
            if hidden_state is None:
                hidden_state = self.init_hidden(obs.shape[0], obs.device)
                
            # Add time dimension if not present
            if len(features.shape) == 2:
                features = features.unsqueeze(1)
                
            features, hidden_state = self.lstm(features, hidden_state)
            features = features.squeeze(1)  # Remove time dimension
            
        # Compute action logits and value
        action_logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        # Compute action probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'value': value,
            'hidden_state': hidden_state if self.use_lstm else None
        }
        
    def select_action(self, obs: torch.Tensor, hidden_state: Optional[Tuple] = None,
                     deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        outputs = self.forward(obs, hidden_state)
        
        if deterministic:
            actions = outputs['action_logits'].argmax(dim=-1)
        else:
            actions = torch.multinomial(outputs['action_probs'], num_samples=1).squeeze(-1)
            
        # Compute log probability
        log_probs = torch.log(outputs['action_probs'] + 1e-8)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        info = {
            'log_prob': action_log_probs,
            'value': outputs['value'],
            'hidden_state': outputs['hidden_state']
        }
        
        return actions, info
        
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
        
    def compute_entropy(self, action_probs: torch.Tensor) -> torch.Tensor:
        return -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)

class CNNWorkerPolicy(nn.Module):
    def __init__(self,
                 grid_size: int,
                 n_channels: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.action_dim = action_dim
        
        # CNN for spatial features
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output dimension
        conv_out_dim = 64 * (grid_size // 2) * (grid_size // 2)
        
        # MLP for combining features
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor and critic heads
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, grid_obs: torch.Tensor, vec_obs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Process grid observation
        conv_features = self.conv_net(grid_obs)
        
        # Process through MLP
        features = self.mlp(conv_features)
        
        # Compute outputs
        action_logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        action_probs = F.softmax(action_logits, dim=-1)
        
        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'value': value
        }
        
    def select_action(self, grid_obs: torch.Tensor, vec_obs: Optional[torch.Tensor] = None,
                     deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        outputs = self.forward(grid_obs, vec_obs)
        
        if deterministic:
            actions = outputs['action_logits'].argmax(dim=-1)
        else:
            actions = torch.multinomial(outputs['action_probs'], num_samples=1).squeeze(-1)
            
        # Compute log probability
        log_probs = torch.log(outputs['action_probs'] + 1e-8)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        info = {
            'log_prob': action_log_probs,
            'value': outputs['value']
        }
        
        return actions, info

class MultiAgentWorkerPolicy(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_dim: int = 256,
                 share_params: bool = True):
        super().__init__()
        
        self.n_agents = n_agents
        self.share_params = share_params
        
        if share_params:
            # Single policy shared across all agents
            self.policy = WorkerPolicy(obs_dim, action_dim, hidden_dim)
        else:
            # Separate policy for each agent
            self.policies = nn.ModuleList([
                WorkerPolicy(obs_dim, action_dim, hidden_dim) 
                for _ in range(n_agents)
            ])
            
    def forward(self, obs: torch.Tensor, agent_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.share_params:
            return self.policy(obs)
        else:
            # Process each agent separately
            outputs = []
            for i in range(self.n_agents):
                agent_obs = obs[agent_ids == i] if agent_ids is not None else obs[i:i+1]
                if agent_obs.shape[0] > 0:
                    outputs.append(self.policies[i](agent_obs))
                    
            # Combine outputs
            combined = {}
            for key in outputs[0].keys():
                if key != 'hidden_state':
                    combined[key] = torch.cat([o[key] for o in outputs], dim=0)
                    
            return combined
            
    def select_action(self, obs: torch.Tensor, agent_ids: Optional[torch.Tensor] = None,
                     deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        if self.share_params:
            return self.policy.select_action(obs, deterministic=deterministic)
        else:
            actions = []
            infos = []
            
            for i in range(self.n_agents):
                agent_obs = obs[agent_ids == i] if agent_ids is not None else obs[i:i+1]
                if agent_obs.shape[0] > 0:
                    action, info = self.policies[i].select_action(agent_obs, deterministic=deterministic)
                    actions.append(action)
                    infos.append(info)
                    
            # Combine results
            actions = torch.cat(actions, dim=0)
            combined_info = {}
            for key in infos[0].keys():
                if key != 'hidden_state':
                    combined_info[key] = torch.cat([info[key] for info in infos], dim=0)
                    
            return actions, combined_info
    
    def compute_entropy(self, action_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of action distribution"""
        # Avoid log(0) by adding small epsilon
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        return entropy.mean()