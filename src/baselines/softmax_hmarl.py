import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from models.worker_policy import MultiAgentWorkerPolicy

class SoftmaxManager(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_tasks: int,
                 hidden_dim: int = 256,
                 embed_dim: int = 128,
                 task_feature_dim: int = 6):
        super().__init__()
        
        self.n_tasks = n_tasks
        
        # Task utility network (same as NL-HMARL)
        self.task_utility_net = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Task embedding network
        self.task_embedder = nn.Linear(task_feature_dim, embed_dim)
        
        # Global encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def compute_task_utilities(self, state: torch.Tensor, task_features: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        n_tasks = task_features.shape[1]
        
        # Embed tasks
        task_embeds = self.task_embedder(task_features)  # [B, T, E]
        
        # Expand state for each task
        state_expanded = state.unsqueeze(1).expand(-1, n_tasks, -1)  # [B, T, S]
        
        # Concatenate state and task embeddings
        inputs = torch.cat([state_expanded, task_embeds], dim=-1)  # [B, T, S+E]
        
        # Compute utilities
        utilities = self.task_utility_net(inputs).squeeze(-1)  # [B, T]
        
        return utilities
        
    def forward(self, state: torch.Tensor, task_features: torch.Tensor, 
                task_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Compute task utilities
        utilities = self.compute_task_utilities(state, task_features)
        
        # Apply mask if provided
        if task_mask is not None:
            utilities = utilities.masked_fill(~task_mask, -float('inf'))
            
        # Check if all tasks are masked
        if torch.all(torch.isinf(utilities)):
            # If all tasks are masked, use uniform distribution
            task_probs = torch.ones_like(utilities) / utilities.shape[1]
        else:
            # Simple softmax over all tasks (no nesting)
            task_probs = F.softmax(utilities, dim=1)
        
        return {
            'task_utilities': utilities,
            'task_probs': task_probs
        }
        
    def select_task(self, state: torch.Tensor, task_features: torch.Tensor,
                   task_mask: Optional[torch.Tensor] = None,
                   deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        outputs = self.forward(state, task_features, task_mask)
        probs = outputs['task_probs']  # [B,T]
        # Sanitize: remove NaN/Inf/negatives; mask invalid; renormalize; uniform fallback
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)
        if task_mask is not None:
            probs = probs * task_mask.float()
        sums = probs.sum(dim=1, keepdim=True)
        zero_rows = (sums <= 1e-12).squeeze(1)
        if zero_rows.any():
            if task_mask is not None:
                valid_counts = task_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                uniform = (task_mask.float() / valid_counts)
            else:
                T = probs.shape[1]
                uniform = torch.full_like(probs, 1.0 / max(1, T))
            probs[zero_rows] = uniform[zero_rows]
            sums = probs.sum(dim=1, keepdim=True)
        probs = probs / sums.clamp(min=1e-12)

        if deterministic:
            task_indices = probs.argmax(dim=1)
        else:
            task_indices = torch.multinomial(torch.clamp(probs, min=1e-8), num_samples=1).squeeze(1)
        info = {
            'task_probs': probs,
            'utilities': outputs['task_utilities']
        }
        return task_indices, info
        
    def compute_entropy(self, task_probs: torch.Tensor) -> torch.Tensor:
        # Shannon entropy
        return -torch.sum(task_probs * torch.log(task_probs + 1e-8), dim=1)

class SoftmaxHMARL:
    def __init__(self,
                 state_dim: int,
                 n_tasks: int,
                 n_agents: int,
                 worker_obs_dim: int,
                 worker_action_dim: int,
                 hidden_dim: int = 256,
                 device: str = 'cpu',
                 task_feature_dim: int = 6):
        
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        
        # Manager (categorical softmax instead of nested logit)
        self.manager = SoftmaxManager(
            state_dim=state_dim,
            n_tasks=n_tasks,
            hidden_dim=hidden_dim,
            embed_dim=128,
            task_feature_dim=task_feature_dim
        ).to(self.device)
        
        # Workers (same as NL-HMARL)
        # MultiAgentWorkerPolicy should already be loaded via exec()
        self.workers = MultiAgentWorkerPolicy(
            obs_dim=worker_obs_dim,
            action_dim=worker_action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            share_params=True
        ).to(self.device)
        
        # Value network for manager
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
    def manager_forward(self, state: torch.Tensor, task_features: torch.Tensor,
                       task_mask: Optional[torch.Tensor] = None) -> Dict:
        
        # Manager decision
        manager_outputs = self.manager(state, task_features, task_mask)
        
        # Value prediction
        value = self.value_net(state).squeeze(-1)
        manager_outputs['value'] = value
        
        return manager_outputs
        
    def select_tasks(self, state: torch.Tensor, task_features: torch.Tensor,
                    task_mask: Optional[torch.Tensor] = None,
                    deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        
        task_indices, info = self.manager.select_task(
            state, task_features, task_mask, deterministic
        )
        
        # Add value to info
        with torch.no_grad():
            info['value'] = self.value_net(state).squeeze(-1)
            
        return task_indices, info
        
    def worker_forward(self, obs: torch.Tensor, agent_ids: Optional[torch.Tensor] = None) -> Dict:
        return self.workers(obs, agent_ids)
        
    def select_worker_actions(self, obs: torch.Tensor, agent_ids: Optional[torch.Tensor] = None,
                            deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        return self.workers.select_action(obs, agent_ids, deterministic)
        
    def compute_manager_loss(self, 
                           states: torch.Tensor,
                           task_features: torch.Tensor,
                           selected_tasks: torch.Tensor,
                           advantages: torch.Tensor,
                           returns: torch.Tensor,
                           task_mask: Optional[torch.Tensor] = None,
                           entropy_coef: float = 0.01) -> Dict[str, torch.Tensor]:
        
        # Forward pass
        outputs = self.manager_forward(states, task_features, task_mask)
        
        # Policy loss
        log_probs = torch.log(outputs['task_probs'] + 1e-8)
        selected_log_probs = log_probs.gather(1, selected_tasks.unsqueeze(1)).squeeze(1)
        policy_loss = -(advantages * selected_log_probs).mean()

        # Value loss - FIXED: Use Huber loss for stability
        value_loss = F.huber_loss(outputs['value'], returns, delta=10.0)
        
        # Entropy regularization
        entropy = self.manager.compute_entropy(outputs['task_probs'])
        entropy_loss = -entropy_coef * entropy.mean()
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'entropy': entropy.mean()
        }
        
    def save(self, path: str):
        torch.save({
            'manager': self.manager.state_dict(),
            'workers': self.workers.state_dict(),
            'value_net': self.value_net.state_dict()
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.manager.load_state_dict(checkpoint['manager'])
        self.workers.load_state_dict(checkpoint['workers'])
        self.value_net.load_state_dict(checkpoint['value_net'])
