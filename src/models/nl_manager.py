import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class NestConfig:
    nest_id: int
    zone: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    task_type: str  # 'regular', 'heavy', 'urgent'
    
class NestedLogitManager(nn.Module):
    def __init__(self, 
                 state_dim: int,
                 n_tasks: int,
                 n_nests: int,
                 hidden_dim: int = 256,
                 embed_dim: int = 128,
                 eta_min: float = 0.1,
                 eta_init: float = 0.5):
        super().__init__()
        
        self.n_tasks = n_tasks
        self.n_nests = n_nests
        self.eta_min = eta_min
        
        # Task utility network u_θ(s_t, i)
        self.task_utility_net = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Task embedding network
        self.task_embedder = nn.Linear(5, embed_dim)  # 5 task features
        
        # Nest score network b_φ(s_t, m)
        self.nest_score_net = nn.Sequential(
            nn.Linear(embed_dim * 2 + n_nests, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Nest dissimilarity parameters λ_m
        self.lambda_params = nn.Parameter(torch.ones(n_nests) * np.log(eta_init))
        
        # Global context encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Set aggregation for nest representation
        self.set_aggregator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU()
        )
        
        # Initialize weights for stability
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)  # Moderate gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
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
        
    def compute_nest_dissimilarities(self) -> torch.Tensor:
        # η_m = σ(λ_m) ∈ (0, 1]
        eta = torch.sigmoid(self.lambda_params)
        # Clip to ensure stability
        eta = torch.clamp(eta, min=self.eta_min, max=1.0)
        return eta
        
    def compute_inclusive_values(self, utilities: torch.Tensor, nest_masks: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        # I_m = log Σ_{j∈N_t^m} exp(u_j / η_m)
        batch_size = utilities.shape[0]
        inclusive_values = []
        
        for m in range(self.n_nests):
            mask = nest_masks[:, m, :]  # [B, T]
            eta_m = eta[m].clamp(min=0.1, max=1.0)  # Ensure eta is in valid range
            
            # Check if nest has any tasks
            if not mask.any():
                # Empty nest, set inclusive value to a small negative value
                I_m = torch.full((batch_size,), -10.0, device=utilities.device)
            else:
                # Masked log-sum-exp with numerical stability
                masked_utils = utilities.clone()
                masked_utils[mask == 0] = -1e10  # Use large negative but not inf
                
                # Normalize utilities for numerical stability
                max_util = masked_utils.max(dim=1, keepdim=True)[0]
                normalized_utils = masked_utils - max_util
                
                # Compute inclusive value for nest m with stability
                I_m = max_util.squeeze() + torch.logsumexp(normalized_utils / eta_m, dim=1)
                
                # Clamp to reasonable range
                I_m = I_m.clamp(min=-100, max=100)
                
            inclusive_values.append(I_m)
            
        inclusive_values = torch.stack(inclusive_values, dim=1)  # [B, M]
        return inclusive_values
        
    def compute_nest_scores(self, state: torch.Tensor, task_features: torch.Tensor, nest_masks: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        
        # Global context
        global_context = self.global_encoder(state)  # [B, E]
        
        # Task embeddings
        task_embeds = self.task_embedder(task_features)  # [B, T, E]
        
        nest_scores = []
        for m in range(self.n_nests):
            mask = nest_masks[:, m, :].unsqueeze(-1)  # [B, T, 1]
            
            # Aggregate task embeddings in nest m
            masked_embeds = task_embeds * mask
            
            # Set aggregation: mean and max pooling
            count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_pool = masked_embeds.sum(dim=1) / count.squeeze(-1)  # [B, E]
            max_pool = masked_embeds.max(dim=1)[0]  # [B, E]
            
            # Combine pooled features
            pooled = self.set_aggregator(torch.cat([mean_pool, max_pool], dim=-1))  # [B, E]
            
            # Nest identifier (one-hot)
            nest_id = torch.zeros(batch_size, self.n_nests, device=state.device)
            nest_id[:, m] = 1
            
            # Concatenate all features for nest score
            nest_input = torch.cat([global_context, pooled, nest_id], dim=-1)
            score = self.nest_score_net(nest_input).squeeze(-1)
            nest_scores.append(score)
            
        nest_scores = torch.stack(nest_scores, dim=1)  # [B, M]
        return nest_scores
        
    def forward(self, state: torch.Tensor, task_features: torch.Tensor, nest_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Compute task utilities u_i
        utilities = self.compute_task_utilities(state, task_features)
        
        # Compute nest dissimilarities η_m
        eta = self.compute_nest_dissimilarities()
        
        # Compute inclusive values I_m
        inclusive_values = self.compute_inclusive_values(utilities, nest_masks, eta)
        
        # Compute nest scores b_m
        nest_scores = self.compute_nest_scores(state, task_features, nest_masks)
        
        # Nest-level probabilities: π_nest(m|s_t) = exp(b_m + η_m * I_m) / Σ_n exp(b_n + η_n * I_n)
        nest_logits = nest_scores + eta.unsqueeze(0) * inclusive_values
        # Clamp logits for numerical stability
        nest_logits = nest_logits.clamp(min=-50, max=50)
        nest_probs = F.softmax(nest_logits, dim=1)
        
        # Task-level probabilities within each nest
        task_probs_per_nest = []
        for m in range(self.n_nests):
            mask = nest_masks[:, m, :]
            eta_m = eta[m]
            
            # π_task(i|s_t, m) = exp(u_i / η_m) / Σ_{j∈N_t^m} exp(u_j / η_m)
            masked_utils = utilities.clone()
            masked_utils[mask == 0] = -1e10  # Use large negative but not inf
            # Clamp eta_m to avoid division issues
            eta_m_safe = eta_m.clamp(min=0.1)
            task_probs_m = F.softmax(masked_utils / eta_m_safe, dim=1)
            task_probs_per_nest.append(task_probs_m)
            
        task_probs_per_nest = torch.stack(task_probs_per_nest, dim=1)  # [B, M, T]
        
        # Joint probabilities: π_M(i|s_t) = Σ_{m: i∈N_t^m} π_nest(m|s_t) * π_task(i|s_t, m)
        joint_probs = torch.zeros_like(utilities)
        for m in range(self.n_nests):
            mask = nest_masks[:, m, :]
            joint_probs += nest_probs[:, m:m+1] * task_probs_per_nest[:, m, :] * mask.float()
            
        return {
            'task_utilities': utilities,
            'nest_probs': nest_probs,
            'task_probs_per_nest': task_probs_per_nest,
            'joint_probs': joint_probs,
            'eta': eta,
            'inclusive_values': inclusive_values,
            'nest_scores': nest_scores
        }
        
    def select_task(self, state: torch.Tensor, task_features: torch.Tensor, nest_masks: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        outputs = self.forward(state, task_features, nest_masks)
        
        # Check for NaN or invalid probabilities
        joint_probs = outputs['joint_probs']
        if torch.isnan(joint_probs).any() or torch.isinf(joint_probs).any():
            # Fallback to uniform distribution
            print("Warning: NaN/Inf detected in joint_probs, using uniform distribution")
            batch_size, n_tasks = joint_probs.shape
            joint_probs = torch.ones_like(joint_probs) / n_tasks
            
        # Ensure probabilities are valid
        joint_probs = joint_probs.clamp(min=1e-10)
        joint_probs = joint_probs / joint_probs.sum(dim=1, keepdim=True)
        
        if deterministic:
            # Select task with highest probability
            task_indices = joint_probs.argmax(dim=1)
        else:
            # Sample from distribution with additional safety
            try:
                task_indices = torch.multinomial(joint_probs, num_samples=1).squeeze(1)
            except RuntimeError:
                # Fallback to argmax if sampling fails
                print("Warning: Sampling failed, using argmax")
                task_indices = joint_probs.argmax(dim=1)
            
        # Determine which nest was selected
        nest_indices = []
        for b in range(state.shape[0]):
            task_idx = task_indices[b]
            for m in range(self.n_nests):
                if nest_masks[b, m, task_idx]:
                    nest_indices.append(m)
                    break
                    
        nest_indices = torch.tensor(nest_indices, device=state.device)
        
        info = {
            'nest_indices': nest_indices,
            'nest_probs': outputs['nest_probs'],
            'task_probs': outputs['joint_probs'],
            'eta': outputs['eta']
        }
        
        return task_indices, info
        
    def compute_entropy(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Nest-level entropy
        nest_entropy = -torch.sum(outputs['nest_probs'] * torch.log(outputs['nest_probs'] + 1e-8), dim=1)
        
        # Task-level entropy within each nest
        task_entropies = []
        for m in range(self.n_nests):
            probs = outputs['task_probs_per_nest'][:, m, :]
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            task_entropies.append(entropy)
            
        task_entropies = torch.stack(task_entropies, dim=1)  # [B, M]
        
        # Weighted average of task entropies
        weighted_task_entropy = torch.sum(outputs['nest_probs'] * task_entropies, dim=1)
        
        return {
            'nest_entropy': nest_entropy,
            'task_entropy': weighted_task_entropy,
            'total_entropy': nest_entropy + weighted_task_entropy
        }

class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)