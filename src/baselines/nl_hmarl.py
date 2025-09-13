import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class NLManager(nn.Module):
    """Nested-Logit manager: chooses a task among a variable task set grouped by nests.

    - Computes per-task utilities from global state + per-task features
    - Groups tasks by `nest_ids` (e.g., zone 0..3)
    - Applies nested-logit choice: P(nest) ∝ exp(eta_m * log ∑ exp(v/eta_m)); P(task|nest) ∝ exp(v/eta_m)
    - Returns per-task probabilities across all tasks
    """

    def __init__(self,
                 state_dim: int,
                 n_tasks: int,
                 n_nests: int = 4,
                 hidden_dim: int = 256,
                 embed_dim: int = 128,
                 learn_eta: bool = False,
                 eta_init: float = 1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_nests = n_nests
        self.learn_eta = learn_eta

        # Task embedding from 5 features
        self.task_embedder = nn.Linear(5, embed_dim)

        # Utility MLP over [global_state, task_embed]
        self.task_utility_net = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Optional learnable eta per nest (positive)
        if learn_eta:
            self.raw_eta = nn.Parameter(torch.ones(n_nests) * float(np.log(np.exp(eta_init) - 1.0)))
        else:
            self.register_buffer('eta_buf', torch.ones(n_nests) * eta_init)

    def _eta(self) -> torch.Tensor:
        if self.learn_eta:
            return torch.nn.functional.softplus(self.raw_eta) + 1e-6
        return self.eta_buf

    def compute_task_utilities(self, state: torch.Tensor, task_features: torch.Tensor) -> torch.Tensor:
        # task_features: [B, T, 5]
        # state: [B, S]
        B, T, _ = task_features.shape
        task_embeds = self.task_embedder(task_features)  # [B, T, E]
        state_expanded = state.unsqueeze(1).expand(-1, T, -1)  # [B, T, S]
        x = torch.cat([state_expanded, task_embeds], dim=-1)
        utils = self.task_utility_net(x).squeeze(-1)  # [B, T]
        return utils

    def forward(self,
                state: torch.Tensor,
                task_features: torch.Tensor,
                nest_ids: torch.Tensor,
                task_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: [B, S]
            task_features: [B, T, 5]
            nest_ids: [B, T] ints in [0, n_nests) or -1 for padded/masked
            task_mask: [B, T] bool mask where True=valid
        Returns:
            dict with keys: 'utilities' [B,T], 'task_probs' [B,T], 'nest_probs' [B,M]
        """
        B, T, _ = task_features.shape
        M = self.n_nests
        utils = self.compute_task_utilities(state, task_features)  # [B, T]
        if task_mask is not None:
            utils = utils.masked_fill(~task_mask, -1e9)

        # Compute nested probabilities per batch via loops for clarity
        task_probs = []
        nest_probs = []
        etas = self._eta()  # [M]
        for b in range(B):
            u = utils[b]  # [T]
            nid = nest_ids[b]  # [T]
            mask = task_mask[b] if task_mask is not None else torch.ones_like(u, dtype=torch.bool)
            # Upper level: U_m = eta_m * log(sum_j exp(u_j / eta_m)) over tasks in nest m
            U_m = []
            for m in range(M):
                eta_m = etas[m]
                idx = (nid == m) & mask
                if idx.any():
                    vals = u[idx]
                    mval = torch.max(vals)
                    lse_in = torch.log(torch.clamp(torch.sum(torch.exp((vals - mval) / eta_m)), min=1e-12))
                    # eta*log(sum exp(v/eta)) computed stably: mval + eta*log(sum exp((v-mval)/eta))
                    U_m.append(mval + eta_m * lse_in)
                else:
                    U_m.append(torch.tensor(-1e9, device=u.device))
            U = torch.stack(U_m)  # [M]
            # Softmax over nests
            p_nest = torch.softmax(U, dim=0)  # [M]
            # Lower level: within each nest, softmax(u/eta_m)
            p_task = torch.zeros_like(u)
            for m in range(M):
                idx = (nid == m) & mask
                if idx.any():
                    eta_m = etas[m]
                    vals = u[idx]
                    mval = torch.max(vals)
                    exps = torch.exp((vals - mval) / eta_m)
                    denom = torch.clamp(exps.sum(), min=1e-12)
                    p_in = exps / denom
                    p_task[idx] = p_nest[m] * p_in
            # If all tasks masked, fall back to uniform over masked ones
            # Convert tensors to Python scalars safely (detach to avoid autograd warnings)
            if bool(mask.any().detach().item()) and (p_task.sum().detach().item() <= 1e-12):
                p_task[mask] = 1.0 / mask.float().sum()
            task_probs.append(p_task)
            nest_probs.append(p_nest)

        task_probs = torch.stack(task_probs, dim=0)
        nest_probs = torch.stack(nest_probs, dim=0)
        return {'task_utilities': utils, 'task_probs': task_probs, 'nest_probs': nest_probs}

    def select_task(self,
                    state: torch.Tensor,
                    task_features: torch.Tensor,
                    nest_ids: torch.Tensor,
                    task_mask: Optional[torch.Tensor] = None,
                    deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        outs = self.forward(state, task_features, nest_ids, task_mask)
        probs = outs['task_probs']
        if deterministic:
            idx = probs.argmax(dim=1)
        else:
            idx = torch.multinomial(torch.clamp(probs, min=1e-8), num_samples=1).squeeze(1)
        return idx, outs


class NLHMARL:
    """Container for NL manager + worker policy and value head for manager."""

    def __init__(self,
                 state_dim: int,
                 n_tasks: int,
                 n_nests: int,
                 worker_obs_dim: int,
                 worker_action_dim: int,
                 n_agents: int,
                 hidden_dim: int = 256,
                 device: str = 'cpu',
                 learn_eta: bool = False,
                 eta_init: float = 1.0):
        import torch
        from models.worker_policy import MultiAgentWorkerPolicy

        self.device = torch.device(device)
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.n_nests = n_nests

        self.manager = NLManager(
            state_dim=state_dim,
            n_tasks=n_tasks,
            n_nests=n_nests,
            hidden_dim=hidden_dim,
            embed_dim=128,
            learn_eta=learn_eta,
            eta_init=eta_init,
        ).to(self.device)

        # Shared workers
        self.workers = MultiAgentWorkerPolicy(
            obs_dim=worker_obs_dim,
            action_dim=worker_action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            share_params=True
        ).to(self.device)

        # Manager value
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

    def manager_forward(self, state, task_features, nest_ids, task_mask=None) -> Dict[str, torch.Tensor]:
        outs = self.manager(state, task_features, nest_ids, task_mask)
        value = self.value_net(state).squeeze(-1)
        outs['value'] = value
        return outs

    def select_tasks(self, state, task_features, nest_ids, task_mask=None, deterministic=False):
        idx, outs = self.manager.select_task(state, task_features, nest_ids, task_mask, deterministic)
        with torch.no_grad():
            outs['value'] = self.value_net(state).squeeze(-1)
        return idx, outs

    def worker_forward(self, obs):
        return self.workers(obs)

    def select_worker_actions(self, obs, deterministic=False):
        return self.workers.select_action(obs, deterministic=deterministic)

    def compute_manager_loss(self,
                             states: torch.Tensor,           # [N,S]
                             task_features: torch.Tensor,    # [N,T,5]
                             nest_ids: torch.Tensor,         # [N,T]
                             selected_tasks: torch.Tensor,   # [N]
                             advantages: torch.Tensor,       # [N]
                             returns: torch.Tensor,          # [N]
                             task_mask: Optional[torch.Tensor] = None,
                             entropy_coef: float = 0.01) -> Dict[str, torch.Tensor]:
        outs = self.manager_forward(states, task_features, nest_ids, task_mask)
        probs = torch.clamp(outs['task_probs'], min=1e-8)
        log_probs = torch.log(probs)
        sel_logp = log_probs.gather(1, selected_tasks.unsqueeze(1)).squeeze(1)
        policy_loss = -(advantages * sel_logp).mean()
        value_loss = F.mse_loss(outs['value'], returns)
        # Entropy of task distribution
        ent = -(probs * torch.log(probs)).sum(dim=1)
        entropy_loss = -entropy_coef * ent.mean()
        total = policy_loss + value_loss + entropy_loss
        return {
            'total_loss': total,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'entropy': ent.mean(),
        }

    def save(self, path: str):
        torch.save({
            'manager': self.manager.state_dict(),
            'workers': self.workers.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)

    def load(self, path: str, map_location=None):
        chk = torch.load(path, map_location=map_location or self.device)
        self.manager.load_state_dict(chk['manager'])
        self.workers.load_state_dict(chk['workers'])
        self.value_net.load_state_dict(chk['value_net'])
