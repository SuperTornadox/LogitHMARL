from typing import Any, Dict, List, Tuple

import torch

from env.batch_tensor_warehouse_env import BatchTensorWarehouseEnv


class TensorVecEnv:
    """Single-process vectorized env built on BatchTensorWarehouseEnv with tensor I/O.

    Changes vs. earlier version:
    - get_features returns a dict of batched torch tensors (no numpy/dicts per env)
    - get_worker_obs returns a [B,N,obs_dim] torch tensor
    """

    def __init__(self, env_config: Dict[str, Any], max_tasks: int, n_envs: int = 32, device: str = 'cuda'):
        dev = device
        if dev == 'cuda' and not torch.cuda.is_available():
            dev = 'cpu'
        self.env = BatchTensorWarehouseEnv(env_config, batch_size=int(max(1, n_envs)), max_tasks=max_tasks, device=dev)
        # Precompute 5x5 offset grids for local patches
        offs = torch.tensor([[-2, -1, 0, 1, 2]], device=self.env.device)
        self._dxs5 = offs.repeat(1, 5).view(1, 1, 5, 5)
        self._dys5 = offs.t().repeat(1, 5).view(1, 1, 5, 5)
        self._eyeN = torch.eye(self.env.N, device=self.env.device, dtype=torch.bool).view(1, self.env.N, self.env.N)

    @torch.no_grad()
    def get_features(self) -> Dict[str, torch.Tensor]:
        """Return batched tensors describing per-env global + task state.

        Keys:
          - state:     [B,S] float32  (global state vector)
          - task_feats:[B,T,5] float32 (per-task features)
          - task_mask: [B,T] bool     (True = valid/pending task)
          - nest_ids:  [B,T] long     (group id per task; here 0/1 by requires_car)
          - free_mask: [B,N] bool     (True = picker is free to assign)
        """
        e = self.env
        B, T, N, H, W = e.B, e.T, e.N, e.H, e.W
        device = e.device

        # Global state
        state = e.get_state_vec()  # [B,S], already on device

        # Task features (normalize shelf coords)
        sh = e.task_shelf.clamp(min=0)
        tf = torch.zeros((B, T, 5), dtype=torch.float32, device=device)
        tf[:, :, 0] = torch.where(e.task_shelf[:, :, 0] >= 0, sh[:, :, 0].float() / max(1, W), 0.0)
        tf[:, :, 1] = torch.where(e.task_shelf[:, :, 1] >= 0, sh[:, :, 1].float() / max(1, H), 0.0)
        tf[:, :, 2] = 1.0
        tf[:, :, 3] = e.task_priority
        rem = torch.clamp(e.task_deadline_abs - e.current_time.view(B, 1), min=0.0)
        tf[:, :, 4] = rem

        # Masks/ids
        task_mask = (e.task_status == 0)
        nest_ids = e.task_req_car.long().clamp(min=0)  # 0/1 bins by requires_car
        free_mask = (e.current_task_idx < 0) & (~e.carrying)

        return {
            'state': state,              # [B,S]
            'task_feats': tf,            # [B,T,5]
            'task_mask': task_mask,      # [B,T]
            'nest_ids': nest_ids,        # [B,T]
            'free_mask': free_mask,      # [B,N]
        }

    def step_with_decisions(self, decisions: List[List[Tuple[int, int]]]) -> List[Dict[str, Any]]:
        return self.env.assign_and_step(decisions)

    def step_with_decisions_and_actions(self, decisions: List[List[Tuple[int, int]]],
                                        actions: List[List[int]]) -> List[Dict[str, Any]]:
        return self.env.assign_and_step_with_actions(decisions, actions)

    @torch.no_grad()
    def step_with_decisions_tensor(self, decisions: List[List[Tuple[int, int]]]) -> Dict[str, torch.Tensor]:
        _ = self.env.assign_and_step(decisions)
        # Use cached tensors on device
        return {
            'step_reward': self.env.last_step_reward,   # [B]
            'next_state': self.env.last_next_state,     # [B,S]
            'rewards_vec': self.env.last_rewards,       # [B,N]
            'dones_vec': self.env.last_dones,           # [B,N]
        }

    @torch.no_grad()
    def step_with_decisions_and_actions_tensor(self, decisions: List[List[Tuple[int, int]]],
                                               actions: List[List[int]]) -> Dict[str, torch.Tensor]:
        _ = self.env.assign_and_step_with_actions(decisions, actions)
        return {
            'step_reward': self.env.last_step_reward,
            'next_state': self.env.last_next_state,
            'rewards_vec': self.env.last_rewards,
            'dones_vec': self.env.last_dones,
        }

    @torch.no_grad()
    def get_worker_obs(self, include_global: bool = True) -> torch.Tensor:
        """Vectorized worker observations [B,N,45 or 40]."""
        e = self.env
        B, N, H, W = e.B, e.N, e.H, e.W
        device = e.device

        # Basic features
        px = e.picker_xy[:, :, 0].float()  # [B,N]
        py = e.picker_xy[:, :, 1].float()
        carrying = e.carrying.float()  # [B,N]
        has_task = (e.current_task_idx >= 0).float()
        f0 = torch.stack([
            px / max(1, W),
            py / max(1, H),
            carrying / 10.0,
            has_task,
            torch.ones((B, N), device=device),  # battery placeholder
        ], dim=-1)  # [B,N,5]

        # Local 5x5 grid patch around each picker (padded with -1)
        gridf = e.grid.float() / 4.0  # [B,H,W]
        padded = torch.full((B, H + 4, W + 4), -1.0, device=device)
        padded[:, 2:H + 2, 2:W + 2] = gridf
        # Offsets for 5x5
        dxs = self._dxs5
        dys = self._dys5
        cx = (px.long() + 2).unsqueeze(-1).unsqueeze(-1)  # [B,N,1,1]
        cy = (py.long() + 2).unsqueeze(-1).unsqueeze(-1)
        xs = (cx + dxs).clamp(0, W + 3)
        ys = (cy + dys).clamp(0, H + 3)
        # Advanced indexing gather
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, 5, 5)
        local5x5 = padded[b_idx, ys, xs].reshape(B, N, 25)  # [B,N,25]

        # Target relative (to shelf if !carrying else to station)
        tid = e.current_task_idx.clamp(min=0)  # [B,N]
        shx = torch.gather(e.task_shelf[:, :, 0], 1, tid).float()
        shy = torch.gather(e.task_shelf[:, :, 1], 1, tid).float()
        stx = torch.gather(e.task_station[:, :, 0], 1, tid).float()
        sty = torch.gather(e.task_station[:, :, 1], 1, tid).float()
        tx = torch.where(e.carrying, stx, shx)
        ty = torch.where(e.carrying, sty, shy)
        tx = torch.where(e.current_task_idx >= 0, tx, px)
        ty = torch.where(e.current_task_idx >= 0, ty, py)
        tgt_dx = (tx - px) / max(1, W)
        tgt_dy = (ty - py) / max(1, H)
        f_tgt = torch.stack([tgt_dx, tgt_dy, torch.ones_like(tgt_dx), torch.ones_like(tgt_dy)], dim=-1)  # [B,N,4]
        f_tgt = torch.where((e.current_task_idx >= 0).unsqueeze(-1), f_tgt, torch.zeros_like(f_tgt))

        # Nearest 3 others: compute manhattan distances
        px_i = px.unsqueeze(2)  # [B,N,1]
        px_j = px.unsqueeze(1)  # [B,1,N]
        py_i = py.unsqueeze(2)
        py_j = py.unsqueeze(1)
        dx_all = (px_j - px_i)  # other - self
        dy_all = (py_j - py_i)
        dist = dx_all.abs() + dy_all.abs()  # [B,N,N]
        # mask self
        big = torch.full_like(dist, 1e9)
        dist = torch.where(self._eyeN, big, dist)
        k = min(3, max(0, N - 1))
        if k > 0:
            idx = torch.topk(dist, k=k, dim=2, largest=False).indices  # [B,N,k]
            dx_k = torch.gather(dx_all, 2, idx)
            dy_k = torch.gather(dy_all, 2, idx)
            # Normalize and pad to 3
            pad_k = 3 - k
            if pad_k > 0:
                dx_k = torch.cat([dx_k, torch.zeros((B, N, pad_k), device=device)], dim=2)
                dy_k = torch.cat([dy_k, torch.zeros((B, N, pad_k), device=device)], dim=2)
        else:
            dx_k = torch.zeros((B, N, 3), device=device)
            dy_k = torch.zeros((B, N, 3), device=device)
        f_nn = torch.stack([dx_k[..., 0] / max(1, W), dy_k[..., 0] / max(1, H),
                            dx_k[..., 1] / max(1, W), dy_k[..., 1] / max(1, H),
                            dx_k[..., 2] / max(1, W), dy_k[..., 2] / max(1, H)], dim=-1)  # [B,N,6]

        # Global extras
        dim = 45 if include_global else 40
        if include_global:
            pending = (e.task_status == 0).float().sum(dim=1) / 20.0  # [B]
            total_completed = e.total_tasks_completed.float() / 100.0 if hasattr(e, 'total_tasks_completed') else torch.zeros((B,), device=device)
            busy = (e.current_task_idx >= 0).float().mean(dim=1)  # [B]
            cur_t = e.current_time / 1000.0
            g = torch.stack([pending, total_completed, cur_t, busy, torch.zeros_like(busy)], dim=-1)  # [B,5]
            g = g.unsqueeze(1).expand(B, N, 5)
            obs = torch.cat([f0, local5x5, f_tgt, f_nn, g], dim=-1)
        else:
            obs = torch.cat([f0, local5x5, f_tgt, f_nn], dim=-1)
        return obs[:, :, :dim].contiguous()

    def close(self):
        # Nothing to close in single-process tensor env
        pass
