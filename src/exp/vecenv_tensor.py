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
        """Return worker observations as a tensor of shape [B, N, 45]."""
        B, N, H, W = self.env.B, self.env.N, self.env.H, self.env.W
        grid = self.env.grid
        obs = torch.zeros((B, N, 45 if include_global else 40), dtype=torch.float32, device=self.env.device)
        for b in range(B):
            for i in range(N):
                px = int(self.env.picker_xy[b, i, 0].item())
                py = int(self.env.picker_xy[b, i, 1].item())
                has_task = int(self.env.current_task_idx[b, i].item() >= 0)
                carrying = 1 if bool(self.env.carrying[b, i].item()) else 0
                vec = [
                    px / max(1, W),
                    py / max(1, H),
                    carrying / 10.0,
                    float(has_task),
                    1.0,  # battery placeholder
                ]
                # local 5x5
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        x = px + dx; y = py + dy
                        if 0 <= x < W and 0 <= y < H:
                            val = float(grid[b, y, x].item()) / 4.0
                        else:
                            val = -1.0
                        vec.append(val)
                # target relative
                if has_task:
                    tid = int(self.env.current_task_idx[b, i].item())
                    if not self.env.carrying[b, i]:
                        tx = int(self.env.task_shelf[b, tid, 0].item()); ty = int(self.env.task_shelf[b, tid, 1].item())
                    else:
                        tx = int(self.env.task_station[b, tid, 0].item()); ty = int(self.env.task_station[b, tid, 1].item())
                    vec.extend([(tx - px) / max(1, W), (ty - py) / max(1, H), 1.0, 1.0])
                else:
                    vec.extend([0.0, 0.0, 0.0, 0.0])
                # nearest 3 others
                dlist = []
                for j in range(N):
                    if j == i: continue
                    ox = int(self.env.picker_xy[b, j, 0].item()); oy = int(self.env.picker_xy[b, j, 1].item())
                    d = abs(ox - px) + abs(oy - py)
                    dlist.append((d, ox - px, oy - py))
                dlist.sort(key=lambda t: t[0])
                for k in range(3):
                    if k < len(dlist):
                        _, dx, dy = dlist[k]
                        vec.extend([dx / max(1, W), dy / max(1, H)])
                    else:
                        vec.extend([0.0, 0.0])
                if include_global:
                    pending = int((self.env.task_status[b] == 0).sum().item())
                    total_completed = int(self.env.total_tasks_completed[b].item()) if hasattr(self.env, 'total_tasks_completed') else 0
                    busy = float((self.env.current_task_idx[b] >= 0).float().mean().item())
                    vec.extend([
                        pending / 20.0,
                        total_completed / 100.0,
                        float(self.env.current_time[b].item()) / 1000.0,
                        busy,
                        0.0,
                    ])
                # pad/crop to fixed dim
                dim = 45 if include_global else 40
                if len(vec) < dim:
                    vec.extend([0.0] * (dim - len(vec)))
                obs[b, i, :dim] = torch.tensor(vec[:dim], dtype=torch.float32, device=self.env.device)
        return obs

    def close(self):
        # Nothing to close in single-process tensor env
        pass
