from typing import Any, Dict, List, Tuple

import torch

from env.batch_tensor_warehouse_env import BatchTensorWarehouseEnv


class TensorVecEnv:
    """Single-process vectorized env on GPU/CPU tensors, API-compatible with SubprocVecEnv."""

    def __init__(self, env_config: Dict[str, Any], max_tasks: int, n_envs: int = 32, device: str = 'cuda'):
        dev = device
        if dev == 'cuda' and not torch.cuda.is_available():
            dev = 'cpu'
        self.env = BatchTensorWarehouseEnv(env_config, batch_size=int(max(1, n_envs)), max_tasks=max_tasks, device=dev)

    def get_features(self) -> List[Dict[str, Any]]:
        return self.env.get_features()

    def step_with_decisions(self, decisions: List[List[Tuple[int, int]]]) -> List[Dict[str, Any]]:
        return self.env.assign_and_step(decisions)

    def step_with_decisions_and_actions(self, decisions: List[List[Tuple[int, int]]],
                                        actions: List[List[int]]) -> List[Dict[str, Any]]:
        return self.env.assign_and_step_with_actions(decisions, actions)

    def get_worker_obs(self, include_global: bool = True) -> List[Dict[str, Any]]:
        """Build per-env worker observations matching 45-dim shape (approximate parity)."""
        outs: List[Dict[str, Any]] = []
        B, N, H, W, T = self.env.B, self.env.N, self.env.H, self.env.W, self.env.T
        grid = self.env.grid  # (B,H,W)
        for b in range(B):
            obs_list = []
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
                    # 与 CPU 版 get_agent_observation 的全局附加维度一致：
                    # [len(task_pool)/20, total_tasks_completed/100, current_time/1000,
                    #  busy_ratio, 0.0]
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
                # pad to 45
                while len(vec) < (45 if include_global else 40):
                    vec.append(0.0)
                obs_list.append(vec[:(45 if include_global else 40)])
            outs.append({'obs': torch.tensor(obs_list, dtype=torch.float32).cpu().numpy()})
        return outs

    def close(self):
        # Nothing to close in single-process tensor env
        pass
