import torch
from typing import Dict, Optional, Tuple

from .tensor_utils import distance_transform_4conn, argmin4_next_action, INF
from .tensor_order_generation import TensorOrderGeneratorTorch


class TensorWarehouseEnv:
    """Torch tensorized warehouse env (GPU-friendly prototype).

    Notes:
      - State, physics and rewards are computed via torch ops on `device`.
      - Grid: 0=free, 2=shelf, 3=station
      - Actions: 0:UP,1:DOWN,2:LEFT,3:RIGHT,4:IDLE (same as dynamic env)
      - This is a minimal viable prototype to demonstrate GPU acceleration; parity
        with the classic env is not guaranteed.
    """

    def __init__(self, config: Dict):
        self.width = int(config.get('width', 64))
        self.height = int(config.get('height', 64))
        self.n_pickers = int(config.get('n_pickers', 8))
        self.n_stations = int(config.get('n_stations', 4))
        self.levels_per_shelf = int(config.get('levels_per_shelf', 3))
        self.time_step = float(config.get('time_step', 2.0))
        self.episode_duration = float(config.get('episode_duration', 1.0))
        self.max_steps = int(self.episode_duration * 3600.0 / self.time_step)
        dev = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(dev)

        self.order_gen = TensorOrderGeneratorTorch(config.get('order_config', {}), device=self.device.type)

        # Buffers (set in reset)
        self.grid = None            # (H,W) int
        self.obstacle = None        # (H,W) bool
        self.stations_xy = None     # (S,2) long
        self.picker_xy = None       # (N,2) long
        self.carrying = None        # (N,) bool
        self.current_task_shelf = None  # (N,2) long, -1 if none
        self.current_task_st = None     # (N,2) long
        self.current_time = 0.0
        self.current_step = 0

    def _build_layout(self):
        H, W = self.height, self.width
        grid = torch.zeros((H, W), dtype=torch.int32, device=self.device)
        # Simple regular shelf lanes: every 4th row is shelf except borders
        for y in range(2, H - 2, 4):
            grid[y, 2:W - 2] = 2
        # Stations at bottom row evenly spaced
        S = max(1, self.n_stations)
        xs = torch.linspace(1, W - 2, S, device=self.device).round().long()
        ys = torch.full((S,), H - 2, dtype=torch.long, device=self.device)
        grid[ys, xs] = 3
        self.grid = grid
        self.obstacle = grid.eq(2)
        self.stations_xy = torch.stack([xs, ys], dim=1)

    def reset(self) -> torch.Tensor:
        self._build_layout()
        H, W = self.height, self.width
        # Spawn pickers randomly on free cells
        free_mask = (~self.obstacle) & self.grid.ne(3)
        fy, fx = free_mask.nonzero(as_tuple=True)
        idx = torch.randperm(fx.numel(), device=self.device)[: self.n_pickers]
        sel = torch.stack([fx[idx], fy[idx]], dim=1)
        self.picker_xy = sel.clone()
        self.carrying = torch.zeros((self.n_pickers,), dtype=torch.bool, device=self.device)
        self.current_task_shelf = torch.full((self.n_pickers, 2), -1, dtype=torch.long, device=self.device)
        self.current_task_st = torch.full((self.n_pickers, 2), -1, dtype=torch.long, device=self.device)
        self.current_time = 0.0
        self.current_step = 0
        return self.get_global_state()

    @torch.no_grad()
    def get_global_state(self) -> torch.Tensor:
        # Minimal global state: normalized time-of-day (sine/cosine) + load approx
        t = self.current_time % 24.0
        s = torch.tensor([
            torch.sin(torch.tensor(t * 3.14159265 / 12.0, device=self.device)),
            torch.cos(torch.tensor(t * 3.14159265 / 12.0, device=self.device)),
            self.picker_xy[:, 0].float().mean() / max(1.0, float(self.width)),
            self.picker_xy[:, 1].float().mean() / max(1.0, float(self.height)),
        ], device=self.device)
        return s

    @torch.no_grad()
    def _assign_simple(self):
        """Assign free pickers with nearest shelf task sampled this step (toy)."""
        # Sample orders in this step window (coarse)
        dt_h = self.time_step / 3600.0
        orders = self.order_gen.sample_window(self.current_time, dt_h, self.width, self.height, self.n_stations)
        k = orders['shelf_xy'].shape[0]
        if k == 0:
            return
        # For simplicity, assign the first min(k, n_free) shelves to free pickers
        free = (~self.carrying) & (self.current_task_shelf[:, 0] < 0)
        if free.any():
            idx_free = torch.nonzero(free, as_tuple=False).squeeze(1)
            take = min(idx_free.numel(), k)
            self.current_task_shelf[idx_free[:take]] = orders['shelf_xy'][:take]
            self.current_task_st[idx_free[:take]] = orders['station_xy'][:take]

    @torch.no_grad()
    def step(self, actions: Dict[int, int]) -> Tuple[torch.Tensor, Dict[int, float], Dict[int, bool], Dict]:
        # Optional on-the-fly assignment (toy)
        self._assign_simple()

        # Actions tensor
        a = torch.full((self.n_pickers,), 4, dtype=torch.long, device=self.device)
        for k, v in actions.items():
            if 0 <= k < self.n_pickers:
                a[k] = int(v)

        # Build per-picker target: shelf (when not carrying) else station
        tgt = torch.where(
            (~self.carrying).unsqueeze(1),
            self.current_task_shelf.clamp(min=0),
            self.current_task_st.clamp(min=0),
        )  # (N,2)
        # Compute multi-source distance on GPU and greedy move suggestion
        H, W = self.height, self.width
        # Sources: mark all target cells as sources
        sources = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        valid_tgt = (tgt[:, 0] >= 0) & (tgt[:, 1] >= 0)
        if valid_tgt.any():
            xs = tgt[valid_tgt, 0].clamp(0, W - 1)
            ys = tgt[valid_tgt, 1].clamp(0, H - 1)
            sources[ys, xs] = True
        dist = distance_transform_4conn(self.obstacle, sources, max_iters=H + W)
        greedy_actions = argmin4_next_action(dist, self.picker_xy, self.obstacle)
        # If external action is move and not invalid, prefer external; else use greedy
        dx = torch.tensor([0, 0, -1, 1, 0], device=self.device)
        dy = torch.tensor([-1, 1, 0, 0, 0], device=self.device)
        ext = a.clone()
        # Resolve invalid external moves (into shelves) by falling back to greedy
        nx = self.picker_xy[:, 0] + dx[ext]
        ny = self.picker_xy[:, 1] + dy[ext]
        inv = (nx < 0) | (ny < 0) | (nx >= W) | (ny >= H) | self.obstacle[ny.clamp(0, H - 1), nx.clamp(0, W - 1)]
        final_actions = torch.where(inv, greedy_actions, ext)

        # Apply moves (allow overlap on free cells; block shelves)
        nx = (self.picker_xy[:, 0] + dx[final_actions]).clamp(0, W - 1)
        ny = (self.picker_xy[:, 1] + dy[final_actions]).clamp(0, H - 1)
        move_ok = ~self.obstacle[ny, nx]
        self.picker_xy = torch.stack([
            torch.where(move_ok, nx, self.picker_xy[:, 0]),
            torch.where(move_ok, ny, self.picker_xy[:, 1])
        ], dim=1)

        # Pick/Drop on IDLE near shelf/station
        rewards: Dict[int, float] = {}
        info = {}
        manhattan = (self.picker_xy - tgt).abs().sum(dim=1)
        near = (manhattan == 1) & valid_tgt
        idle = (final_actions == 4)
        # Pick
        can_pick = near & (~self.carrying)
        pick_reward = can_pick.float() * 2.0
        self.carrying = torch.where(can_pick, torch.ones_like(self.carrying), self.carrying)
        # Drop
        can_drop = near & self.carrying
        drop_reward = can_drop.float() * 3.0
        # Clear task after drop
        done_drop = can_drop & idle
        self.current_task_shelf = torch.where(done_drop.unsqueeze(1), torch.full_like(self.current_task_shelf, -1), self.current_task_shelf)
        self.current_task_st = torch.where(done_drop.unsqueeze(1), torch.full_like(self.current_task_st, -1), self.current_task_st)
        self.carrying = torch.where(done_drop, torch.zeros_like(self.carrying), self.carrying)

        # Step reward: movement small bonus if reducing distance, idle penalty otherwise
        curd = dist[self.picker_xy[:, 1], self.picker_xy[:, 0]]
        step_rew = torch.zeros((self.n_pickers,), dtype=torch.float32, device=self.device)
        step_rew = step_rew + pick_reward + drop_reward
        step_rew = step_rew + torch.where(final_actions != 4, 0.05, -0.05)

        # Collect per-agent dicts for compatibility
        for i in range(self.n_pickers):
            rewards[i] = float(step_rew[i].item())

        # Time advance
        self.current_time += self.time_step / 3600.0
        self.current_step += 1
        done = self.current_step >= self.max_steps
        dones = {i: done for i in range(self.n_pickers)}
        return self.get_global_state(), rewards, dones, info

