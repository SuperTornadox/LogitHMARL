import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple


class BatchTensorWarehouseEnv:
    """Batched tensorized warehouse env running entirely on a single device.

    This class is designed to mimic the minimal interface used by NL-HMARL
    subproc trainers via a vectorized wrapper: `get_features()` and
    `assign_and_step` semantics.

    Notes:
      - This is a first functional version; full parity with the CPU env
        (deadlines, value decay, congestion, energy) will be added in phases.
      - Shapes:
          B: batch_size (number of envs)
          N: n_pickers
          T: max_tasks
      - Grid codes: 0 free, 2 shelf, 3 station
    """

    def __init__(self, env_config: Dict[str, Any], *, batch_size: int = 32, max_tasks: int = 20,
                 device: str = 'cuda'):
        self.cfg = dict(env_config)
        self.B = int(batch_size)
        self.T = int(max_tasks)
        self.N = int(self.cfg.get('n_pickers', 8))
        self.W = int(self.cfg.get('width', 32))
        self.H = int(self.cfg.get('height', 32))
        self.S = int(self.cfg.get('n_stations', 4))
        self.time_step = float(self.cfg.get('time_step', 2.0))
        self.episode_duration = float(self.cfg.get('episode_duration', 1.0))
        self.max_steps = int(self.episode_duration * 3600.0 / self.time_step)

        self.device = torch.device(device)
        # Order config (scaled to simulation)
        ocfg = dict(self.cfg.get('order_config', {}))
        ocfg.setdefault('base_rate', 60.0)
        ocfg.setdefault('peak_hours', [(9, 12), (14, 17), (19, 21)])
        ocfg.setdefault('peak_multiplier', 1.6)
        ocfg.setdefault('off_peak_multiplier', 0.7)
        ocfg.setdefault('pattern_period_hours', 24)
        ocfg.setdefault('scale_pattern_to_simulation', True)
        ocfg['simulation_hours'] = self.episode_duration
        self.order_cfg = ocfg
        # Reward configuration (aligning gradually with classic env)
        self.rew_cfg = {
            'idle_penalty': -0.05,
            'move_bonus': 0.05,
            'move_toward_target': 0.1,
            'late_penalty': -5.0,
            'battery_low_penalty': -1.0,
        }
        # Weight thresholds + zone split
        wth = self.cfg.get('weight_thresholds', {'medium': 30.0, 'heavy': 70.0, 'forklift_only': 90.0})
        self.wt_medium = float(wth.get('medium', 30.0))
        self.wt_heavy = float(wth.get('heavy', 70.0))
        self.wt_forklift_only = float(wth.get('forklift_only', 90.0))
        self.col_aisle = int(self.cfg.get('col_aisle', self.W // 2))
        # Levels & forklift config
        self.levels_per_shelf = int(self.cfg.get('levels_per_shelf', 3))
        self.forklift_ratio = float(self.cfg.get('forklift_ratio', 0.2))
        self.min_forklifts = int(self.cfg.get('min_forklifts', 0))
        # Speed config（用于张量化速度/效率；若未提供则使用默认）
        scfg = dict(self.cfg.get('speed_config', {
            'base_speed': {'regular': 1.0, 'forklift': 1.2},
            'carry_alpha': {'regular': 1.0, 'forklift': 0.5},
            'congestion_mult': 0.7,
        }))
        self.base_speed_reg = float(scfg.get('base_speed', {}).get('regular', 1.0))
        self.base_speed_fork = float(scfg.get('base_speed', {}).get('forklift', 1.2))
        self.carry_alpha_reg = float(scfg.get('carry_alpha', {}).get('regular', 1.0))
        self.carry_alpha_fork = float(scfg.get('carry_alpha', {}).get('forklift', 0.5))
        # 拥堵减速：若提供 congestion_reduction 直接用；否则由 congestion_mult → reduction=1-congestion_mult
        if 'congestion_reduction' in scfg:
            self.congestion_red = float(scfg.get('congestion_reduction', 0.3))
        else:
            self.congestion_red = max(0.0, min(1.0, 1.0 - float(scfg.get('congestion_mult', 1.0))))

        # Buffers initialized in reset
        self.grid = None              # (B,H,W) int32
        self.obstacle = None          # (B,H,W) bool (shelves)
        self.stations = None          # (B,S,2) long
        self.picker_xy = None         # (B,N,2) long
        self.carrying = None          # (B,N) bool
        self.current_task_idx = None  # (B,N) long, -1 if none
        # Task pools (fixed cap T per env)
        self.task_shelf = None        # (B,T,2) long
        self.task_station = None      # (B,T,2) long
        self.task_req_car = None      # (B,T) bool
        self.task_status = None       # (B,T) long: 0=pending,1=assigned,2=done
        self.task_assigned = None     # (B,T) long: picker idx or -1

        self.current_time = None      # (B,) float
        self.current_step = None      # (B,) long
        self.battery = None           # (B,N) float
        self.total_value_completed = None  # (B,) float

        self.reset()
        # Cached tensors from last step for fast tensor I/O
        self.last_rewards = None        # (B,N) float
        self.last_dones = None          # (B,N) float
        self.last_step_reward = None    # (B,) float
        self.last_next_state = None     # (B,S) float

    def _build_layout(self):
        B, H, W, S = self.B, self.H, self.W, self.S
        grid = torch.zeros((B, H, W), dtype=torch.int32, device=self.device)
        # Regular shelf lanes: every 4th row except borders
        for y in range(2, H - 2, 4):
            grid[:, y, 2:W - 2] = 2
        # Stations at bottom
        xs = torch.linspace(1, W - 2, S, device=self.device).round().long()
        ys = torch.full((S,), H - 2, dtype=torch.long, device=self.device)
        # Broadcast stations per batch
        st = torch.stack([xs, ys], dim=1).unsqueeze(0).expand(B, -1, -1).contiguous()
        grid[torch.arange(B, device=self.device).unsqueeze(1), ys.unsqueeze(0).expand(B, -1), xs.unsqueeze(0).expand(B, -1)] = 3
        self.grid = grid
        self.obstacle = grid.eq(2)
        self.stations = st

    @torch.no_grad()
    def reset(self):
        self._build_layout()
        B, N, H, W, T = self.B, self.N, self.H, self.W, self.T
        # Spawn pickers on random free cells (avoid shelves)
        self.picker_xy = torch.zeros((B, N, 2), dtype=torch.long, device=self.device)
        for b in range(B):
            free = (~self.obstacle[b]) & self.grid[b].ne(3)
            fy, fx = free.nonzero(as_tuple=True)
            perm = torch.randperm(fx.numel(), device=self.device)[:N]
            sel = torch.stack([fx[perm], fy[perm]], dim=1)
            self.picker_xy[b] = sel
        self.carrying = torch.zeros((B, N), dtype=torch.bool, device=self.device)
        self.current_task_idx = torch.full((B, N), -1, dtype=torch.long, device=self.device)
        # Picker types (forklift vs regular)
        self.picker_is_forklift = torch.zeros((B, N), dtype=torch.bool, device=self.device)
        for b in range(B):
            k = max(int(round(self.forklift_ratio * N)), self.min_forklifts)
            k = min(max(0, k), N)
            if k > 0:
                idx = torch.randperm(N, device=self.device)[:k]
                self.picker_is_forklift[b, idx] = True
        # Initialize empty task buffers (status 3 = empty)
        self.task_shelf = torch.full((B, T, 2), -1, dtype=torch.long, device=self.device)
        self.task_station = torch.full((B, T, 2), -1, dtype=torch.long, device=self.device)
        self.task_req_car = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        self.task_status = torch.full((B, T), 3, dtype=torch.long, device=self.device)
        self.task_assigned = torch.full((B, T), -1, dtype=torch.long, device=self.device)
        self.task_value_base = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        self.task_deadline_abs = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        self.task_arrival_time = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        self.task_priority = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        self.task_zone = torch.full((B, T), -1, dtype=torch.long, device=self.device)
        self.task_weight = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        self.current_time = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.current_step = torch.zeros((B,), dtype=torch.long, device=self.device)
        self.battery = torch.full((B, N), 100.0, dtype=torch.float32, device=self.device)
        self.total_value_completed = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.total_orders_received = torch.zeros((B,), dtype=torch.long, device=self.device)
        self.total_orders_completed = torch.zeros((B,), dtype=torch.long, device=self.device)
        self.total_tasks_completed = torch.zeros((B,), dtype=torch.long, device=self.device)
        self.on_time_completions = torch.zeros((B,), dtype=torch.long, device=self.device)
        self.zone_loads = torch.zeros((B, 4), dtype=torch.float32, device=self.device)
        return self.get_state_vec()  # (B,S)

    @torch.no_grad()
    def get_state_vec(self) -> torch.Tensor:
        """构造与 CPU 版 get_global_state 等价的 81 维全局状态向量。
        维度结构：
        3（sin/cos/rate）+ 1（pending/100）+ 4（按重量 pending/50）+
        64（zone×weight: cnt/20, avg_pri, 0.5, avg_rem）+
        3（busy/bat_mean/forklift_busy）+ 4（zone load 占比）+
        2（完成率/按时率）
        """
        B, T = self.B, self.T
        # 时间 & 到达率
        t = self.current_time % 24.0
        s1 = torch.sin(t * 3.14159265 / 12.0)
        s2 = torch.cos(t * 3.14159265 / 12.0)
        rate = torch.tensor([self._arrival_rate(float(x)) for x in self.current_time.tolist()], device=self.device, dtype=torch.float32) / 100.0
        # Pending 总数
        pending_mask = (self.task_status == 0)
        pending_norm = pending_mask.float().sum(dim=1) / 100.0
        # 重量分类 masks（按阈值）
        w = self.task_weight
        fk = (w >= self.wt_forklift_only)
        hv = (w >= self.wt_heavy) & (w < self.wt_forklift_only)
        md = (w >= self.wt_medium) & (w < self.wt_heavy)
        lt = (w < self.wt_medium)
        # 按重量 pending 计数
        wc_fk = (pending_mask & fk).float().sum(dim=1) / 50.0
        wc_hv = (pending_mask & hv).float().sum(dim=1) / 50.0
        wc_md = (pending_mask & md).float().sum(dim=1) / 50.0
        wc_lt = (pending_mask & lt).float().sum(dim=1) / 50.0
        # zone × 重量统计
        z_summ = []
        for z in range(4):
            zmask = (self.task_zone == z)
            for cls_mask in [fk, hv, md, lt]:
                mask = pending_mask & zmask & cls_mask
                cnt = mask.float().sum(dim=1) / 20.0
                # 平均 priority / remaining_time
                pri = torch.where(mask, self.task_priority, torch.zeros_like(self.task_priority))
                rem = torch.clamp(self.task_deadline_abs - self.current_time.view(B, 1), min=0.0)
                rem = torch.where(mask, rem, torch.zeros_like(rem))
                denom = mask.float().sum(dim=1).clamp(min=1e-6)
                avg_pri = pri.sum(dim=1) / denom
                avg_rem = rem.sum(dim=1) / denom
                z_summ.extend([cnt, avg_pri, torch.full_like(avg_pri, 0.5), avg_rem])
        # 拣货员统计
        busy_ratio = (self.current_task_idx >= 0).float().mean(dim=1)
        bat_mean = self.battery.mean(dim=1) / 100.0
        forklifts_total = self.picker_is_forklift.float().sum(dim=1).clamp(min=1.0)
        forklifts_busy = (self.picker_is_forklift & (self.current_task_idx >= 0)).float().sum(dim=1) / forklifts_total
        # Zone 负载占比
        zl_total = self.zone_loads.sum(dim=1).clamp(min=1e-6)
        zl = [self.zone_loads[:, i] / zl_total for i in range(4)]
        # 绩效
        perf1 = self.total_orders_completed.float() / self.total_orders_received.float().clamp(min=1.0)
        perf2 = self.on_time_completions.float() / self.total_tasks_completed.float().clamp(min=1.0)
        # 拼接
        parts = [s1, s2, rate, pending_norm, wc_fk, wc_hv, wc_md, wc_lt] + z_summ + [busy_ratio, bat_mean, forklifts_busy] + zl + [perf1, perf2]
        return torch.stack(parts, dim=1)

    def _arrival_rate(self, time_hour: float) -> float:
        period = float(max(1e-6, self.order_cfg.get('pattern_period_hours', 24)))
        sim_h = float(max(1e-6, self.order_cfg.get('simulation_hours', 24)))
        scale = bool(self.order_cfg.get('scale_pattern_to_simulation', False))
        if scale:
            t_sim = time_hour % sim_h
            hour = (t_sim / sim_h) * period
        else:
            hour = time_hour % period
        for s, e in self.order_cfg.get('peak_hours', []):
            if s <= hour < e:
                return float(self.order_cfg.get('base_rate', 60.0)) * float(self.order_cfg.get('peak_multiplier', 1.6))
        if hour >= 22 or hour < 6:
            return float(self.order_cfg.get('base_rate', 60.0)) * float(self.order_cfg.get('off_peak_multiplier', 0.7)) * 0.5
        return float(self.order_cfg.get('base_rate', 60.0))

    def _cls_name(self, b: int, tid: int) -> str:
        """Map task weight to class name for reward scaling."""
        w = float(self.task_weight[b, tid].item()) if hasattr(self, 'task_weight') else 0.0
        if w >= self.wt_forklift_only:
            return 'forklift_only'
        if w >= self.wt_heavy:
            return 'heavy'
        if w >= self.wt_medium:
            return 'medium'
        return 'light'

    @torch.no_grad()
    def _spawn_new_tasks(self):
        dt_h = self.time_step / 3600.0
        B, T, W, H = self.B, self.T, self.W, self.H
        for b in range(B):
            lam = self._arrival_rate(float(self.current_time[b].item())) * dt_h
            k = torch.poisson(torch.tensor([lam], device=self.device)).long().item()
            if k <= 0:
                continue
            empties = torch.nonzero(self.task_status[b] >= 2, as_tuple=False).squeeze(1)
            if empties.numel() <= 0:
                continue
            take = min(int(empties.numel()), k)
            idx = empties[:take]
            sx = torch.randint(0, W, (take,), device=self.device)
            sy = torch.randint(0, H, (take,), device=self.device)
            stx = torch.randint(0, W, (take,), device=self.device)
            sty = torch.randint(0, H, (take,), device=self.device)
            self.task_shelf[b, idx, 0] = sx
            self.task_shelf[b, idx, 1] = sy
            self.task_station[b, idx, 0] = stx
            self.task_station[b, idx, 1] = sty
            # values/deadlines/priority
            base_val = torch.randint(50, 101, (take,), device=self.device).float()
            self.task_value_base[b, idx] = base_val
            dl = (torch.rand((take,), device=self.device) * 0.4 + 0.1)
            self.task_deadline_abs[b, idx] = self.current_time[b] + dl
            self.task_arrival_time[b, idx] = self.current_time[b]
            self.task_priority[b, idx] = (torch.rand((take,), device=self.device) * 0.6 + 0.4)
            # weight & requires_car
            w = torch.rand((take,), device=self.device) * 100.0
            self.task_weight[b, idx] = w
            self.task_req_car[b, idx] = (w >= self.wt_forklift_only)
            # zone
            midy = self.H // 2
            zx = (sx >= self.col_aisle).long()
            zy = (sy >= midy).long()
            self.task_zone[b, idx] = zy * 2 + zx
            self.task_status[b, idx] = 0
            # stats
            self.total_orders_received[b] += take

    @torch.no_grad()
    def get_features(self) -> List[Dict[str, Any]]:
        """Return per-env feature dicts matching SubprocVecEnv output keys."""
        outs: List[Dict[str, Any]] = []
        B, T = self.B, self.T
        state_vec = self.get_state_vec()  # (B,S)
        # Build per-env features
        for b in range(B):
            # task_feats: (T,5): shelf x,y normalized + dummy size + priority + remaining time
            sh = self.task_shelf[b].clamp(min=0)
            tf = torch.zeros((T, 5), dtype=torch.float32, device=self.device)
            tf[:, 0] = torch.where(sh[:, 0] >= 0, sh[:, 0].float() / max(1, self.W), torch.zeros_like(sh[:, 0]).float())
            tf[:, 1] = torch.where(sh[:, 1] >= 0, sh[:, 1].float() / max(1, self.H), torch.zeros_like(sh[:, 1]).float())
            tf[:, 2] = 1.0
            tf[:, 3] = self.task_priority[b]
            rem = torch.clamp(self.task_deadline_abs[b] - self.current_time[b], min=0.0)
            tf[:, 4] = rem
            # task ids just 0..T-1
            tids = torch.arange(T, device=self.device, dtype=torch.long)
            requires = self.task_req_car[b]
            free_pids = torch.nonzero((self.current_task_idx[b] < 0) & (~self.carrying[b]), as_tuple=False).squeeze(1)
            outs.append({
                'state_vec': state_vec[b].detach().cpu().numpy().astype('float32'),
                'task_feats': tf.detach().cpu().numpy().astype('float32'),
                'task_ids': tids.detach().cpu().numpy().astype('int64'),
                'requires': requires.detach().cpu().numpy().astype('bool'),
                'free_pids': free_pids.detach().cpu().numpy().astype('int64'),
            })
        return outs

    @torch.no_grad()
    def assign_and_step(self, per_env_decisions: List[List[Tuple[int, int]]]) -> List[Dict[str, Any]]:
        """Apply decisions and step one time unit for all envs.

        decisions[b] is a list of (pid, task_id) for env b.
        """
        B, N, T, H, W = self.B, self.N, self.T, self.H, self.W
        # Spawn new tasks for this step
        self._spawn_new_tasks()
        # Apply assignments
        for b in range(B):
            if b >= len(per_env_decisions):
                continue
            for (pid, tid) in per_env_decisions[b]:
                if not (0 <= pid < N and 0 <= tid < T):
                    continue
                if self.task_status[b, tid].item() != 0:  # not pending
                    continue
                if self.current_task_idx[b, pid].item() >= 0:  # already busy
                    continue
                self.task_status[b, tid] = 1  # assigned
                self.task_assigned[b, tid] = int(pid)
                self.current_task_idx[b, pid] = int(tid)

        # Heuristic actions: move toward shelf when not carrying; else toward station
        actions = torch.full((B, N), 4, dtype=torch.long, device=self.device)
        dir_xy = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=torch.long, device=self.device)

        # One greedy step per picker
        for b in range(B):
            for i in range(N):
                tid = self.current_task_idx[b, i].item()
                if tid < 0:
                    continue
                tgt = self.task_shelf[b, tid] if not self.carrying[b, i] else self.task_station[b, tid]
                # Choose direction minimizing L1 distance (avoid shelves)
                px, py = self.picker_xy[b, i]
                best_a = 4
                best_d = 10 ** 9
                for a in range(4):
                    nx = px + dir_xy[a, 0]
                    ny = py + dir_xy[a, 1]
                    if nx < 0 or ny < 0 or nx >= W or ny >= H:
                        continue
                    if self.obstacle[b, ny, nx]:
                        continue
                    d = (tgt[0] - nx).abs().item() + (tgt[1] - ny).abs().item()
                    if d < best_d:
                        best_d = d
                        best_a = a
                # If adjacent, idle to trigger pick/drop
                if (px - tgt[0]).abs().item() + (py - tgt[1]).abs().item() == 1:
                    best_a = 4
                actions[b, i] = best_a

        # 计算每个拣货员的“本步可移动格数”（速度）
        base_speed = torch.where(self.picker_is_forklift,
                                 torch.full((B, N), self.base_speed_fork, device=self.device),
                                 torch.full((B, N), self.base_speed_reg, device=self.device))
        # 任务权重 gather
        tid_mat = self.current_task_idx.clone()
        tid_safe = torch.clamp(tid_mat, min=0)
        task_w = self.task_weight  # (B,T)
        wt = torch.gather(task_w, 1, tid_safe)
        wt = torch.where(tid_mat >= 0, wt, torch.zeros_like(wt))
        thr_base = torch.full_like(wt, self.wt_forklift_only)
        weight_term = torch.clamp((thr_base - wt + 10.0) / torch.clamp(thr_base + 10.0, min=1e-6), 0.0, 2.0)
        congest = torch.full((B, N), self.congestion_red, device=self.device)
        eff = weight_term * (1.0 - congest)
        carry_alpha = torch.where(self.picker_is_forklift,
                                  torch.full((B, N), self.carry_alpha_fork, device=self.device),
                                  torch.full((B, N), self.carry_alpha_reg, device=self.device))
        eff = torch.where(self.carrying, eff * carry_alpha, eff)
        speed = base_speed * eff
        steps_int = torch.clamp(torch.floor(speed), min=0).to(torch.long)
        frac = torch.clamp(speed - steps_int.float(), min=0.0)
        extra = (torch.rand_like(frac) < frac).to(torch.long)
        steps_to_move = steps_int + extra

        # 目标坐标（按当前是否携带选择 shelf/station）
        shelf_x = torch.gather(self.task_shelf[:, :, 0], 1, tid_safe)
        shelf_y = torch.gather(self.task_shelf[:, :, 1], 1, tid_safe)
        station_x = torch.gather(self.task_station[:, :, 0], 1, tid_safe)
        station_y = torch.gather(self.task_station[:, :, 1], 1, tid_safe)
        tgt_x = torch.where(self.carrying, station_x, shelf_x)
        tgt_y = torch.where(self.carrying, station_y, shelf_y)
        tgt_x = torch.where(tid_mat >= 0, tgt_x, self.picker_xy[:, :, 0].long())
        tgt_y = torch.where(tid_mat >= 0, tgt_y, self.picker_xy[:, :, 1].long())

        # 向量化多步移动
        rewards = torch.zeros((B, N), dtype=torch.float32, device=self.device)
        cur_x = self.picker_xy[:, :, 0].long().clone()
        cur_y = self.picker_xy[:, :, 1].long().clone()
        # 动作对应的位移
        dxv = torch.tensor([0, 0, -1, 1, 0], device=self.device)
        dyv = torch.tensor([-1, 1, 0, 0, 0], device=self.device)
        dx_act = dxv[actions]
        dy_act = dyv[actions]
        max_steps = int(steps_to_move.max().item()) if B * N > 0 else 0
        b_idx = torch.arange(B, device=self.device).view(B, 1).expand(B, N)
        for s in range(max_steps):
            alive = (actions < 4) & (steps_to_move > s)
            if not bool(alive.any().detach().item()):
                break
            nx = cur_x + dx_act
            ny = cur_y + dy_act
            inb = (nx >= 0) & (ny >= 0) & (nx < W) & (ny < H)
            obs = self.obstacle[b_idx, ny.clamp(0, H - 1), nx.clamp(0, W - 1)]
            ok = alive & inb & (~obs)
            # 朝目标更近奖励
            old_dist = (cur_x - tgt_x).abs() + (cur_y - tgt_y).abs()
            new_dist = (nx.clamp(0, W - 1) - tgt_x).abs() + (ny.clamp(0, H - 1) - tgt_y).abs()
            improve = ok & (new_dist < old_dist)
            rewards = rewards + improve.float() * float(self.rew_cfg.get('move_toward_target', self.rew_cfg.get('move_bonus', 0.05)))
            cur_x = torch.where(ok, nx.clamp(0, W - 1), cur_x)
            cur_y = torch.where(ok, ny.clamp(0, H - 1), cur_y)
        self.picker_xy[:, :, 0] = cur_x
        self.picker_xy[:, :, 1] = cur_y

        # 向量化 Pick/Drop（IDLE 且相邻触发）
        idle_mask = (actions == 4)
        has_task = (tid_mat := self.current_task_idx).ge(0)
        px = self.picker_xy[:, :, 0].long(); py = self.picker_xy[:, :, 1].long()
        # 相邻 shelf/station
        shx = torch.gather(self.task_shelf[:, :, 0], 1, tid_safe)
        shy = torch.gather(self.task_shelf[:, :, 1], 1, tid_safe)
        stx = torch.gather(self.task_station[:, :, 0], 1, tid_safe)
        sty = torch.gather(self.task_station[:, :, 1], 1, tid_safe)
        adj_shelf = (px - shx).abs() + (py - shy).abs() == 1
        adj_station = (px - stx).abs() + (py - sty).abs() == 1
        # Pick：idle & ~carrying & has_task & adj_shelf
        can_pick = idle_mask & (~self.carrying) & has_task & adj_shelf
        req = torch.gather(self.task_req_car, 1, tid_safe)
        req = torch.where(has_task, req, torch.zeros_like(req))
        allowed_pick = can_pick & (~req | self.picker_is_forklift)
        disallowed_pick = can_pick & req & (~self.picker_is_forklift)
        # pick reward 按类别与类型缩放
        w = torch.gather(self.task_weight, 1, tid_safe)
        cls_fk = (w >= self.wt_forklift_only)
        cls_hv = (w >= self.wt_heavy) & (w < self.wt_forklift_only)
        cls_md = (w >= self.wt_medium) & (w < self.wt_heavy)
        cls_lt = (w < self.wt_medium)
        cfg_rw = self.cfg.get('reward_config', {})
        pick_base = cfg_rw.get('pick_base', {'forklift_only': 4.0, 'heavy': 3.0, 'medium': 2.0, 'light': 1.0})
        pb = (torch.full_like(w, float(pick_base.get('forklift_only', 4.0))) * cls_fk.float() +
              torch.full_like(w, float(pick_base.get('heavy', 3.0))) * cls_hv.float() +
              torch.full_like(w, float(pick_base.get('medium', 2.0))) * cls_md.float() +
              torch.full_like(w, float(pick_base.get('light', 1.0))) * cls_lt.float())
        fork_eff = cfg_rw.get('forklift_eff', {'forklift_only': 2.0, 'heavy': 1.8, 'medium': 1.2, 'light': 1.1})
        reg_eff = cfg_rw.get('regular_eff', {'forklift_only': 0.0, 'heavy': 1.0, 'medium': 1.0, 'light': 1.0})
        fe = (torch.full_like(w, float(fork_eff.get('forklift_only', 2.0))) * cls_fk.float() +
              torch.full_like(w, float(fork_eff.get('heavy', 1.8))) * cls_hv.float() +
              torch.full_like(w, float(fork_eff.get('medium', 1.2))) * cls_md.float() +
              torch.full_like(w, float(fork_eff.get('light', 1.1))) * cls_lt.float())
        re = (torch.full_like(w, float(reg_eff.get('forklift_only', 0.0))) * cls_fk.float() +
              torch.full_like(w, float(reg_eff.get('heavy', 1.0))) * cls_hv.float() +
              torch.full_like(w, float(reg_eff.get('medium', 1.0))) * cls_md.float() +
              torch.full_like(w, float(reg_eff.get('light', 1.0))) * cls_lt.float())
        eff_sel = torch.where(self.picker_is_forklift, fe, re)
        pick_reward_mat = pb * eff_sel
        rewards = rewards + torch.where(allowed_pick, pick_reward_mat.float(), torch.zeros_like(pick_reward_mat))
        rewards = rewards + torch.where(disallowed_pick, torch.full_like(pick_reward_mat, -0.5), torch.zeros_like(pick_reward_mat))
        self.carrying = torch.where(allowed_pick, torch.ones_like(self.carrying), self.carrying)
        # Drop：idle & carrying & has_task & adj_station
        can_drop = idle_mask & self.carrying & has_task & adj_station
        base_val = torch.gather(self.task_value_base, 1, tid_safe)
        D = torch.gather(self.task_deadline_abs, 1, tid_safe)
        cur_t = self.current_time.view(B, 1).expand(B, N)
        val = torch.where(cur_t <= D, base_val,
                          torch.where(cur_t < 2 * D, base_val * (2 * D - cur_t) / torch.clamp(D, min=1e-6), torch.zeros_like(base_val)))
        drop_base = cfg_rw.get('drop_base', {'forklift_only': 5.0, 'heavy': 4.0, 'medium': 2.5, 'light': 1.5})
        db = (torch.full_like(w, float(drop_base.get('forklift_only', 5.0))) * cls_fk.float() +
              torch.full_like(w, float(drop_base.get('heavy', 4.0))) * cls_hv.float() +
              torch.full_like(w, float(drop_base.get('medium', 2.5))) * cls_md.float() +
              torch.full_like(w, float(drop_base.get('light', 1.5))) * cls_lt.float())
        drop_reward_mat = db * eff_sel + val
        rewards = rewards + torch.where(can_drop, drop_reward_mat.float(), torch.zeros_like(drop_reward_mat))
        late_pen = torch.where(cur_t > D, torch.full_like(val, float(self.rew_cfg.get('late_penalty', -5.0))), torch.zeros_like(val))
        rewards = rewards + torch.where(can_drop, late_pen, torch.zeros_like(late_pen))
        # 完成任务状态更新
        done_mask = can_drop
        # 更新 task_status 到 2
        # 将 (B,N) 标量 drop 对应的 (B,T) 索引位置设置为 2
        if bool(done_mask.any().detach().item()):
            # 为每个 (b,i) 取 tid_safe[b,i]
            b_lin = torch.arange(B, device=self.device).view(B, 1).expand(B, N)
            tid_lin = tid_safe
            # 仅对 done 的位置赋值
            self.task_status[b_lin[done_mask], tid_lin[done_mask]] = 2
            self.task_assigned[b_lin[done_mask], tid_lin[done_mask]] = -1
            # 当前拣货员清空任务 & 取消携带
            self.current_task_idx[done_mask] = -1
            self.carrying[done_mask] = False
            # 统计
            inc = done_mask.float().sum(dim=1).long()
            self.total_tasks_completed += inc
            self.total_orders_completed += inc
            self.on_time_completions += ((cur_t <= D) & done_mask).float().sum(dim=1).long()
        # Idle 惩罚（统一累加）
        rewards = rewards + torch.where(actions == 4, torch.full_like(rewards, self.rew_cfg['idle_penalty']), torch.zeros_like(rewards))

        # Battery and congestion penalties
        # Battery drains per move/idle
        moved_mask = (actions < 4) | (steps_to_move > 0)
        self.battery = self.battery - torch.where(moved_mask, torch.full_like(self.battery, 0.1), torch.full_like(self.battery, 0.05))
        low = self.battery < 20.0
        rewards = rewards + (low.float() * self.rew_cfg['battery_low_penalty'])
        # Congestion penalty via convolution (8-neighborhood)
        occ_flat = torch.zeros((B, H * W), dtype=torch.float32, device=self.device)
        idx_flat = (self.picker_xy[:, :, 1].long() * W + self.picker_xy[:, :, 0].long()).clamp(0, H * W - 1)
        occ_flat.scatter_add_(1, idx_flat, torch.ones_like(idx_flat, dtype=torch.float32))
        occ = occ_flat.view(B, 1, H, W)
        nb = F.conv2d(occ, torch.ones((1, 1, 3, 3), device=self.device), padding=1) - occ
        b_idx = torch.arange(B, device=self.device).view(B, 1).expand(B, N)
        neigh = nb.squeeze(1)[b_idx, self.picker_xy[:, :, 1].long(), self.picker_xy[:, :, 0].long()]
        rewards = rewards + (neigh.ge(1.0).float() * -0.5)
        # 区域均衡（全局奖励，按人头均摊）
        total = self.zone_loads.sum(dim=1)
        # 使用标准差作为不均衡度量，参考 CPU 版条件
        std = torch.std(self.zone_loads, dim=1)
        zb_bonus = float(self.cfg.get('reward_config', {}).get('zone_balance_bonus', 0.3))
        mask_bal = (total > 0) & (std < torch.maximum(torch.full_like(total, 1.0), total * 0.1))
        if mask_bal.any():
            rewards[mask_bal] += (zb_bonus / max(1, N))

        # Advance time
        self.current_time += self.time_step / 3600.0
        self.current_step += 1

        # Cache tensor outputs
        st_vec = self.get_state_vec()  # (B,S)
        done_b = (self.current_step >= self.max_steps)
        self.last_rewards = rewards.detach().clone()
        self.last_dones = done_b.view(B, 1).float().expand(B, N).detach().clone()
        self.last_step_reward = rewards.sum(dim=1).detach().clone()
        self.last_next_state = st_vec.detach().clone()

        # Backward-compatible numpy list-of-dict for existing callers
        outs: List[Dict[str, Any]] = []
        for b in range(B):
            outs.append({
                'step_reward': float(self.last_step_reward[b].item()),
                'next_state_vec': self.last_next_state[b].cpu().numpy().astype('float32'),
                'rewards_vec': self.last_rewards[b].cpu().numpy().astype('float32'),
                'dones_vec': self.last_dones[b].cpu().numpy().astype('float32'),
            })
        return outs

    @torch.no_grad()
    def assign_and_step_with_actions(self, per_env_decisions: List[List[Tuple[int, int]]],
                                     actions_per_env: List[List[int]]) -> List[Dict[str, Any]]:
        """Apply decisions and low-level actions for workers, then step.

        actions_per_env[b] is a list of length N with action ints per picker.
        """
        B, N, T, H, W = self.B, self.N, self.T, self.H, self.W
        # Spawn new tasks then apply assignments
        self._spawn_new_tasks()
        # Apply assignments (same as above)
        for b in range(B):
            if b < len(per_env_decisions):
                for (pid, tid) in per_env_decisions[b]:
                    if 0 <= pid < N and 0 <= tid < T and self.task_status[b, tid].item() == 0 and self.current_task_idx[b, pid].item() < 0:
                        self.task_status[b, tid] = 1
                        self.task_assigned[b, tid] = int(pid)
                        self.current_task_idx[b, pid] = int(tid)

        dir_xy = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=torch.long, device=self.device)
        rewards = torch.zeros((B, N), dtype=torch.float32, device=self.device)
        # Apply provided actions per env
        for b in range(B):
            acts = actions_per_env[b] if b < len(actions_per_env) else [4] * N
            for i in range(N):
                a = int(acts[i]) if i < len(acts) else 4
                px, py = self.picker_xy[b, i]
                if a < 4:
                    nx = px + dir_xy[a, 0]
                    ny = py + dir_xy[a, 1]
                    if 0 <= nx < W and 0 <= ny < H and (not self.obstacle[b, ny, nx]):
                        self.picker_xy[b, i, 0] = nx
                        self.picker_xy[b, i, 1] = ny
                        rewards[b, i] += self.rew_cfg['move_bonus']
                else:
                    tid = self.current_task_idx[b, i].item()
                    if tid >= 0:
                        sh = self.task_shelf[b, tid]
                        st = self.task_station[b, tid]
                        # Pick if adjacent to shelf and not carrying
                        if (not self.carrying[b, i]) and (abs(int(self.picker_xy[b, i, 0]) - int(sh[0])) + abs(int(self.picker_xy[b, i, 1]) - int(sh[1])) == 1):
                            req = bool(self.task_req_car[b, tid].item())
                            if req and (not bool(self.picker_is_forklift[b, i].item())):
                                rewards[b, i] += -0.5
                            else:
                                self.carrying[b, i] = True
                                cls_name = self._cls_name(b, tid)
                                pick_base = self.cfg.get('reward_config', {}).get('pick_base', {'forklift_only': 4.0, 'heavy': 3.0, 'medium': 2.0, 'light': 1.0})
                                fork_eff = self.cfg.get('reward_config', {}).get('forklift_eff', {'forklift_only': 2.0, 'heavy': 1.8, 'medium': 1.2, 'light': 1.1})
                                reg_eff = self.cfg.get('reward_config', {}).get('regular_eff', {'forklift_only': 0.0, 'heavy': 1.0, 'medium': 1.0, 'light': 1.0})
                                eff_type = fork_eff if bool(self.picker_is_forklift[b, i].item()) else reg_eff
                                rewards[b, i] += float(pick_base.get(cls_name, 1.0) * eff_type.get(cls_name, 1.0))
                        # Drop if adjacent to station and carrying
                        if self.carrying[b, i] and (abs(int(self.picker_xy[b, i, 0]) - int(st[0])) + abs(int(self.picker_xy[b, i, 1]) - int(st[1])) == 1):
                            self.carrying[b, i] = False
                            cur_t = float(self.current_time[b].item())
                            base = float(self.task_value_base[b, tid].item())
                            D = float(self.task_deadline_abs[b, tid].item())
                            if cur_t <= D:
                                val = base
                            elif cur_t < 2 * D:
                                val = base * (2 * D - cur_t) / max(1e-6, D)
                            else:
                                val = 0.0
                            drop_base = self.cfg.get('reward_config', {}).get('drop_base', {'forklift_only': 5.0, 'heavy': 4.0, 'medium': 2.5, 'light': 1.5})
                            fork_eff = self.cfg.get('reward_config', {}).get('forklift_eff', {'forklift_only': 2.0, 'heavy': 1.8, 'medium': 1.2, 'light': 1.1})
                            reg_eff = self.cfg.get('reward_config', {}).get('regular_eff', {'forklift_only': 0.0, 'heavy': 1.0, 'medium': 1.0, 'light': 1.0})
                            cls_name = self._cls_name(b, tid)
                            eff_type = fork_eff if bool(self.picker_is_forklift[b, i].item()) else reg_eff
                            rewards[b, i] += float(drop_base.get(cls_name, 1.0) * eff_type.get(cls_name, 1.0)) + float(val)
                            if cur_t > D:
                                rewards[b, i] += self.rew_cfg['late_penalty']
                            self.task_status[b, tid] = 2
                            self.task_assigned[b, tid] = -1
                            self.current_task_idx[b, i] = -1
                if a == 4:
                    rewards[b, i] += self.rew_cfg['idle_penalty']
        # Battery and congestion
        moved_mask = torch.zeros((B, N), dtype=torch.bool, device=self.device)
        for b in range(B):
            for i in range(N):
                a = int(actions_per_env[b][i]) if (b < len(actions_per_env) and i < len(actions_per_env[b])) else 4
                moved_mask[b, i] = (a < 4)
        self.battery = self.battery - torch.where(moved_mask, torch.full_like(self.battery, 0.1), torch.full_like(self.battery, 0.05))
        low = self.battery < 20.0
        rewards = rewards + (low.float() * self.rew_cfg['battery_low_penalty'])
        # Congestion via convolution
        occ_flat = torch.zeros((B, H * W), dtype=torch.float32, device=self.device)
        idx_flat = (self.picker_xy[:, :, 1].long() * W + self.picker_xy[:, :, 0].long()).clamp(0, H * W - 1)
        occ_flat.scatter_add_(1, idx_flat, torch.ones_like(idx_flat, dtype=torch.float32))
        occ = occ_flat.view(B, 1, H, W)
        nb = F.conv2d(occ, torch.ones((1, 1, 3, 3), device=self.device), padding=1) - occ
        b_idx = torch.arange(B, device=self.device).view(B, 1).expand(B, N)
        neigh = nb.squeeze(1)[b_idx, self.picker_xy[:, :, 1].long(), self.picker_xy[:, :, 0].long()]
        rewards = rewards + (neigh.ge(1.0).float() * -0.5)
        # 区域均衡（全局奖励）
        total = self.zone_loads.sum(dim=1)
        std = torch.std(self.zone_loads, dim=1)
        zb_bonus = float(self.cfg.get('reward_config', {}).get('zone_balance_bonus', 0.3))
        mask_bal = (total > 0) & (std < torch.maximum(torch.full_like(total, 1.0), total * 0.1))
        if mask_bal.any():
            rewards[mask_bal] += (zb_bonus / max(1, N))
        # Advance time (single increment)
        self.current_time += self.time_step / 3600.0
        self.current_step += 1

        # Cache tensor outputs
        st_vec = self.get_state_vec()
        done_b = (self.current_step >= self.max_steps)
        self.last_rewards = rewards.detach().clone()
        self.last_dones = done_b.view(B, 1).float().expand(B, N).detach().clone()
        self.last_step_reward = rewards.sum(dim=1).detach().clone()
        self.last_next_state = st_vec.detach().clone()

        outs: List[Dict[str, Any]] = []
        for b in range(B):
            outs.append({
                'step_reward': float(self.last_step_reward[b].item()),
                'next_state_vec': self.last_next_state[b].cpu().numpy().astype('float32'),
                'rewards_vec': self.last_rewards[b].cpu().numpy().astype('float32'),
                'dones_vec': self.last_dones[b].cpu().numpy().astype('float32'),
            })
        return outs
