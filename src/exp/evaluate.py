from typing import List, Dict, Optional, Callable
import os
import numpy as np
from tqdm import tqdm

from exp.env_factory import create_test_env
from exp.assigners import (
    SimpleReturnAssigner,
    FixedOptimalAssigner,
    assign_tasks_dynamic,
    assign_tasks_dynamic_s_shape,
    assign_tasks_dynamic_return,
    assign_tasks_dynamic_optimal,
)
from exp.actions import smart_navigate, convert_to_dynamic_actions, find_adjacent_accessible_position
from exp.trainers import train_flat_dqn


def evaluate_method(method_name: str,
                    width: int, height: int, n_pickers: int, n_shelves: int, n_stations: int,
                    order_rate: int, max_items: int,
                    n_episodes: int, target_orders: int, max_time_limit: int,
                    training_steps: int, batch_size: int, learning_rate: float,
                    buffer_size: int, update_freq: int, target_update_freq: int,
                    hidden_dim: int,
                    # Debug/visualization options
                    verbose: bool = False,
                    log_every: int = 1,
                    save_plots: bool = False,
                    plot_dir: str = None,
                    plot_every: int = 1,
                    debug_first_episode_only: bool = True,
                    make_animation: bool = True,
                    animation_fps: int = 5,
                    # Runtime control: external control hook for env
                    control_hook=None,
                    speed_function=None,
                    # Plot config
                    plot_figsize=None,
                    **kwargs) -> Dict:
    """精简评估：支持规则三法 + DQN（Guided/Pure）。
    其余方法（Softmax/NL-HMARL）可按需扩展。
    返回：包含完成率/平均完成时间/等待时间/利用率等指标的字典。
    """


    # 允许主程序传入自定义环境构造器（例如直接使用 DynamicWarehouseEnv）
    env_ctor: Optional[Callable[[Dict], object]] = kwargs.get('env_ctor')
    if env_ctor is not None:
        cfg = {
            'width': width,
            'height': height,
            'n_pickers': n_pickers,
            'n_stations': n_stations,
            'n_charging_pads': 1,
            'levels_per_shelf': 3,
            'time_step': 2.0,
            'order_config': {
                'base_rate': order_rate,
                'peak_hours': [(9, 12), (14, 17)],
                'peak_multiplier': 1.6,
                'off_peak_multiplier': 0.7,
                'simulation_hours': 2,
            },
        }
        # Merge any extra env config passed by caller (e.g., forklift_ratio, min_forklifts)
        extra = kwargs.get('env_extra')
        if isinstance(extra, dict):
            # 深度合并 order_config（若提供），避免完全覆盖默认到达率/高峰配置
            if 'order_config' in extra and isinstance(extra['order_config'], dict):
                cfg['order_config'].update(extra['order_config'])
                # 其余顶层键仍可覆盖
                extra = {k: v for k, v in extra.items() if k != 'order_config'}
            cfg.update(extra)
        env = env_ctor(cfg)
    else:
        env = create_test_env(width, height, n_pickers, n_shelves, n_stations, order_rate, max_items)
    
    model = None
    # Strict method matching: DQN or NL-HMARL
    is_learning = method_name in ('DQN-Guided', 'DQN-Pure')
    is_pure = (method_name == 'DQN-Pure')

    if is_learning:
        dqn_cfg = kwargs.get('dqn_cfg', {}) if isinstance(kwargs.get('dqn_cfg', {}), dict) else {}
        # Auto-detect device if requested
        _dev = str(dqn_cfg.get('device', 'auto'))
        if _dev == 'auto':
            try:
                import torch as _torch  # type: ignore
                if _torch.cuda.is_available():
                    _dev = 'cuda'
                elif hasattr(_torch.backends, 'mps') and getattr(_torch.backends.mps, 'is_available', lambda: False)():
                    _dev = 'mps'
                else:
                    _dev = 'cpu'
            except Exception:
                _dev = 'cpu'
        n_envs_cfg = int(dqn_cfg.get('n_envs', 1))
        if n_envs_cfg > 1:
            print(f"[info] Using SubprocVecEnv with n_envs={n_envs_cfg} for DQN")
            from exp.trainers import train_flat_dqn_subproc
            model = train_flat_dqn_subproc(
                width, height, n_pickers, n_shelves, n_stations,
                order_rate, max_items,
                training_steps=training_steps,
                pure_learning=is_pure,
                hidden_dim=hidden_dim,
                lr=learning_rate,
                batch_size=batch_size,
                buffer_size=buffer_size,
                update_freq=update_freq,
                target_update_freq=target_update_freq,
                log_metrics=True,
                log_every=max(1, training_steps // 200),
                metrics_dir='results/train_metrics',
                metrics_tag=method_name,
                device=_dev,
                n_envs=n_envs_cfg,
            )
        else:
            model = train_flat_dqn(width, height, n_pickers, n_shelves, n_stations,
                                   order_rate, max_items,
                                   training_steps=training_steps,
                                   pure_learning=is_pure,
                                   hidden_dim=hidden_dim,
                                   lr=learning_rate,
                                   batch_size=batch_size,
                                   buffer_size=buffer_size,
                                   update_freq=update_freq,
                                   target_update_freq=target_update_freq,
                                   # metrics logging
                                   log_metrics=True,
                                   log_every=max(1, training_steps // 200),  # ~200 points
                                   metrics_dir='results/train_metrics',
                                   metrics_tag=method_name,
                                   # pass speed function so env can step
                                   speed_function=speed_function,
                                   device=_dev)
    elif method_name in ('NL-HMARL', 'NLHMARL', 'NL_HMARL') and env_ctor is not None:
        from exp.trainers import train_nl_hmarl, train_nl_hmarl_subproc
        nl_cfg = kwargs.get('nl_cfg', {}) if isinstance(kwargs.get('nl_cfg', {}), dict) else {}
        _nl_dev = str(nl_cfg.get('device', 'cpu'))
        if _nl_dev.lower() == 'auto':
            try:
                import torch as _torch  # type: ignore
                if _torch.cuda.is_available():
                    _nl_dev = 'cuda'
                elif hasattr(_torch.backends, 'mps') and getattr(_torch.backends.mps, 'is_available', lambda: False)():
                    _nl_dev = 'mps'
                else:
                    _nl_dev = 'cpu'
            except Exception:
                _nl_dev = 'cpu'
        # Train on a fresh env with same config (use subproc vecenv if n_envs>1)
        n_envs_cfg = int(nl_cfg.get('n_envs', 1))
        if n_envs_cfg > 1:
            print(f"[info] Using SubprocVecEnv with n_envs={n_envs_cfg} for NL-HMARL")
            model = train_nl_hmarl_subproc(
                env_config=cfg,
                training_steps=training_steps,
                hidden_dim=int(nl_cfg.get('hidden_dim', hidden_dim)),
                lr=float(nl_cfg.get('manager_lr', learning_rate)),
                max_tasks=int(nl_cfg.get('max_tasks', 20)),
                gamma=float(nl_cfg.get('gamma', 0.99)),
                entropy_coef=float(nl_cfg.get('entropy_coef_manager', 0.01)),
                n_nests=int(nl_cfg.get('n_nests', 4)),
                learn_eta=bool(nl_cfg.get('learn_eta', False)),
                eta_init=float(nl_cfg.get('eta_init', 1.0)),
                device=_nl_dev,
                n_envs=n_envs_cfg,
                log_metrics=True,
                log_every=int(nl_cfg.get('train_log_every', max(1, training_steps // 200))),
                metrics_dir='results/train_metrics',
                metrics_tag='NL-HMARL',
            )
        else:
            model = train_nl_hmarl(
                env_ctor=env_ctor,
                env_config=cfg,
                training_steps=training_steps,
                hidden_dim=int(nl_cfg.get('hidden_dim', hidden_dim)),
                lr=float(nl_cfg.get('manager_lr', learning_rate)),
                max_tasks=int(nl_cfg.get('max_tasks', 20)),
                gamma=float(nl_cfg.get('gamma', 0.99)),
                update_every=int(nl_cfg.get('update_every', 8)),
                entropy_coef=float(nl_cfg.get('entropy_coef_manager', 0.01)),
                n_nests=int(nl_cfg.get('n_nests', 4)),
                learn_eta=bool(nl_cfg.get('learn_eta', False)),
                eta_init=float(nl_cfg.get('eta_init', 1.0)),
                device=_nl_dev,
                speed_function=speed_function,
                log_metrics=True,
                log_every=int(nl_cfg.get('train_log_every', max(1, training_steps // 200))),
                metrics_dir='results/train_metrics',
                metrics_tag='NL-HMARL',
                n_envs=1,
            )
    elif method_name in ('NL-HMARL-AC', 'NLHMARL-AC', 'NL_HMARL_AC') and env_ctor is not None:
        from exp.trainers import train_nl_hmarl_ac
        nl_cfg = kwargs.get('nl_cfg', {}) if isinstance(kwargs.get('nl_cfg', {}), dict) else {}
        _nl_dev = str(nl_cfg.get('device', 'cpu'))
        if _nl_dev.lower() == 'auto':
            try:
                import torch as _torch  # type: ignore
                if _torch.cuda.is_available():
                    _nl_dev = 'cuda'
                elif hasattr(_torch.backends, 'mps') and getattr(_torch.backends.mps, 'is_available', lambda: False)():
                    _nl_dev = 'mps'
                else:
                    _nl_dev = 'cpu'
            except Exception:
                _nl_dev = 'cpu'
        n_envs_cfg = int(nl_cfg.get('n_envs', 1))
        if n_envs_cfg > 1:
            print(f"[info] Using SubprocVecEnv with n_envs={n_envs_cfg} for NL-HMARL-AC")
            from exp.trainers import train_nl_hmarl_ac_subproc
            model = train_nl_hmarl_ac_subproc(
                env_config=cfg,
                training_steps=training_steps,
                hidden_dim=int(nl_cfg.get('hidden_dim', hidden_dim)),
                lr_manager=float(nl_cfg.get('manager_lr', learning_rate)),
                lr_workers=float(nl_cfg.get('worker_lr', learning_rate)),
                max_tasks=int(nl_cfg.get('max_tasks', 20)),
                gamma=float(nl_cfg.get('gamma', 0.99)),
                entropy_coef_manager=float(nl_cfg.get('entropy_coef_manager', 0.01)),
                entropy_coef_workers=float(nl_cfg.get('entropy_coef_workers', 0.01)),
                n_nests=int(nl_cfg.get('n_nests', 4)),
                learn_eta=bool(nl_cfg.get('learn_eta', False)),
                eta_init=float(nl_cfg.get('eta_init', 1.0)),
                device=_nl_dev,
                n_envs=n_envs_cfg,
                log_metrics=True,
                log_every=int(nl_cfg.get('train_log_every', max(1, training_steps // 200))),
                metrics_dir='results/train_metrics',
                metrics_tag='NL-HMARL-AC',
            )
        else:
            model = train_nl_hmarl_ac(
                env_ctor=env_ctor,
                env_config=cfg,
                training_steps=training_steps,
                hidden_dim=int(nl_cfg.get('hidden_dim', hidden_dim)),
                lr_manager=float(nl_cfg.get('manager_lr', learning_rate)),
                lr_workers=float(nl_cfg.get('worker_lr', learning_rate)),
                max_tasks=int(nl_cfg.get('max_tasks', 20)),
                gamma=float(nl_cfg.get('gamma', 0.99)),
                entropy_coef_manager=float(nl_cfg.get('entropy_coef_manager', 0.01)),
                entropy_coef_workers=float(nl_cfg.get('entropy_coef_workers', 0.01)),
                n_nests=int(nl_cfg.get('n_nests', 4)),
                learn_eta=bool(nl_cfg.get('learn_eta', False)),
                eta_init=float(nl_cfg.get('eta_init', 1.0)),
                device=_nl_dev,
                speed_function=speed_function,
                log_metrics=True,
                log_every=int(nl_cfg.get('train_log_every', max(1, training_steps // 200))),
                metrics_dir='results/train_metrics',
                metrics_tag='NL-HMARL-AC',
                n_envs=int(nl_cfg.get('n_envs', 1)),
            )

    # 规则分配器（静态环境用；动态环境由实验侧分配或简单就地导航）
    # 动态环境使用对应的任务池分配器（S-Shape/Return/Optimal）
    # 分配器：包一层以注入 value 权重
    assign_value_weight = float(kwargs.get('assign_value_weight', 0.05))
    if method_name == 'S-Shape':
        dynamic_assign = lambda e: assign_tasks_dynamic_s_shape(e, value_weight=assign_value_weight)
    elif method_name == 'Return':
        dynamic_assign = lambda e: assign_tasks_dynamic_return(e, value_weight=assign_value_weight)
    elif method_name == 'Optimal':
        dynamic_assign = lambda e: assign_tasks_dynamic_optimal(e, value_weight=assign_value_weight)
    elif method_name in ('NL-HMARL', 'NLHMARL', 'NL_HMARL') and model is not None:
        # Use trained NL manager for assignment; workers use heuristic navigation during eval
        from exp.obs import get_global_state, get_task_features
        from env.dynamic_warehouse_env import TaskStatus
        import numpy as _np
        nl_cfg = kwargs.get('nl_cfg', {}) if isinstance(kwargs.get('nl_cfg', {}), dict) else {}
        _det_eval = bool(nl_cfg.get('deterministic_eval', False))
        def _assign_with_model(e):
            # Build features
            state_vec = get_global_state(e)
            task_feats = get_task_features(e, max_tasks=model.n_tasks, pending_only=True)
            nest_ids = _np.full((model.n_tasks,), -1, dtype=_np.int64)
            mask = _np.zeros((model.n_tasks,), dtype=_np.bool_)
            t_list = [t for t in getattr(e, 'task_pool', []) if t.status == TaskStatus.PENDING][:model.n_tasks]
            for i, t in enumerate(t_list):
                # Nest by forklift requirement: 1 if requires_car else 0
                nest_ids[i] = 1 if bool(getattr(t, 'requires_car', False)) else 0
                mask[i] = (t.status == TaskStatus.PENDING)
            # For each free picker, sample a task and assign (without duplication)
            local_mask = mask.copy()
            free_ids = [i for i, p in enumerate(e.pickers) if getattr(p, 'current_task', None) is None and len(p.carrying_items) == 0]
            if not free_ids or not local_mask.any():
                return 0
            import torch as _torch
            s = _torch.tensor(state_vec, dtype=_torch.float32).unsqueeze(0)
            tf = _torch.tensor(task_feats, dtype=_torch.float32).unsqueeze(0)
            nid = _torch.tensor(nest_ids, dtype=_torch.long).unsqueeze(0)
            assigned = 0
            for pid in free_ids:
                m = _torch.tensor(local_mask, dtype=_torch.bool).unsqueeze(0)
                sel, _ = model.select_tasks(s, tf, nid, m, deterministic=_det_eval)
                idx = int(sel.item())
                if not local_mask[idx] or idx >= len(t_list):
                    continue
                t = t_list[idx]
                if t.status != TaskStatus.PENDING:
                    continue
                t.status = TaskStatus.ASSIGNED
                t.assigned_picker = pid
                e.pickers[pid].current_task = t
                local_mask[idx] = False
                assigned += 1
            return assigned
        dynamic_assign = _assign_with_model
    elif method_name in ('NL-HMARL-AC', 'NLHMARL-AC', 'NL_HMARL_AC') and model is not None:
        # Assignment via NL manager; worker actions via learned policy
        from exp.obs import get_global_state, get_task_features, get_agent_observation
        from env.dynamic_warehouse_env import TaskStatus
        import numpy as _np
        import torch as _torch
        nl_cfg = kwargs.get('nl_cfg', {}) if isinstance(kwargs.get('nl_cfg', {}), dict) else {}
        _det_eval = bool(nl_cfg.get('deterministic_eval', False))
        def _assign_with_model(e):
            state_vec = get_global_state(e)
            task_feats = get_task_features(e, max_tasks=model.n_tasks, pending_only=True)
            nest_ids = _np.full((model.n_tasks,), -1, dtype=_np.int64)
            mask = _np.zeros((model.n_tasks,), dtype=_np.bool_)
            t_list = [t for t in getattr(e, 'task_pool', []) if t.status == TaskStatus.PENDING][:model.n_tasks]
            for i, t in enumerate(t_list):
                # Nest by forklift requirement: 1 if requires_car else 0
                nest_ids[i] = 1 if bool(getattr(t, 'requires_car', False)) else 0
                mask[i] = (t.status == TaskStatus.PENDING)
            local_mask = mask.copy()
            free_ids = [i for i, p in enumerate(e.pickers) if getattr(p, 'current_task', None) is None and len(p.carrying_items) == 0]
            if not free_ids or not local_mask.any():
                return 0
            s = _torch.tensor(state_vec, dtype=_torch.float32).unsqueeze(0)
            tf = _torch.tensor(task_feats, dtype=_torch.float32).unsqueeze(0)
            nid = _torch.tensor(nest_ids, dtype=_torch.long).unsqueeze(0)
            assigned = 0
            for pid in free_ids:
                m = _torch.tensor(local_mask, dtype=_torch.bool).unsqueeze(0)
                sel, _ = model.select_tasks(s, tf, nid, m, deterministic=_det_eval)
                idx = int(sel.item())
                if not local_mask[idx] or idx >= len(t_list):
                    continue
                t = t_list[idx]
                if t.status != TaskStatus.PENDING:
                    continue
                t.status = TaskStatus.ASSIGNED
                t.assigned_picker = pid
                e.pickers[pid].current_task = t
                local_mask[idx] = False
                assigned += 1
            return assigned
        dynamic_assign = _assign_with_model
    else:
        dynamic_assign = lambda e: assign_tasks_dynamic(e)

    all_episode_metrics = []
    # Prepare plotting directory if enabled
    if save_plots:
        base_plot_dir = plot_dir or os.path.join('results', 'frames')
        os.makedirs(base_plot_dir, exist_ok=True)
    ep_bar = tqdm(range(n_episodes), desc=f"Eval {method_name}", ncols=100)
    for ep in ep_bar:
        env.reset()
        # 注册外部控制 hook（按需）
        if control_hook is not None:
            try:
                env.set_control_hook(control_hook)
            except Exception:
                pass
        # 注册统一速度函数（必须，与环境强制要求一致）；并做一次覆盖范围校验
        if speed_function is None:
            raise RuntimeError("speed_function is required by environment; please pass speed_function=... to evaluate_method")
        env.set_speed_function(speed_function)
        # 立即做一次校验：应为每个 picker 提供正数速度
        speeds_probe = speed_function(env)
        if not isinstance(speeds_probe, dict):
            raise RuntimeError("speed_function must return a dict {picker_id: speed}.")
        missing = [p.id for p in env.pickers if p.id not in speeds_probe]
        if missing:
            raise RuntimeError(f"speed_function missing speed for pickers {missing}.")
        bad = {pid: v for pid, v in speeds_probe.items() if float(v) <= 0}
        if bad:
            raise RuntimeError(f"speed_function returned non-positive speeds: {bad}.")
        do_verbose = verbose and (not debug_first_episode_only or ep == 0)
        # Decide whether to save frames for this episode
        save_plots_this = save_plots and (not debug_first_episode_only or ep == 0)
        # Subdirectory per episode for frames (only when enabled for this episode)
        if save_plots_this:
            ep_dir = os.path.join(base_plot_dir, f"{method_name}_ep{ep:02d}")
            os.makedirs(ep_dir, exist_ok=True)
        if do_verbose:
            print(f"\n=== [{method_name}] Episode {ep+1}/{n_episodes} ===")
        metrics = {
            'steps': 0,
            'orders_completed': 0,
            'orders_generated': 0,
            # value 统计
            'raw_value_completed': 0,      # 原始（base）价值累计（仅完成的任务）
            'decayed_value_completed': 0,  # 衰减后价值累计（仅完成的任务）
            # 其他调试指标（保留但不进入最终 results 表）
            'picker_utilization': 0.0,
            'completion_time': 0,
            'order_waiting_times': [],
            'pick_attempts': 0, 'pick_success': 0,
            'drop_attempts': 0, 'drop_success': 0,
        }
        # 每步记录表：累计/base/decayed/penalty 以及本步增量
        step_log = []
        prev_base_cum = 0
        prev_decayed_cum = 0
        prev_penalty_cum = 0

        def _append_step_row(step_idx: int):
            nonlocal prev_base_cum, prev_decayed_cum, prev_penalty_cum
            base_cum = int(metrics.get('raw_value_completed', 0))
            decayed_cum = int(metrics.get('decayed_value_completed', 0))
            penalty_cum = int(getattr(env, 'total_value_penalty', 0))
            row = {
                'method': method_name,
                'episode': int(ep),
                'step': int(step_idx),
                'time_h': float(env.current_time),
                'base_cum': base_cum,
                'decayed_cum': decayed_cum,
                'penalty_cum': penalty_cum,
                'base_add': base_cum - prev_base_cum,
                'decayed_add': decayed_cum - prev_decayed_cum,
                'penalty_add': penalty_cum - prev_penalty_cum,
            }
            step_log.append(row)
            prev_base_cum, prev_decayed_cum, prev_penalty_cum = base_cum, decayed_cum, penalty_cum
        for step in range(max_time_limit):
            # 记录步前状态用于统计成功/尝试
            prev_carrying = [len(p.carrying_items) > 0 for p in env.pickers]
            prev_pos = [(p.x, p.y) for p in env.pickers]
            prev_tasks = [getattr(p, 'current_task', None) for p in env.pickers]

            # 每步先进行一次分配（按方法）
            dynamic_assign(env)
            actions = {}
            # 学习法：直接用模型选；规则法：简单导航
            if is_learning and model is not None:
                # 简单策略：未携货→靠近“货架相邻可达格”；相邻则IDLE；携货→靠近站点，相邻则IDLE
                # Try GPU batch nav first
                try:
                    import torch
                    from exp.gpu_nav import gpu_smart_navigate_batch
                    grid_t = torch.as_tensor(env.grid, dtype=torch.int32)
                    N = len(env.pickers)
                    picker_xy = torch.tensor([[p.x, p.y] for p in env.pickers], dtype=torch.long)
                    target_xy = torch.full((N, 2), -1, dtype=torch.long)
                    for i, p in enumerate(env.pickers):
                        if getattr(p, 'current_task', None) is None:
                            continue
                        t = p.current_task
                        if len(p.carrying_items) == 0 and t.shelf_id is not None and t.shelf_id < len(env.shelves):
                            sh = env.shelves[t.shelf_id]
                            adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
                            tx, ty = (adj if adj is not None else (sh['x'], sh['y']))
                        elif len(p.carrying_items) > 0 and t.station_id is not None and t.station_id < len(env.stations):
                            st = env.stations[t.station_id]
                            tx, ty = (st['x'], st['y'])
                        else:
                            continue
                        target_xy[i] = torch.tensor([tx, ty], dtype=torch.long)
                    nav = gpu_smart_navigate_batch(grid_t, picker_xy, target_xy)
                    actions = {}
                    manhattan = (picker_xy - target_xy.clamp(min=0)).abs().sum(dim=1)
                    for i in range(N):
                        if target_xy[i, 0] >= 0 and manhattan[i].item() == 1:
                            actions[i] = 4
                        else:
                            actions[i] = int(nav[i].item())
                except Exception:
                    actions = {}
                    for i, p in enumerate(env.pickers):
                        if getattr(p, 'current_task', None) is None:
                            actions[i] = 4
                            continue
                        t = p.current_task
                        if len(p.carrying_items) == 0:
                            sh = env.shelves[t.shelf_id]
                            adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
                            if adj is None or (p.x, p.y) == adj or (abs(p.x - sh['x']) + abs(p.y - sh['y']) == 1):
                                actions[i] = 4
                            else:
                                actions[i] = smart_navigate(p, adj, env)
                        else:
                            st = env.stations[t.station_id]
                            if abs(p.x - st['x']) + abs(p.y - st['y']) == 1:
                                actions[i] = 4
                            else:
                                actions[i] = smart_navigate(p, (st['x'], st['y']), env)
            elif method_name in ('NL-HMARL-AC', 'NLHMARL-AC', 'NL_HMARL_AC') and model is not None:
                # Use worker policy to act; sanitize invalid moves
                from exp.obs import get_agent_observation
                import torch as _torch
                import numpy as _np
                obs_batch = [get_agent_observation(env, p, include_global=True) for p in env.pickers]
                obs_tensor = _torch.tensor(_np.vstack(obs_batch), dtype=_torch.float32)
                with _torch.no_grad():
                    outs = model.workers(obs_tensor)
                    probs = outs['action_probs']
                    acts = _torch.multinomial(_torch.clamp(probs, min=1e-8), num_samples=1).squeeze(1)
                for i, p in enumerate(env.pickers):
                    a = int(acts[i].item())
                    if a in (0, 1, 2, 3):
                        dd = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[a]
                        nx, ny = p.x + dd[0], p.y + dd[1]
                        invalid = not (0 <= nx < env.width and 0 <= ny < env.height) or (env.grid[ny, nx] == 2)
                        if invalid:
                            t = getattr(p, 'current_task', None)
                            target = None
                            if t is not None:
                                if p.carrying_items and t.station_id is not None and t.station_id < len(env.stations):
                                    st = env.stations[t.station_id]
                                    target = (st['x'], st['y'])
                                elif (not p.carrying_items) and t.shelf_id is not None and t.shelf_id < len(env.shelves):
                                    sh = env.shelves[t.shelf_id]
                                    adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
                                    target = adj if adj is not None else (sh['x'], sh['y'])
                            if target is not None:
                                a = smart_navigate(p, target, env)
                            else:
                                a = 4
                    actions[i] = a
            else:
                for i, p in enumerate(env.pickers):
                    # 规则：未携货靠近“货架相邻可达格”，相邻则IDLE；携货靠近站点，相邻则IDLE
                    if getattr(p, 'current_task', None) is None:
                        actions[i] = 4
                        continue
                    t = p.current_task
                    if len(p.carrying_items) == 0:
                        sh = env.shelves[t.shelf_id]
                        adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
                        if adj is None:
                            actions[i] = 4
                        elif (p.x, p.y) == adj or (abs(p.x - sh['x']) + abs(p.y - sh['y']) == 1):
                            actions[i] = 4
                        else:
                            actions[i] = smart_navigate(p, adj, env)
                    else:
                        st = env.stations[t.station_id]
                        actions[i] = 4 if abs(p.x - st['x']) + abs(p.y - st['y']) == 1 \
                                          else smart_navigate(p, (st['x'], st['y']), env)

            # （移除 pre 阶段调试输出，仅保留 post 阶段）

            # 评估阶段启发式动作已是环境索引空间 → 使用 input_space='env'
            # 在下发前，防止选择会撞上货架/越界的方向：若检测到无效移动，用 smart_navigate 修正。
            raw_actions = dict(actions)
            for i, a in list(actions.items()):
                if a in (0, 1, 2, 3):
                    p = env.pickers[i]
                    dd = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[a]
                    nx, ny = p.x + dd[0], p.y + dd[1]
                    invalid = not (0 <= nx < env.width and 0 <= ny < env.height) or (env.grid[ny, nx] == 2)
                    if invalid:
                        # 依据当前状态选择合理目标，再次调用 smart_navigate 修正方向
                        t = getattr(p, 'current_task', None)
                        target = None
                        if t is not None:
                            if p.carrying_items and t.station_id is not None and t.station_id < len(env.stations):
                                st = env.stations[t.station_id]
                                adj_st = find_adjacent_accessible_position(env, (st['x'], st['y']), (p.x, p.y))
                                target = adj_st if adj_st is not None else (st['x'], st['y'])
                            elif (not p.carrying_items) and t.shelf_id is not None and t.shelf_id < len(env.shelves):
                                sh = env.shelves[t.shelf_id]
                                adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
                                target = adj if adj is not None else (sh['x'], sh['y'])
                        if target is not None:
                            actions[i] = smart_navigate(p, target, env)
                        else:
                            actions[i] = 4  # 回退为原地
            env_actions = convert_to_dynamic_actions(actions, env, input_space='env')
            _, _, dones, info = env.step(env_actions)

            # 调试：打印本步速度与动作（执行后更准确）
            if do_verbose and (log_every and log_every > 0) and (step % log_every == 0):
                def _act_str(a):
                    m = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'IDLE', 5: 'PICK', 6: 'DROP'}
                    return m.get(int(a), str(a))
                # Post-step summary with updated value/carrying (decayed)
                _cum_val = int(getattr(env, 'total_value_completed', 0))
                _carry_base = 0
                _carry_decay = 0
                for __p in env.pickers:
                    try:
                        if getattr(__p, 'carrying_items', None) and getattr(__p, 'current_task', None) is not None:
                            comps = env.get_task_value_components(__p.current_task)
                            _carry_base += int(comps.get('base', 0))
                            _carry_decay += int(comps.get('decay', 0))
                    except Exception:
                        pass
                _pen = int(getattr(env, 'total_value_penalty', 0))
                print(f"Step {step} post: value(decayed)={_cum_val}, carrying(base-decay)={_carry_base}-{_carry_decay}, penalty={_pen}")
                print("    speeds/actions:")
                for i, p in enumerate(env.pickers):
                    sp = getattr(env, 'last_speeds', {}).get(i, getattr(p, 'speed', 1.0))
                    fx = getattr(p, 'fx', float(p.x))
                    fy = getattr(p, 'fy', float(p.y))
                    # 事件检测：基于携货标志的变化
                    now_carry = bool(p.carrying_items)
                    was_carry = bool(prev_carrying[i])
                    evt = 'PICK' if (not was_carry and now_carry) else ('DROP' if (was_carry and not now_carry) else '-')
                    # 当前携带价值（若有，显示为 base-decay）
                    carry_base = 0
                    carry_decay = 0
                    if now_carry and getattr(p, 'current_task', None) is not None:
                        try:
                            comps = env.get_task_value_components(p.current_task)
                            carry_base = int(comps.get('base', 0))
                            carry_decay = int(comps.get('decay', 0))
                        except Exception:
                            carry_base, carry_decay = 0, 0
                    print(f"      - picker {i}: speed={sp:.3f}, raw={_act_str(raw_actions.get(i, 4))}, env={_act_str(env_actions.get(i, 4))}, pos=({fx:.2f},{fy:.2f}), evt={evt}, carry(base-decay)={carry_base}-{carry_decay}")
                    # 当速度很低时，打印更详细的分解信息
                    comps = getattr(env, 'last_speed_components', {}).get(i, None)
                    if comps and sp <= 0.2:
                        print(
                            "        details: base_used={:.3f}, picker_base={:.3f}, override={}, carrying={}, weight={:.2f}, wt_term={:.3f}, cong_red={:.3f}, using_speed_fn={}"
                            .format(
                                comps.get('base_used', float('nan')),
                                comps.get('picker_base', float('nan')),
                                'NA' if np.isnan(comps.get('override', float('nan'))) else f"{comps.get('override'):.3f}",
                                int(comps.get('carrying', 0.0)),
                                comps.get('weight', 0.0),
                                comps.get('weight_term', 0.0),
                                comps.get('congestion_red', 0.0),
                                int(comps.get('using_speed_fn', 0.0))
                            )
                        )

            # 可视化帧保存
            if save_plots_this and (step % max(1, plot_every) == 0):
                save_path = os.path.join(ep_dir, f"t{step:04d}.png")
                if plot_figsize is not None:
                    env.plot(save_path=save_path, show=False, figsize=plot_figsize)
                else:
                    env.plot(save_path=save_path, show=False)

            # 统计 PICK/DROP 尝试与成功
            for i, p in enumerate(env.pickers):
                t = getattr(p, 'current_task', None)
                # 拣尝试：步前未携货、有任务，且动作为 IDLE（或5/6），并且与货架相邻
                if prev_tasks[i] is not None and not prev_carrying[i]:
                    if env_actions.get(i, 4) == 4:
                        if prev_tasks[i].shelf_id is not None and prev_tasks[i].shelf_id < len(env.shelves):
                            sh = env.shelves[prev_tasks[i].shelf_id]
                            if abs(prev_pos[i][0]-sh['x']) + abs(prev_pos[i][1]-sh['y']) == 1:
                                metrics['pick_attempts'] += 1
                                if len(p.carrying_items) > 0:
                                    metrics['pick_success'] += 1
                # 投尝试：步前携货、有任务，且动作为 IDLE，并且与站点相邻
                if prev_tasks[i] is not None and prev_carrying[i]:
                    if env_actions.get(i, 4) == 4:
                        if prev_tasks[i].station_id is not None and prev_tasks[i].station_id < len(env.stations):
                            st = env.stations[prev_tasks[i].station_id]
                            if abs(prev_pos[i][0]-st['x']) + abs(prev_pos[i][1]-st['y']) == 1:
                                metrics['drop_attempts'] += 1
                                if len(p.carrying_items) == 0:
                                    metrics['drop_success'] += 1
            
            # 简单终止：累计本步完成的任务数，达到目标则提前结束
            completed_this_step = len(info.get('tasks_completed', []))
            if completed_this_step > 0:
                metrics['orders_completed'] += completed_this_step
                if do_verbose:
                    tids = info.get('tasks_completed', [])
                    print("    completed tasks:")
                    # 为每个完成的任务逐行打印详细信息
                    for tid in tids:
                        # 在任务池中查找该任务以获取详情
                        t = None
                        for _t in getattr(env, 'task_pool', []):
                            if getattr(_t, 'task_id', None) == tid:
                                t = _t
                                break
                        if t is not None:
                            comp_t = getattr(t, 'completion_time', None)
                            on_time = (comp_t is not None and comp_t <= getattr(t, 'deadline', float('inf')))
                            wnum = getattr(t, 'weight', None)
                            wcls = getattr(t, 'weight_class', '?')
                            try:
                                comps = env.get_task_value_components(t, at_time=comp_t if comp_t is not None else env.current_time)
                                __val_decayed = int(comps.get('decayed', 0))
                                __val_base = int(comps.get('base', 0))
                                __val_decay = int(comps.get('decay', 0))
                                # 统计本步完成任务的 value（原始与衰减）
                                metrics['raw_value_completed'] += max(0, __val_base)
                                metrics['decayed_value_completed'] += max(0, __val_decayed)
                            except Exception:
                                __val_decayed, __val_base, __val_decay = 0, 0, 0
                            print(
                                f"      - id={tid}, order={getattr(t,'order_id',None)}, shelf={getattr(t,'shelf_id',None)}, station={getattr(t,'station_id',None)}, "
                                f"weight={wnum if wnum is not None else 'NA'}({wcls}), val+=(decayed={__val_decayed}, base={__val_base}, decay={__val_decay}), "
                                f"car={getattr(t,'requires_car', False)}, t={comp_t if comp_t is not None else env.current_time:.3f}h, on_time={on_time}"
                            )
                        else:
                            print(f"      - id={tid}")
                # 记录首个达标的完成时间
                if (isinstance(target_orders, (int, float)) and target_orders > 0 
                        and metrics['orders_completed'] >= target_orders and metrics['completion_time'] == 0):
                    metrics['completion_time'] = metrics['steps'] + 1  # 当前步计入完成
                    # 记录本步数据再提前结束本集
                    _append_step_row(metrics['steps'] + 1)
                    break
                # 提前为下一步分配新任务：本步已完成、已释放的拣货员在下一步直接开始新任务
                dynamic_assign(env)
            metrics['steps'] += 1
            _append_step_row(metrics['steps'])
            if metrics['steps'] >= max_time_limit:
                metrics['completion_time'] = max_time_limit
                break
        
        # 记录本集完成价值与利用率
        metrics['picker_utilization'] = np.mean([env.picker_utilization[i]/max(1, metrics['steps'])
                                                  for i in range(env.n_pickers)])
        # 总计 value：环境口径（衰减后的完成值 - 销毁罚没）
        metrics['value_completed'] = int(getattr(env, 'total_value_completed', 0))
        # 本集累计罚没
        metrics['penalty_cum'] = int(getattr(env, 'total_value_penalty', 0))
        all_episode_metrics.append(metrics)
        # 更新每个 episode 的简要进度
        pick_rate = (metrics['pick_success'] / metrics['pick_attempts']) if metrics['pick_attempts'] else 0.0
        drop_rate = (metrics['drop_success'] / metrics['drop_attempts']) if metrics['drop_attempts'] else 0.0
        ep_bar.set_postfix(pick=f"{pick_rate:.2f}", drop=f"{drop_rate:.2f}", steps=metrics['steps'])

        # 将本集的帧合成为动图/视频（仅对保存帧的集）
        if save_plots_this and make_animation:
            try:
                _compile_frames_to_media(ep_dir, fps=animation_fps)
            except Exception as e:
                print(f"[warn] Failed to compile animation for {method_name} ep{ep}: {e}")

        # 保存每步 penalty/base/decayed 记录
        try:
            import pandas as _pd  # type: ignore
            steps_dir = os.path.join('results', 'steps', method_name)
            os.makedirs(steps_dir, exist_ok=True)
            _df = _pd.DataFrame(step_log)
            _csv_path = os.path.join(steps_dir, f"ep{ep:02d}_steps.csv")
            _df.to_csv(_csv_path, index=False)
            if do_verbose:
                print(f"[info] wrote step log: {_csv_path}")
        except Exception as e:
            try:
                import csv as _csv
                steps_dir = os.path.join('results', 'steps', method_name)
                os.makedirs(steps_dir, exist_ok=True)
                _csv_path = os.path.join(steps_dir, f"ep{ep:02d}_steps.csv")
                with open(_csv_path, 'w', newline='') as f:
                    writer = _csv.DictWriter(f, fieldnames=['method','episode','step','time_h','base_cum','decayed_cum','penalty_cum','base_add','decayed_add','penalty_add'])
                    writer.writeheader()
                    for r in step_log:
                        writer.writerow(r)
                if do_verbose:
                    print(f"[info] wrote step log (csv): {_csv_path}")
            except Exception as e2:
                print(f"[warn] failed to write step log: {e}; fallback error: {e2}")

    # 汇总
    # 汇总：仅保留 value 相关列（原始、衰减、总计）
    total_raw_value = int(sum(m.get('raw_value_completed', 0) for m in all_episode_metrics))
    total_decayed_value = int(sum(m.get('decayed_value_completed', 0) for m in all_episode_metrics))
    total_value = int(sum(m.get('value_completed', 0) for m in all_episode_metrics))  # 包含销毁罚没
    total_penalty = int(sum(m.get('penalty_cum', 0) for m in all_episode_metrics))
    results = {
        'method': method_name,
        'raw_value': total_raw_value,
        'decayed_value': total_decayed_value,
        'total_value': total_value,
        'penalty': total_penalty,
    }
    return results


def _compile_frames_to_media(frames_dir: str, fps: int = 5) -> None:
    """将 frames_dir 下的 tXXXX.png 合成为 GIF（优先）和 MP4（若可用）。
    输出文件保存到 frames_dir 下，文件名分别为 animation.gif / animation.mp4。
    """
    files = [f for f in os.listdir(frames_dir) if f.endswith('.png') and f.startswith('t')]
    if not files:
        return
    files.sort()
    paths = [os.path.join(frames_dir, f) for f in files]

    # 1) GIF via PIL
    gif_path = os.path.join(frames_dir, 'animation.gif')
    try:
        from PIL import Image  # type: ignore
        imgs = [Image.open(p) for p in paths]
        duration = max(1, int(1000 / max(1, fps)))  # ms per frame
        imgs[0].save(
            gif_path,
            save_all=True,
            append_images=imgs[1:],
            duration=duration,
            loop=0
        )
        # Close images
        for im in imgs:
            try:
                im.close()
            except Exception:
                pass
    except Exception as e:
        print(f"[warn] GIF export skipped (PIL not available or failed): {e}")

    # 2) MP4 via imageio (best-effort)
    mp4_path = os.path.join(frames_dir, 'animation.mp4')
    try:
        import imageio.v2 as imageio  # type: ignore
        with imageio.get_writer(mp4_path, fps=fps) as writer:
            for p in paths:
                writer.append_data(imageio.imread(p))
    except Exception as e:
        # Non-fatal if codecs not present
        print(f"[warn] MP4 export skipped: {e}")
