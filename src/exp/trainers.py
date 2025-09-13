from typing import Dict, Optional, Callable
import numpy as np
import torch
from tqdm import tqdm

from baselines.flat_marl_dqn import FlatMARLDQN
from exp.obs import get_agent_observation
from exp.actions import get_valid_actions, convert_to_dynamic_actions
from exp.assigners import assign_tasks_dynamic
from exp.obs import get_global_state, get_task_features
from exp.vecenv import SubprocVecEnv
from exp.actions import find_adjacent_accessible_position, smart_navigate


def _try_gpu_nav_actions(env):
    """Attempt to compute heuristic navigation actions on GPU in batch.
    Returns a dict {i: action} or None on failure.
    """
    try:
        import torch
        from exp.gpu_nav import gpu_smart_navigate_batch
        # Grid to tensor
        grid = torch.as_tensor(env.grid, dtype=torch.int32)
        # Picker and target tensors
        N = len(env.pickers)
        picker_xy = torch.tensor([[p.x, p.y] for p in env.pickers], dtype=torch.long)
        target_xy = torch.full((N, 2), -1, dtype=torch.long)
        for i, p in enumerate(env.pickers):
            t = getattr(p, 'current_task', None)
            if t is None:
                continue
            if len(p.carrying_items) == 0:
                if t.shelf_id is None or t.shelf_id >= len(env.shelves):
                    continue
                sh = env.shelves[t.shelf_id]
                adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
                tx, ty = (adj if adj is not None else (sh['x'], sh['y']))
            else:
                if t.station_id is None or t.station_id >= len(env.stations):
                    continue
                st = env.stations[t.station_id]
                tx, ty = (st['x'], st['y'])
            target_xy[i] = torch.tensor([tx, ty], dtype=torch.long)
        actions_t = gpu_smart_navigate_batch(grid, picker_xy, target_xy)
        # Force IDLE when adjacent to allow pick/drop
        tx = target_xy[:, 0].clamp(min=0)
        ty = target_xy[:, 1].clamp(min=0)
        manhattan = (picker_xy - torch.stack([tx, ty], dim=1)).abs().sum(dim=1)
        adj_mask = (target_xy[:, 0] >= 0) & (manhattan == 1)
        actions_t = torch.where(adj_mask, torch.full_like(actions_t, 4), actions_t)
        return {i: int(actions_t[i].item()) for i in range(N)}
    except Exception:
        return None


def _enable_torch_perf():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision('high')  # PyTorch >= 1.12
    except Exception:
        pass


def train_flat_dqn(
    width: int, height: int,
    n_pickers: int, n_shelves: int, n_stations: int,
    order_rate: int, max_items: int,
    training_steps: int = 5000,
    pure_learning: bool = False,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_size: int = 10000,
    update_freq: int = 4,
    target_update_freq: int = 100,
    # metrics logging
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: Optional[str] = 'results/train_metrics',
    metrics_tag: Optional[str] = None,
    # runtime env control
    speed_function: Optional[Callable] = None,
    # device
    device: str = 'cpu',
):
    """精简版 Flat-DQN 训练（适配当前环境）。返回已训练的模型。
    说明：
    - 观测：40/45 维（是否包含全局信息由 pure_learning 控制）
    - 动作：7（0..6，5/6 在下发时归一到 4）
    - 回放与目标网络更新等细节由 FlatMARLDQN 内部处理
    """
    _enable_torch_perf()
    from exp.env_factory import create_test_env
    env = create_test_env(width, height, n_pickers, n_shelves, n_stations, order_rate, max_items)

    obs_dim = 40 if pure_learning else 45
    action_dim = 7  # 与环境动作索引对齐: 0..3=UP/DOWN/LEFT/RIGHT, 4=IDLE, 5/6=PICK/DROP
    model = FlatMARLDQN(
        state_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_pickers,
        hidden_dim=hidden_dim,
        lr=lr,
        target_update_freq=target_update_freq,
        batch_size=batch_size,
        buffer_size=buffer_size,
        use_double_dqn=True,
        use_dueling=True,
        device=device,
    )

    _ = env.reset()
    # Register speed function (required by environment)
    if speed_function is None:
        # Fallback: use per-picker base speed as constant
        def speed_function(e):
            return {p.id: float(getattr(p, 'speed', 1.0)) for p in e.pickers}
    env.set_speed_function(speed_function)
    pbar = tqdm(range(training_steps), desc='Train DQN', ncols=100)
    avg_reward_ema = None
    last_loss: Optional[float] = None

    # Metrics buffers
    steps_log, eps_log, loss_log, stepR_log, avgR_log = [], [], [], [], []
    q_logs = []  # list of length-7 arrays (mean Q per action)
    for step in pbar:
        # 分配任务（关键）：为空闲拣货员分配可执行任务，避免永远没有拣/投事件
        assign_tasks_dynamic(env)
        # 收集观测（批量）
        obs_batch = [get_agent_observation(env, p, include_global=not pure_learning) for p in env.pickers]
        obs_tensor = torch.tensor(np.vstack(obs_batch), dtype=torch.float32, device=model.device)
        # 有效动作掩码（批量）
        masks = np.vstack([np.array(get_valid_actions(env, p), dtype=np.int32) for p in env.pickers])  # (N,7)
        # 批量 Q 计算
        with torch.no_grad():
            q_vals = model.q_network(obs_tensor)  # (N,7) on device
        q_np = q_vals.detach().cpu().numpy()
        # 应用掩码：非法动作置为 -inf
        q_np[masks == 0] = -np.inf
        # epsilon-greedy（批量）：与原实现一致地每步推进 steps_done 按代理数
        eps_start = getattr(model, 'epsilon_start', 1.0)
        eps_end = getattr(model, 'epsilon_end', 0.05)
        eps_decay = max(1, int(getattr(model, 'epsilon_decay', 100000)))
        # 使用当前 steps_done 代表本步前的进度，近似所有代理同一 epsilon
        cur_eps = eps_end + (eps_start - eps_end) * np.exp(-float(model.steps_done) / float(eps_decay))
        explore = (np.random.rand(len(env.pickers)) < cur_eps)
        chosen_actions: Dict[int, int] = {}
        for i in range(len(env.pickers)):
            if explore[i]:
                valid_idx = np.where(masks[i] == 1)[0]
                if len(valid_idx) == 0:
                    chosen_actions[i] = int(np.argmax(q_np[i])) if np.all(np.isfinite(q_np[i])) else 4
                else:
                    chosen_actions[i] = int(np.random.choice(valid_idx))
            else:
                # 贪心选择（已屏蔽非法动作）
                chosen_actions[i] = int(np.nanargmax(q_np[i])) if np.any(np.isfinite(q_np[i])) else 4
        # 推进 epsilon 计数（按代理数），并更新缓存值
        model.steps_done += len(env.pickers)
        model.epsilon = eps_end + (eps_start - eps_end) * np.exp(-float(model.steps_done) / float(eps_decay))
        # 环境索引无需重排，仅将 5/6 归一为 4 下发
        env_actions = convert_to_dynamic_actions(chosen_actions, env, input_space='env')
        next_obs, rewards, dones, info = env.step(env_actions)
        # 统计训练过程的平均奖励（EMA）
        step_reward = float(sum(rewards.values())) if isinstance(rewards, dict) else float(rewards)
        if avg_reward_ema is None:
            avg_reward_ema = step_reward
        else:
            avg_reward_ema = 0.98 * avg_reward_ema + 0.02 * step_reward
        next_obs_batch = [get_agent_observation(env, p, include_global=not pure_learning) for p in env.pickers]
        # 存回放
        for i in range(n_pickers):
            # 存储原始 DQN 动作索引，确保 5/6 的经验能够被学习
            model.store_transition(
                obs_batch[i],
                chosen_actions[i],
                rewards.get(i, 0.0),
                next_obs_batch[i],
                dones.get(i, False)
            )
        # 训练
        if step % update_freq == 0:
            loss = model.train_step()
            if loss is not None:
                last_loss = float(loss)
            # 更新进度条信息
            pbar.set_postfix(avgR=f"{avg_reward_ema:.2f}", loss=f"{(last_loss or 0):.3f}")

        # Metrics logging (every log_every steps)
        if log_metrics and (step % max(1, log_every) == 0):
            try:
                # Mean Q per action over current batch of observations
                obs_tensor = torch.tensor(np.vstack(obs_batch), dtype=torch.float32, device=model.device)
                with torch.no_grad():
                    q_vals = model.q_network(obs_tensor).detach().cpu().numpy()  # (n_agents, 7)
                q_mean = q_vals.mean(axis=0)
            except Exception:
                q_mean = np.zeros((7,), dtype=np.float32)
            steps_log.append(step)
            eps_log.append(float(getattr(model, 'epsilon', 0.0)))
            loss_log.append(float('nan') if last_loss is None else float(last_loss))
            stepR_log.append(float(step_reward))
            avgR_log.append(float(avg_reward_ema))
            q_logs.append(q_mean.tolist())

    # Save metrics after training
    if log_metrics:
        try:
            import os
            import pandas as pd
            import matplotlib.pyplot as plt
            tag = metrics_tag or 'DQN'
            out_dir = metrics_dir or 'results/train_metrics'
            out_dir = os.path.join(out_dir, tag)
            os.makedirs(out_dir, exist_ok=True)

            # Build DataFrame
            cols = ['q_up', 'q_down', 'q_left', 'q_right', 'q_idle', 'q_pick', 'q_drop']
            q_arr = np.array(q_logs) if len(q_logs) > 0 else np.zeros((0, 7))
            df = pd.DataFrame({
                'step': steps_log,
                'epsilon': eps_log,
                'loss': loss_log,
                'step_reward': stepR_log,
                'avg_reward_ema': avgR_log,
                # Unified columns for cross-method comparison (DQN fills NaN)
                'policy_loss': [float('nan')] * len(steps_log),
                'value_loss': [float('nan')] * len(steps_log),
                'entropy_loss': [float('nan')] * len(steps_log),
                'entropy': [float('nan')] * len(steps_log),
            })
            if len(q_logs) > 0:
                for i, c in enumerate(cols):
                    df[c] = q_arr[:, i]
            df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)

            # Plots
            # 1) Epsilon
            if len(steps_log) > 0:
                plt.figure(figsize=(6, 3))
                plt.plot(steps_log, eps_log, label='epsilon')
                plt.xlabel('step'); plt.ylabel('epsilon'); plt.title(f'{tag} Epsilon')
                plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'epsilon.png'))
                plt.close()

                # 2) Loss
                plt.figure(figsize=(6, 3))
                plt.plot(steps_log, loss_log, label='loss', alpha=0.8)
                plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{tag} Loss (logged)')
                plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'loss.png'))
                plt.close()

                # 3) Rewards
                plt.figure(figsize=(6, 3))
                plt.plot(steps_log, stepR_log, label='step_reward', alpha=0.5)
                plt.plot(steps_log, avgR_log, label='avg_reward_ema', alpha=0.9)
                plt.legend(); plt.xlabel('step'); plt.ylabel('reward'); plt.title(f'{tag} Rewards')
                plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'rewards.png'))
                plt.close()

                # 4) Q values per action
                if len(q_logs) > 0:
                    plt.figure(figsize=(7, 4))
                    for i, c in enumerate(cols):
                        plt.plot(steps_log, q_arr[:, i], label=c)
                    plt.legend(ncol=3, fontsize=8)
                    plt.xlabel('step'); plt.ylabel('Q'); plt.title(f'{tag} Mean Q per action')
                    plt.grid(alpha=0.3); plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, 'q_values.png'))
                    plt.close()
        except Exception as e:
            # Metrics are best-effort; avoid breaking training if plotting fails
            print(f'[warn] Failed to save training metrics: {e}')
    return model


def train_flat_dqn_subproc(
    width: int, height: int,
    n_pickers: int, n_shelves: int, n_stations: int,
    order_rate: int, max_items: int,
    training_steps: int = 5000,
    pure_learning: bool = False,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_size: int = 10000,
    update_freq: int = 4,
    target_update_freq: int = 100,
    # logging
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: Optional[str] = 'results/train_metrics',
    metrics_tag: Optional[str] = None,
    # device
    device: str = 'cpu',
    # vec
    n_envs: int = 4,
):
    """Flat-DQN with SubprocVecEnv parallel rollout.

    Notes:
    - Each env contains n_pickers agents; observations are concatenated across envs for a single forward.
    - Actions are picked per-agent with epsilon-greedy and sent back to each env.
    """
    _enable_torch_perf()
    # Build env_config consistent with env_factory.create_test_env
    env_config = {
        'width': width,
        'height': height,
        'n_pickers': n_pickers,
        'n_shelves': n_shelves,
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
    vec = SubprocVecEnv(int(max(1, n_envs)), env_config, max_tasks=20)

    obs_dim = 40 if pure_learning else 45
    action_dim = 7
    # Keep per-env target update cadence similar under parallelism
    eff_target_update = max(1, int(target_update_freq // max(1, int(max(1, n_envs)))))
    model = FlatMARLDQN(
        state_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_pickers,
        hidden_dim=hidden_dim,
        lr=lr,
        target_update_freq=eff_target_update,
        batch_size=batch_size,
        buffer_size=buffer_size,
        use_double_dqn=True,
        use_dueling=True,
        device=device,
    )

    try:
        from tqdm import tqdm
        pbar = tqdm(range(training_steps), desc='Train DQN (subproc)', ncols=100)
    except Exception:
        pbar = range(training_steps)

    avg_reward_ema = None
    last_loss: Optional[float] = None
    steps_log, eps_log, loss_log, stepR_log, avgR_log = [], [], [], [], []
    q_logs = []

    include_global = not pure_learning
    used_envs = int(max(1, n_envs))
    total_agents = used_envs * n_pickers
    import time as _time
    _t0 = _time.time()
    eff_update_freq = max(1, int(update_freq // used_envs))
    for step in pbar:
        # Collect observations and masks from all envs
        outs = vec.get_dqn_obs(include_global=include_global)
        # Stack
        obs_batches = [o['obs'] for o in outs if 'obs' in o]
        mask_batches = [o['masks'] for o in outs if 'masks' in o]
        if len(obs_batches) == 0:
            continue
        obs_all = np.vstack(obs_batches)
        masks_all = np.vstack(mask_batches)
        obs_tensor = torch.tensor(obs_all, dtype=torch.float32, device=model.device)
        with torch.no_grad():
            q_t = model.q_network(obs_tensor)  # (E*N,7)
        mask_t = torch.tensor(masks_all, dtype=torch.bool, device=model.device)
        q_t = q_t.masked_fill(~mask_t, float('-inf'))
        greedy = torch.argmax(q_t, dim=1).detach().cpu().numpy()
        # Epsilon per-current step across all agents
        eps_start = getattr(model, 'epsilon_start', 1.0)
        eps_end = getattr(model, 'epsilon_end', 0.05)
        eps_decay = max(1, int(getattr(model, 'epsilon_decay', 100000)))
        cur_eps = eps_end + (eps_start - eps_end) * np.exp(-float(model.steps_done) / float(eps_decay))
        explore = (np.random.rand(obs_all.shape[0]) < cur_eps)
        chosen = greedy.copy()
        if np.any(explore):
            for i in np.where(explore)[0]:
                valid_idx = np.where(masks_all[i] == 1)[0]
                if len(valid_idx) == 0:
                    continue
                chosen[i] = int(np.random.choice(valid_idx))
        # Update epsilon counter (per agent across all envs)
        model.steps_done += obs_all.shape[0]
        model.epsilon = eps_end + (eps_start - eps_end) * np.exp(-float(model.steps_done) / float(eps_decay))
        # Split chosen actions back per env
        actions_per_env: List[List[int]] = []
        offset = 0
        for o in outs:
            n = o['obs'].shape[0]
            actions_per_env.append(chosen[offset:offset+n].tolist())
            offset += n
        step_outs = vec.step_dqn(actions_per_env, include_global=include_global)
        # Aggregate rewards and store transitions
        total_step_reward = 0.0
        next_obs_all = []
        rewards_all = []
        dones_all = []
        offset = 0
        for i, o in enumerate(outs):
            n = o['obs'].shape[0]
            so = step_outs[i]
            total_step_reward += float(so.get('step_reward', 0.0))
            next_obs_chunk = np.array(so.get('next_obs'), dtype=np.float32)
            r_vec = np.array(so.get('rewards_vec'), dtype=np.float32)
            d_vec = np.array(so.get('dones_vec'), dtype=np.float32)
            # Store transitions per-agent
            for j in range(n):
                model.store_transition(
                    o['obs'][j],
                    actions_per_env[i][j],
                    float(r_vec[j]),
                    next_obs_chunk[j],
                    bool(d_vec[j] > 0.5),
                )
            next_obs_all.append(next_obs_chunk)
            rewards_all.append(r_vec)
            dones_all.append(d_vec)
            offset += n

        step_reward_mean = total_step_reward / max(1, len(step_outs))
        if avg_reward_ema is None:
            avg_reward_ema = step_reward_mean
        else:
            avg_reward_ema = 0.98 * avg_reward_ema + 0.02 * step_reward_mean

        # Train
        if step % eff_update_freq == 0:
            loss = model.train_step()
            if loss is not None:
                last_loss = float(loss)
        # Metrics per log_every
        if log_metrics and (step % max(1, log_every) == 0):
            try:
                with torch.no_grad():
                    qm = model.q_network(torch.tensor(obs_all, dtype=torch.float32, device=model.device)).detach().cpu().numpy().mean(axis=0)
            except Exception:
                qm = np.zeros((7,), dtype=np.float32)
            steps_log.append(step)
            eps_log.append(float(getattr(model, 'epsilon', 0.0)))
            loss_log.append(float('nan') if last_loss is None else float(last_loss))
            stepR_log.append(float(step_reward_mean))
            avgR_log.append(float(avg_reward_ema))
            q_logs.append(qm.tolist())

        try:
            elapsed = max(1e-6, (_time.time() - _t0))
            env_steps = (step + 1) * used_envs
            sps = env_steps / elapsed
            pbar.set_postfix(envs=used_envs, env_steps=env_steps, sps=f"{sps:.1f}", avgR=f"{avg_reward_ema:.2f}", loss=f"{(last_loss or 0):.3f}")
        except Exception:
            pass

    # Save metrics
    if log_metrics:
        try:
            import os
            import pandas as pd
            import matplotlib.pyplot as plt
            tag = metrics_tag or 'DQN'
            out_dir = metrics_dir or 'results/train_metrics'
            out_dir = os.path.join(out_dir, tag + '_subproc')
            os.makedirs(out_dir, exist_ok=True)
            cols = ['q_up', 'q_down', 'q_left', 'q_right', 'q_idle', 'q_pick', 'q_drop']
            q_arr = np.array(q_logs) if len(q_logs) > 0 else np.zeros((0, 7))
            df = pd.DataFrame({
                'step': steps_log,
                'epsilon': eps_log,
                'loss': loss_log,
                'step_reward': stepR_log,
                'avg_reward_ema': avgR_log,
                'policy_loss': [float('nan')] * len(steps_log),
                'value_loss': [float('nan')] * len(steps_log),
                'entropy_loss': [float('nan')] * len(steps_log),
                'entropy': [float('nan')] * len(steps_log),
            })
            if len(q_logs) > 0:
                for i, c in enumerate(cols):
                    df[c] = q_arr[:, i]
            df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            if len(steps_log) > 0:
                plt.figure(figsize=(6, 3))
                plt.plot(steps_log, eps_log, label='epsilon')
                plt.xlabel('step'); plt.ylabel('epsilon'); plt.title(f'{tag} Epsilon (subproc)')
                plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'epsilon.png')); plt.close()
                plt.figure(figsize=(6, 3))
                plt.plot(steps_log, loss_log, label='loss', alpha=0.8)
                plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{tag} Loss (subproc)')
                plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'loss.png')); plt.close()
                plt.figure(figsize=(6, 3))
                plt.plot(steps_log, stepR_log, label='step_reward', alpha=0.5)
                plt.plot(steps_log, avgR_log, label='avg_reward_ema', alpha=0.9)
                plt.legend(); plt.xlabel('step'); plt.ylabel('reward'); plt.title(f'{tag} Rewards (subproc)')
                plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'rewards.png')); plt.close()
                if len(q_logs) > 0:
                    plt.figure(figsize=(7, 4))
                    for i, c in enumerate(cols):
                        plt.plot(steps_log, q_arr[:, i], label=c)
                    plt.legend(ncol=3, fontsize=8)
                    plt.xlabel('step'); plt.ylabel('Q'); plt.title(f'{tag} Mean Q per action (subproc)')
                    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'q_values.png')); plt.close()
        except Exception as e:
            print(f'[warn] Failed to save DQN (subproc) metrics: {e}')
    try:
        vec.close()
    except Exception:
        pass
    return model


def train_nl_hmarl(
    *,
    env_ctor,
    env_config: dict,
    training_steps: int = 5000,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    max_tasks: int = 20,
    gamma: float = 0.99,
    update_every: int = 8,
    entropy_coef: float = 0.01,
    # NL manager structure
    n_nests: int = 4,
    learn_eta: bool = False,
    eta_init: float = 1.0,
    device: str = 'cpu',
    speed_function=None,
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'NL-HMARL',
    # vectorized envs
    n_envs: int = 1,
):
    _enable_torch_perf()
    """Train NL-HMARL manager with a simple A2C objective; workers use heuristic navigation during training.

    Notes:
    - Global state built via exp.obs.get_global_state
    - Task features via exp.obs.get_task_features (5-dim per task)
    - Nests are task.zone (0..3)
    - Manager reward uses sum of env step rewards (global) per decision step
    - Uses 1-step return: R = r + gamma * V(s') and advantage A = R - V(s)
    """
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL
    from exp.actions import smart_navigate, find_adjacent_accessible_position
    from env.dynamic_warehouse_env import TaskStatus

    # Build training envs
    if speed_function is None:
        def speed_function(e):
            return {p.id: float(getattr(p, 'speed', 1.0)) for p in e.pickers}
    envs = [env_ctor(dict(env_config)) for _ in range(max(1, int(n_envs)))]
    for ev in envs:
        ev.set_speed_function(speed_function)
        ev.reset()

    # Dimensions
    state_dim = int(get_global_state(envs[0]).shape[0])
    worker_obs_dim = 45  # reuse agent obs with include_global=True
    worker_action_dim = 7
    n_agents = envs[0].n_pickers
    n_nests = 4

    model = NLHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,
        n_nests=n_nests,
        worker_obs_dim=worker_obs_dim,
        worker_action_dim=worker_action_dim,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        device=device,
        learn_eta=learn_eta,
        eta_init=eta_init,
    )
    optim = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr)

    # Logging buffers
    steps_log, loss_log, reward_log = [], [], []
    env_steps_log = []
    env_steps_log = []
    pol_log, val_log, entL_log, ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL', ncols=100)
    for step in pbar:
        # === Vectorized pass across envs ===
        decisions_all = []
        returns_all = []
        cur_loss = float('nan'); cur_pl = float('nan'); cur_vl = float('nan'); cur_el = float('nan'); cur_ent = float('nan')
        for env in envs:
            state_vec = get_global_state(env)
            task_feats = get_task_features(env, max_tasks=max_tasks, pending_only=True)
            nest_ids = np.full((max_tasks,), -1, dtype=np.int64)
            mask = np.zeros((max_tasks,), dtype=np.bool_)
            pending_tasks = [t for t in env.task_pool if t.status == TaskStatus.PENDING][:max_tasks]
            for i, t in enumerate(pending_tasks):
                nest_ids[i] = 1 if bool(getattr(t, 'requires_car', False)) else 0
                mask[i] = (t.status == TaskStatus.PENDING)
            free_pids = [i for i, p in enumerate(env.pickers) if p.current_task is None and len(p.carrying_items) == 0]
            local_mask = mask.copy()
            decisions = []
            for pid in free_pids:
                if not local_mask.any():
                    break
                s = torch.tensor(state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)
                tf = torch.tensor(task_feats, dtype=torch.float32, device=model.device).unsqueeze(0)
                nid = torch.tensor(nest_ids, dtype=torch.long, device=model.device).unsqueeze(0)
                m = torch.tensor(local_mask, dtype=torch.bool, device=model.device).unsqueeze(0)
                with torch.no_grad():
                    sel, _ = model.select_tasks(s, tf, nid, m, deterministic=False)
                idx = int(sel.item())
                if not local_mask[idx]:
                    continue
                t_list = [t for t in env.task_pool if t.status == TaskStatus.PENDING][:max_tasks]
                if idx >= len(t_list):
                    continue
                t = t_list[idx]
                if not (t.status == TaskStatus.PENDING):
                    continue
                t.status = TaskStatus.ASSIGNED
                t.assigned_picker = pid
                env.pickers[pid].current_task = t
                local_mask[idx] = False
                decisions.append((s, tf, nid, m, torch.tensor(idx, dtype=torch.long, device=model.device)))
            # Heuristic actions and step
            actions = _try_gpu_nav_actions(env)
            if actions is None:
                actions = {}
                for i, p in enumerate(env.pickers):
                    t = getattr(p, 'current_task', None)
                    if t is None:
                        actions[i] = 4; continue
                    if len(p.carrying_items) == 0:
                        if t.shelf_id is None or t.shelf_id >= len(env.shelves):
                            actions[i] = 4
                        else:
                            sh = env.shelves[t.shelf_id]
                            adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
                            if adj is None or (p.x, p.y) == adj or (abs(p.x - sh['x']) + abs(p.y - sh['y']) == 1):
                                actions[i] = 4
                            else:
                                actions[i] = smart_navigate(p, adj, env)
                    else:
                        if t.station_id is None or t.station_id >= len(env.stations):
                            actions[i] = 4
                        else:
                            st = env.stations[t.station_id]
                            actions[i] = 4 if abs(p.x - st['x']) + abs(p.y - st['y']) == 1 else smart_navigate(p, (st['x'], st['y']), env)
            env_actions = convert_to_dynamic_actions(actions, env, input_space='env')
            _, rewards, _, _ = env.step(env_actions)
            step_reward = float(sum(rewards.values())) if isinstance(rewards, dict) else float(rewards)
            if decisions:
                next_state_vec = get_global_state(env)
                with torch.no_grad():
                    v_next = model.value_net(torch.tensor(next_state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0).item()
                r = torch.full((len(decisions),), step_reward, dtype=torch.float32, device=model.device)
                # Accumulate
                decisions_all.extend(decisions)
                returns_all.append(r + gamma * v_next)
        # Single batched update across envs
        if len(decisions_all) > 0:
            batch_states = torch.cat([d[0] for d in decisions_all], dim=0)
            batch_tf = torch.cat([d[1] for d in decisions_all], dim=0)
            batch_nid = torch.cat([d[2] for d in decisions_all], dim=0)
            batch_mask = torch.cat([d[3] for d in decisions_all], dim=0)
            batch_idx = torch.stack([d[4] for d in decisions_all], dim=0).to(model.device)
            returns = torch.cat(returns_all, dim=0)
            with torch.no_grad():
                v = model.value_net(batch_states).squeeze(-1)
            adv = returns - v
            loss_dict = model.compute_manager_loss(
                batch_states, batch_tf, batch_nid, batch_idx, adv, returns,
                task_mask=batch_mask, entropy_coef=entropy_coef
            )
            optim.zero_grad(); loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            optim.step()
            cur_loss = float(loss_dict['total_loss'].item())
            cur_pl = float(loss_dict['policy_loss'].item())
            cur_vl = float(loss_dict['value_loss'].item())
            cur_el = float(loss_dict['entropy_loss'].item())
            cur_ent = float(loss_dict['entropy'].item())
        else:
            cur_loss = float('nan'); cur_pl = float('nan'); cur_vl = float('nan'); cur_el = float('nan'); cur_ent = float('nan')

        # Logging (buffer only; write CSV/PNG after training for lower overhead)
        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step)
            loss_log.append(cur_loss)
            reward_log.append(step_reward)
            pol_log.append(cur_pl)
            val_log.append(cur_vl)
            entL_log.append(cur_el)
            ent_log.append(cur_ent)
        pbar.set_postfix(rew=f"{step_reward:.2f}", loss=f"{(cur_loss if np.isfinite(cur_loss) else 0):.3f}")

    # Save metrics after training (CSV + one PNG, like DQN)
    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'NL-HMARL')
            os.makedirs(out_dir, exist_ok=True)
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.DataFrame({
                'step': steps_log,
                'loss': loss_log,
                'step_reward': reward_log,
                'policy_loss': pol_log,
                'value_loss': val_log,
                'entropy_loss': entL_log,
                'entropy': ent_log,
            })
            df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            if len(steps_log) > 0:
                plt.figure(figsize=(7, 4))
                plt.plot(steps_log, pol_log, label='policy_loss')
                plt.plot(steps_log, val_log, label='value_loss')
                plt.plot(steps_log, entL_log, label='entropy_loss')
                plt.legend(fontsize=8)
                plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{metrics_tag} Manager Loss Components')
                plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'manager_losses.png'))
                plt.close()
        except Exception as e:
            print(f"[warn] Failed to save NL-HMARL metrics: {e}")
    return model


def train_nl_hmarl_subproc(
    *,
    env_config: dict,
    training_steps: int = 5000,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    max_tasks: int = 20,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    n_nests: int = 4,
    learn_eta: bool = False,
    eta_init: float = 1.0,
    device: str = 'cpu',
    n_envs: int = 4,
    # logging
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'NL-HMARL',
):
    _enable_torch_perf()
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL

    used_envs = int(max(1, n_envs))
    vec = SubprocVecEnv(used_envs, env_config, max_tasks=max_tasks)
    f0 = vec.get_features_tensor(device=device)
    state_dim = int(f0['state'].shape[1])

    # Model
    model = NLHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,
        n_nests=n_nests,
        worker_obs_dim=45,
        worker_action_dim=7,
        n_agents=int(env_config.get('n_pickers', 1)),
        hidden_dim=hidden_dim,
        device=device,
        learn_eta=learn_eta,
        eta_init=eta_init,
    )
    optim = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr)

    # Logs
    steps_log, loss_log, reward_log = [], [], []
    pol_log, val_log, entL_log, ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL (subproc)', ncols=100)
    import time as _time
    _t0 = _time.time()
    for step in pbar:
        feats = vec.get_features_tensor(device=model.device)
        state = feats['state']
        task_feats = feats['task_feats']
        nest_ids = feats['nest_ids']
        task_mask = feats['task_mask']
        free_mask = feats['free_mask']

        B, T = task_mask.shape
        per_env_decisions = [[] for _ in range(B)]
        batch_states, batch_tf, batch_nid, batch_mask, batch_idx = [], [], [], [], []
        free_lists = [torch.nonzero(free_mask[b], as_tuple=False).squeeze(1).tolist() for b in range(B)]
        next_ptr = [0 for _ in range(B)]
        local_mask = task_mask.clone()
        max_free = int(max((len(fl) for fl in free_lists), default=0))
        for _it in range(max_free):
            active_envs = [b for b in range(B) if next_ptr[b] < len(free_lists[b]) and bool(local_mask[b].any().item())]
            if not active_envs:
                break
            s_act = state[active_envs]
            tf_act = task_feats[active_envs]
            nid_act = nest_ids[active_envs]
            m_act = local_mask[active_envs]
            with torch.no_grad():
                sel, _ = model.select_tasks(s_act, tf_act, nid_act, m_act, deterministic=False)
            for j, b in enumerate(active_envs):
                idx = int(sel[j].item())
                if idx < 0 or not bool(m_act[j, idx].item()):
                    continue
                pid = int(free_lists[b][next_ptr[b]])
                per_env_decisions[b].append((pid, idx))
                batch_states.append(s_act[j:j+1])
                batch_tf.append(tf_act[j:j+1])
                batch_nid.append(nid_act[j:j+1])
                batch_mask.append(m_act[j:j+1].clone())
                batch_idx.append(torch.tensor([idx], dtype=torch.long, device=model.device))
                local_mask[b, idx] = False
                next_ptr[b] += 1

        # Env step with tensor collation
        stp = vec.step_with_decisions_tensor(per_env_decisions, device=model.device)
        if len(batch_idx) > 0:
            next_states = stp['next_state']  # [B,S]
            with torch.no_grad():
                v_next_env = model.value_net(next_states).squeeze(-1)  # [B]
            rewards_env = stp['step_reward']  # [B]
            returns_list = []
            env_has = torch.tensor([len(per_env_decisions[b]) for b in range(B)], device=model.device)
            for b in range(B):
                k = int(env_has[b].item())
                if k > 0:
                    ret_b = torch.full((k,), float(rewards_env[b].item()), dtype=torch.float32, device=model.device) + gamma * v_next_env[b]
                    returns_list.append(ret_b)
            returns = torch.cat(returns_list, dim=0) if len(returns_list) > 0 else torch.empty((0,), device=model.device)
            batch_states_t = torch.cat(batch_states, dim=0)
            batch_tf_t = torch.cat(batch_tf, dim=0)
            batch_nid_t = torch.cat(batch_nid, dim=0)
            batch_mask_t = torch.cat(batch_mask, dim=0)
            batch_idx_t = torch.cat(batch_idx, dim=0)
            with torch.no_grad():
                v = model.value_net(batch_states_t).squeeze(-1)
            adv = returns - v
            loss_dict = model.compute_manager_loss(batch_states_t, batch_tf_t, batch_nid_t, batch_idx_t, adv, returns,
                                                   task_mask=batch_mask_t, entropy_coef=entropy_coef)
            optim.zero_grad(); loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            optim.step()
            cur_loss = float(loss_dict['total_loss'].item())
            cur_pl = float(loss_dict['policy_loss'].item())
            cur_vl = float(loss_dict['value_loss'].item())
            cur_el = float(loss_dict['entropy_loss'].item())
            cur_ent = float(loss_dict['entropy'].item())
        else:
            cur_loss = float('nan'); cur_pl = float('nan'); cur_vl = float('nan'); cur_el = float('nan'); cur_ent = float('nan')

        # Logging buffer
        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step)
            loss_log.append(cur_loss)
            reward_log.append(float(stp['step_reward'].mean().item()))
            pol_log.append(cur_pl); val_log.append(cur_vl); entL_log.append(cur_el); ent_log.append(cur_ent)
        try:
            elapsed = max(1e-6, (_time.time() - _t0))
            env_steps = (step + 1) * used_envs
            sps = env_steps / elapsed
            step_rew = float(stp['step_reward'].mean().item())
            pbar.set_postfix(envs=used_envs, env_steps=env_steps, sps=f"{sps:.1f}", rew=f"{step_rew:.2f}", loss=f"{(cur_loss if np.isfinite(cur_loss) else 0):.3f}")
        except Exception:
            pass

    # Save metrics after training
    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'NL-HMARL')
            os.makedirs(out_dir, exist_ok=True)
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.DataFrame({
                'step': steps_log,
                'loss': loss_log,
                'step_reward': reward_log,
                'policy_loss': pol_log,
                'value_loss': val_log,
                'entropy_loss': entL_log,
                'entropy': ent_log,
            })
            df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            if len(steps_log) > 0:
                plt.figure(figsize=(7, 4))
                plt.plot(steps_log, pol_log, label='policy_loss')
                plt.plot(steps_log, val_log, label='value_loss')
                plt.plot(steps_log, entL_log, label='entropy_loss')
                plt.legend(fontsize=8)
                plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{metrics_tag} Manager Loss Components')
                plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'manager_losses.png'))
                plt.close()
        except Exception as e:
            print(f"[warn] Failed to save NL-HMARL (subproc) metrics: {e}")
    # Close vecenv
    try:
        vec.close()
    except Exception:
        pass
    return model


def train_nl_hmarl_ac(
    *,
    env_ctor,
    env_config: dict,
    training_steps: int = 5000,
    hidden_dim: int = 256,
    lr_manager: float = 1e-3,
    lr_workers: float = 1e-3,
    max_tasks: int = 20,
    gamma: float = 0.99,
    entropy_coef_manager: float = 0.01,
    entropy_coef_workers: float = 0.01,
    # NL manager structure
    n_nests: int = 4,
    learn_eta: bool = False,
    eta_init: float = 1.0,
    device: str = 'cpu',
    speed_function=None,
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'NL-HMARL-AC',
    # vectorized envs
    n_envs: int = 1,
):
    _enable_torch_perf()
    """Train NL-HMARL with Actor-Critic workers.

    - Manager: same A2C as train_nl_hmarl
    - Workers: per-step A2C with shared parameters across agents
    """
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL
    from exp.actions import smart_navigate, find_adjacent_accessible_position, convert_to_dynamic_actions
    from exp.obs import get_agent_observation
    from env.dynamic_warehouse_env import TaskStatus

    if speed_function is None:
        def speed_function(e):
            return {p.id: float(getattr(e, 'speed', 1.0)) for p in e.pickers}
    envs = [env_ctor(dict(env_config)) for _ in range(max(1, int(n_envs)))]
    for ev in envs:
        ev.set_speed_function(speed_function)
        ev.reset()

    # Dimensions
    state_dim = int(get_global_state(envs[0]).shape[0])
    worker_obs_dim = 45
    worker_action_dim = 7
    n_agents = envs[0].n_pickers
    n_nests = 4

    model = NLHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,
        n_nests=n_nests,
        worker_obs_dim=worker_obs_dim,
        worker_action_dim=worker_action_dim,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        device=device,
        learn_eta=learn_eta,
        eta_init=eta_init,
    )
    opt_manager = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr_manager)
    opt_workers = torch.optim.Adam(model.workers.parameters(), lr=lr_workers)

    steps_log, m_loss_log, w_loss_log, reward_log = [], [], [], []
    env_steps_log = []
    m_pl_log, m_vl_log, m_entL_log, m_ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL-AC', ncols=100)

    for step in pbar:
        # Accumulators across envs
        decisions_all = []
        returns_all = []
        obs_all = []
        actions_all = []
        rewards_all = []
        dones_all = []
        # Per-env pass
        for env in envs:
            state_vec = get_global_state(env)
            task_feats = get_task_features(env, max_tasks=max_tasks, pending_only=True)
            nest_ids = np.full((max_tasks,), -1, dtype=np.int64)
            mask = np.zeros((max_tasks,), dtype=np.bool_)
            t_list = [t for t in env.task_pool if t.status == TaskStatus.PENDING][:max_tasks]
            for i, t in enumerate(t_list):
                nest_ids[i] = 1 if bool(getattr(t, 'requires_car', False)) else 0
                mask[i] = (t.status == TaskStatus.PENDING)
            free_pids = [i for i, p in enumerate(env.pickers) if p.current_task is None and len(p.carrying_items) == 0]
            local_mask = mask.copy()
            decisions = []
            for pid in free_pids:
                if not local_mask.any():
                    break
                s = torch.tensor(state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)
                tf = torch.tensor(task_feats, dtype=torch.float32, device=model.device).unsqueeze(0)
                nid = torch.tensor(nest_ids, dtype=torch.long, device=model.device).unsqueeze(0)
                m = torch.tensor(local_mask, dtype=torch.bool, device=model.device).unsqueeze(0)
                with torch.no_grad():
                    sel, _ = model.select_tasks(s, tf, nid, m, deterministic=False)
                idx = int(sel.item())
                if not local_mask[idx] or idx >= len(t_list):
                    continue
                t = t_list[idx]
                if t.status != TaskStatus.PENDING:
                    continue
                t.status = TaskStatus.ASSIGNED
                t.assigned_picker = pid
                env.pickers[pid].current_task = t
                local_mask[idx] = False
                decisions.append((s, tf, nid, m, torch.tensor(idx, dtype=torch.long, device=model.device)))
            # Workers act
            obs_batch = [get_agent_observation(env, p, include_global=True) for p in env.pickers]
            obs_tensor = torch.tensor(np.vstack(obs_batch), dtype=torch.float32, device=model.device)
            out = model.workers(obs_tensor)
            with torch.no_grad():
                actions_idx = torch.multinomial(torch.clamp(out['action_probs'], min=1e-8), num_samples=1).squeeze(1)
            actions = {}
            # Try GPU nav to fix invalids in batch
            gpu_actions = _try_gpu_nav_actions(env)
            for i, p in enumerate(env.pickers):
                a = int(actions_idx[i].item())
                if a in (0, 1, 2, 3):
                    dd = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[a]
                    nx, ny = p.x + dd[0], p.y + dd[1]
                    invalid = not (0 <= nx < env.width and 0 <= ny < env.height) or (env.grid[ny, nx] == 2)
                    if invalid and gpu_actions is not None:
                        a = gpu_actions.get(i, 4)
                    elif invalid:
                        t = getattr(p, 'current_task', None)
                        target = None
                        if t is not None:
                            if p.carrying_items and t.station_id is not None and t.station_id < len(env.stations):
                                st = env.stations[t.station_id]; target = (st['x'], st['y'])
                            elif (not p.carrying_items) and t.shelf_id is not None and t.shelf_id < len(env.shelves):
                                sh = env.shelves[t.shelf_id]
                                adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
                                target = adj if adj is not None else (sh['x'], sh['y'])
                        if target is not None:
                            a = smart_navigate(p, target, env)
                        else:
                            a = 4
                actions[i] = a
            env_actions = convert_to_dynamic_actions(actions, env, input_space='env')
            _, rewards, dones, _ = env.step(env_actions)
            # Accumulate buffers for batched updates
            if decisions:
                next_state_vec = get_global_state(env)
                with torch.no_grad():
                    v_next = model.value_net(torch.tensor(next_state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0).item()
                step_rew = float(sum(rewards.values())) if isinstance(rewards, dict) else float(rewards)
                r = torch.full((len(decisions),), step_rew, dtype=torch.float32, device=model.device)
                decisions_all.extend(decisions)
                returns_all.append(r + gamma * v_next)
            obs_all.append(obs_tensor)
            actions_all.append(torch.tensor([actions[i] for i in range(n_agents)], dtype=torch.long, device=model.device))
            # Reward/done vectors for worker A2C
            if isinstance(rewards, dict):
                r_vec = torch.tensor([float(rewards.get(i, 0.0)) for i in range(n_agents)], dtype=torch.float32, device=model.device)
                d_vec = torch.tensor([1.0 if dones.get(i, False) else 0.0 for i in range(n_agents)], dtype=torch.float32, device=model.device)
            else:
                r_avg = float(rewards) / max(1, n_agents)
                r_vec = torch.full((n_agents,), r_avg, dtype=torch.float32, device=model.device)
                d_vec = torch.zeros((n_agents,), dtype=torch.float32, device=model.device)
            rewards_all.append(r_vec)
            dones_all.append(d_vec)
        # Manager batched update
        if len(decisions_all) > 0:
            batch_states = torch.cat([d[0] for d in decisions_all], dim=0)
            batch_tf = torch.cat([d[1] for d in decisions_all], dim=0)
            batch_nid = torch.cat([d[2] for d in decisions_all], dim=0)
            batch_mask = torch.cat([d[3] for d in decisions_all], dim=0)
            batch_idx = torch.stack([d[4] for d in decisions_all], dim=0).to(model.device)
            returns = torch.cat(returns_all, dim=0)
            with torch.no_grad():
                v = model.value_net(batch_states).squeeze(-1)
            adv = returns - v
            m_losses = model.compute_manager_loss(batch_states, batch_tf, batch_nid, batch_idx, adv, returns,
                                                 task_mask=batch_mask, entropy_coef=entropy_coef_manager)
            opt_manager.zero_grad(); m_losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            opt_manager.step()
            cur_m_loss = float(m_losses['total_loss'].item())
            cur_m_pl = float(m_losses['policy_loss'].item())
            cur_m_vl = float(m_losses['value_loss'].item())
            cur_m_el = float(m_losses['entropy_loss'].item())
            cur_m_ent = float(m_losses['entropy'].item())
        else:
            cur_m_loss = float('nan'); cur_m_pl = float('nan'); cur_m_vl = float('nan'); cur_m_el = float('nan'); cur_m_ent = float('nan')
        # Workers batched update
        if len(obs_all) > 0:
            obs_b = torch.cat(obs_all, dim=0)
            actions_b = torch.cat(actions_all, dim=0)
            out2 = model.workers(obs_b)
            log_probs_all = torch.log(torch.clamp(out2['action_probs'], min=1e-8))
            act_logp = log_probs_all.gather(1, actions_b.unsqueeze(1)).squeeze(1)
            # Next values
            next_obs_all = []
            for env in envs:
                next_obs_all.extend([get_agent_observation(env, p, include_global=True) for p in env.pickers])
            next_obs_tensor = torch.tensor(np.vstack(next_obs_all), dtype=torch.float32, device=model.device)
            with torch.no_grad():
                next_vals = model.workers(next_obs_tensor)['value']
            r_vec = torch.cat(rewards_all, dim=0)
            d_vec = torch.cat(dones_all, dim=0)
            returns_w = r_vec + gamma * next_vals * (1.0 - d_vec)
            adv_w = returns_w - out2['value']
            policy_loss = -(adv_w.detach() * act_logp).mean()
            value_loss = torch.nn.functional.mse_loss(out2['value'], returns_w.detach())
            entropy = -(out2['action_probs'] * torch.log(torch.clamp(out2['action_probs'], min=1e-8))).sum(dim=1).mean()
            total_w_loss = policy_loss + value_loss - entropy_coef_workers * entropy
            opt_workers.zero_grad(); total_w_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.workers.parameters(), 1.0)
            opt_workers.step()
            cur_w_loss = float(total_w_loss.item())
        else:
            cur_w_loss = float('nan')

        # 6) Logging (buffer only; write CSV/PNG after training)
        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step)
            m_loss_log.append(cur_m_loss)
            w_loss_log.append(cur_w_loss)
            reward_log.append(step_rew)
            m_pl_log.append(cur_m_pl)
            m_vl_log.append(cur_m_vl)
            m_entL_log.append(cur_m_el)
            m_ent_log.append(cur_m_ent)
        pbar.set_postfix(rew=f"{step_rew:.2f}", mL=f"{0 if not np.isfinite(cur_m_loss) else cur_m_loss:.3f}", wL=f"{cur_w_loss:.3f}")

    # Save metrics after training (CSV + one PNG)
    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'NL-HMARL-AC')
            os.makedirs(out_dir, exist_ok=True)
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.DataFrame({
                'step': steps_log,
                'manager_loss': m_loss_log,
                'worker_loss': w_loss_log,
                'step_reward': reward_log,
                'policy_loss': m_pl_log,
                'value_loss': m_vl_log,
                'entropy_loss': m_entL_log,
                'entropy': m_ent_log,
            })
            df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            if len(steps_log) > 0:
                plt.figure(figsize=(7, 4))
                plt.plot(steps_log, m_pl_log, label='policy_loss')
                plt.plot(steps_log, m_vl_log, label='value_loss')
                plt.plot(steps_log, m_entL_log, label='entropy_loss')
                plt.legend(fontsize=8)
                plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{metrics_tag} Manager Loss Components')
                plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'manager_losses.png'))
                plt.close()
        except Exception as e:
            print(f"[warn] Failed to save NL-HMARL-AC metrics: {e}")
    return model


def train_nl_hmarl_ac_subproc(
    *,
    env_config: dict,
    training_steps: int = 5000,
    hidden_dim: int = 256,
    lr_manager: float = 1e-3,
    lr_workers: float = 1e-3,
    max_tasks: int = 20,
    gamma: float = 0.99,
    entropy_coef_manager: float = 0.01,
    entropy_coef_workers: float = 0.01,
    # NL manager structure
    n_nests: int = 4,
    learn_eta: bool = False,
    eta_init: float = 1.0,
    device: str = 'cpu',
    n_envs: int = 4,
    # logging
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'NL-HMARL-AC',
):
    _enable_torch_perf()
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL

    used_envs = int(max(1, n_envs))
    vec = SubprocVecEnv(used_envs, env_config, max_tasks=max_tasks)
    f0 = vec.get_features()[0]
    # Dimensions
    state_dim = int(np.array(f0['state_vec'], dtype=np.float32).shape[0])
    worker_obs_dim = 45
    worker_action_dim = 7
    n_agents = int(env_config.get('n_pickers', 1))

    model = NLHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,
        n_nests=n_nests,
        worker_obs_dim=worker_obs_dim,
        worker_action_dim=worker_action_dim,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        device=device,
        learn_eta=learn_eta,
        eta_init=eta_init,
    )
    opt_manager = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr_manager)
    opt_workers = torch.optim.Adam(model.workers.parameters(), lr=lr_workers)

    steps_log, m_loss_log, w_loss_log, reward_log = [], [], [], []
    m_pl_log, m_vl_log, m_entL_log, m_ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL-AC (subproc)', ncols=100)
    import time as _time
    _t0 = _time.time()

    for step in pbar:
        feats_list = vec.get_features()
        per_env_decisions = [[] for _ in range(len(feats_list))]
        decisions_all = []
        returns_all = []
        # Build decisions with manager
        for ei, f in enumerate(feats_list):
            state_vec = np.array(f['state_vec'], dtype=np.float32)
            task_feats = np.array(f['task_feats'], dtype=np.float32)
            task_ids = np.array(f['task_ids'], dtype=np.int64)
            requires = np.array(f['requires'], dtype=np.bool_)
            free_pids = list(np.array(f['free_pids'], dtype=np.int64))
            T = int(task_feats.shape[0])
            nest_ids = np.zeros((T,), dtype=np.int64)
            nest_ids[:len(requires)] = requires.astype(np.int64)
            mask = np.zeros((T,), dtype=bool)
            mask[:len(task_ids)] = True
            local_mask = mask.copy()
            for pid in free_pids:
                if not local_mask.any():
                    break
                s = torch.tensor(state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)
                tf = torch.tensor(task_feats, dtype=torch.float32, device=model.device).unsqueeze(0)
                nid = torch.tensor(nest_ids, dtype=torch.long, device=model.device).unsqueeze(0)
                m = torch.tensor(local_mask, dtype=torch.bool, device=model.device).unsqueeze(0)
                with torch.no_grad():
                    sel, _ = model.select_tasks(s, tf, nid, m, deterministic=False)
                idx = int(sel.item())
                if idx < 0 or idx >= len(task_ids) or not local_mask[idx]:
                    continue
                per_env_decisions[ei].append((int(pid), int(task_ids[idx])))
                local_mask[idx] = False
                decisions_all.append((s, tf, nid, m, torch.tensor(idx, dtype=torch.long, device=model.device)))
        # Worker obs and actions
        obs_list = vec.get_worker_obs(include_global=True)
        actions_per_env: List[List[int]] = []
        obs_all = []
        for ei, ob in enumerate(obs_list):
            obs = np.array(ob.get('obs'), dtype=np.float32)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device)
            out = model.workers(obs_tensor)
            with torch.no_grad():
                a_idx = torch.multinomial(torch.clamp(out['action_probs'], min=1e-8), num_samples=1).squeeze(1)
            actions = [int(a_idx[i].item()) for i in range(obs.shape[0])]
            actions_per_env.append(actions)
            obs_all.append(obs_tensor)
        # Step envs with decisions and worker actions
        outs = vec.step_with_decisions_and_actions(per_env_decisions, actions_per_env)

        # Manager returns
        cur_m_loss = float('nan'); cur_m_pl = float('nan'); cur_m_vl = float('nan'); cur_m_el = float('nan'); cur_m_ent = float('nan')
        for ei, out in enumerate(outs):
            if len(per_env_decisions[ei]) == 0:
                continue
            r = float(out.get('step_reward', 0.0))
            nsv = np.array(out.get('next_state_vec'), dtype=np.float32)
            with torch.no_grad():
                v_next = model.value_net(torch.tensor(nsv, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0)
            returns_all.append(torch.full((len(per_env_decisions[ei]),), r, dtype=torch.float32, device=model.device) + gamma * v_next.item())
        if len(decisions_all) > 0:
            batch_states = torch.cat([d[0] for d in decisions_all], dim=0)
            batch_tf = torch.cat([d[1] for d in decisions_all], dim=0)
            batch_nid = torch.cat([d[2] for d in decisions_all], dim=0)
            batch_mask = torch.cat([d[3] for d in decisions_all], dim=0)
            batch_idx = torch.stack([d[4] for d in decisions_all], dim=0).to(model.device)
            returns = torch.cat(returns_all, dim=0)
            with torch.no_grad():
                v = model.value_net(batch_states).squeeze(-1)
            adv = returns - v
            loss_dict = model.compute_manager_loss(batch_states, batch_tf, batch_nid, batch_idx, adv, returns,
                                                   task_mask=batch_mask, entropy_coef=entropy_coef_manager)
            opt_manager.zero_grad(); loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            opt_manager.step()
            cur_m_loss = float(loss_dict['total_loss'].item())
            cur_m_pl = float(loss_dict['policy_loss'].item())
            cur_m_vl = float(loss_dict['value_loss'].item())
            cur_m_el = float(loss_dict['entropy_loss'].item())
            cur_m_ent = float(loss_dict['entropy'].item())

        # Worker A2C update
        actions_all = []
        rewards_all = []
        dones_all = []
        next_obs_all = []
        for ei, out in enumerate(outs):
            actions_all.append(torch.tensor(actions_per_env[ei], dtype=torch.long, device=model.device))
            rewards_all.append(torch.tensor(np.array(out.get('rewards_vec'), dtype=np.float32), dtype=torch.float32, device=model.device))
            dones_all.append(torch.tensor(np.array(out.get('dones_vec'), dtype=np.float32), dtype=torch.float32, device=model.device))
            next_obs_all.append(torch.tensor(np.array(out.get('next_obs'), dtype=np.float32), dtype=torch.float32, device=model.device))
        if len(obs_all) > 0:
            obs_b = torch.cat(obs_all, dim=0)
            actions_b = torch.cat(actions_all, dim=0)
            out2 = model.workers(obs_b)
            log_probs_all = torch.log(torch.clamp(out2['action_probs'], min=1e-8))
            act_logp = log_probs_all.gather(1, actions_b.unsqueeze(1)).squeeze(1)
            next_obs_tensor = torch.cat(next_obs_all, dim=0)
            with torch.no_grad():
                next_vals = model.workers(next_obs_tensor)['value']
            r_vec = torch.cat(rewards_all, dim=0)
            d_vec = torch.cat(dones_all, dim=0)
            returns_w = r_vec + gamma * next_vals * (1.0 - d_vec)
            adv_w = returns_w - out2['value']
            policy_loss = -(adv_w.detach() * act_logp).mean()
            value_loss = torch.nn.functional.mse_loss(out2['value'], returns_w.detach())
            entropy = -(out2['action_probs'] * torch.log(torch.clamp(out2['action_probs'], min=1e-8))).sum(dim=1).mean()
            total_w_loss = policy_loss + value_loss - entropy_coef_workers * entropy
            opt_workers.zero_grad(); total_w_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.workers.parameters(), 1.0)
            opt_workers.step()
            cur_w_loss = float(total_w_loss.item())
        else:
            cur_w_loss = float('nan')

        # Logging state
        try:
            step_rew = float(np.mean([float(o.get('step_reward', 0.0)) for o in outs]))
        except Exception:
            step_rew = 0.0
        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step)
            m_loss_log.append(cur_m_loss)
            w_loss_log.append(cur_w_loss)
            reward_log.append(step_rew)
            m_pl_log.append(cur_m_pl)
            m_vl_log.append(cur_m_vl)
            m_entL_log.append(cur_m_el)
            m_ent_log.append(cur_m_ent)
        try:
            elapsed = max(1e-6, (_time.time() - _t0))
            env_steps = (step + 1) * used_envs
            sps = env_steps / elapsed
            pbar.set_postfix(envs=used_envs, env_steps=env_steps, sps=f"{sps:.1f}", rew=f"{step_rew:.2f}", mL=f"{0 if not np.isfinite(cur_m_loss) else cur_m_loss:.3f}", wL=f"{cur_w_loss:.3f}")
        except Exception:
            pass

    # Save metrics
    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', (metrics_tag or 'NL-HMARL-AC') + '_subproc')
            os.makedirs(out_dir, exist_ok=True)
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.DataFrame({
                'step': steps_log,
                'manager_loss': m_loss_log,
                'worker_loss': w_loss_log,
                'step_reward': reward_log,
                'policy_loss': m_pl_log,
                'value_loss': m_vl_log,
                'entropy_loss': m_entL_log,
                'entropy': m_ent_log,
            })
            df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            if len(steps_log) > 0:
                plt.figure(figsize=(7, 4))
                plt.plot(steps_log, m_pl_log, label='policy_loss')
                plt.plot(steps_log, m_vl_log, label='value_loss')
                plt.plot(steps_log, m_entL_log, label='entropy_loss')
                plt.legend(fontsize=8)
                plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{metrics_tag} Manager Loss Components (subproc)')
                plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'manager_losses.png')); plt.close()
        except Exception as e:
            print(f"[warn] Failed to save NL-HMARL-AC (subproc) metrics: {e}")
    try:
        vec.close()
    except Exception:
        pass
    return model


def train_nl_hmarl_tensorvec(
    *,
    env_config: dict,
    training_steps: int = 5000,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    max_tasks: int = 20,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    n_nests: int = 4,
    learn_eta: bool = False,
    eta_init: float = 1.0,
    device: str = 'cuda',
    n_envs: int = 32,
    # logging
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'NL-HMARL',
):
    from exp.vecenv_tensor import TensorVecEnv
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL

    vec = TensorVecEnv(env_config, max_tasks=max_tasks, n_envs=int(max(1, n_envs)), device=device)
    f = vec.get_features()
    state_dim = int(f['state'].shape[1])

    model = NLHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,
        n_nests=n_nests,
        worker_obs_dim=45,
        worker_action_dim=7,
        n_agents=int(env_config.get('n_pickers', 1)),
        hidden_dim=hidden_dim,
        device=device,
        learn_eta=learn_eta,
        eta_init=eta_init,
    )
    optim = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr)

    steps_log, loss_log, reward_log = [], [], []
    pol_log, val_log, entL_log, ent_log = [], [], [], []
    # Warmup one step to spawn tasks so manager has candidates from step 1
    try:
        B_warm = int(max(1, n_envs)); N_warm = int(env_config.get('n_pickers', 1))
        _ = vec.step_with_decisions_and_actions_tensor([[] for _ in range(B_warm)], [[4] * N_warm for _ in range(B_warm)])
    except Exception:
        pass
    used_envs = int(max(1, n_envs))
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL (tensorvec)', ncols=100)
    for step in pbar:
        feats = vec.get_features()
        state = feats['state'].to(model.device)
        task_feats = feats['task_feats'].to(model.device)
        nest_ids = feats['nest_ids'].to(model.device)
        task_mask = feats['task_mask'].to(model.device)
        free_mask = feats['free_mask'].to(model.device)

        B, T = task_mask.shape
        free_lists = [torch.nonzero(free_mask[b], as_tuple=False).squeeze(1).tolist() for b in range(B)]
        next_ptr = [0 for _ in range(B)]
        per_env_decisions = [[] for _ in range(B)]

        batch_states, batch_tf, batch_nid, batch_mask, batch_idx = [], [], [], [], []
        local_mask = task_mask.clone()
        max_free = int(max((len(fl) for fl in free_lists), default=0))
        for it in range(max_free):
            active_envs = [b for b in range(B) if next_ptr[b] < len(free_lists[b]) and bool(local_mask[b].any().item())]
            if len(active_envs) == 0:
                break
            s_act = state[active_envs]
            tf_act = task_feats[active_envs]
            nid_act = nest_ids[active_envs]
            m_act = local_mask[active_envs]
            with torch.no_grad():
                sel, _ = model.select_tasks(s_act, tf_act, nid_act, m_act, deterministic=False)
            for j, b in enumerate(active_envs):
                idx = int(sel[j].item())
                if idx < 0 or not bool(m_act[j, idx].item()):
                    continue
                pid = int(free_lists[b][next_ptr[b]])
                per_env_decisions[b].append((pid, idx))
                batch_states.append(s_act[j:j+1])
                batch_tf.append(tf_act[j:j+1])
                batch_nid.append(nid_act[j:j+1])
                batch_mask.append(m_act[j:j+1].clone())
                batch_idx.append(torch.tensor([idx], dtype=torch.long, device=model.device))
                local_mask[b, idx] = False
                next_ptr[b] += 1

        stp = vec.step_with_decisions_tensor(per_env_decisions)

        cur_loss = float('nan'); cur_pl = float('nan'); cur_vl = float('nan'); cur_el = float('nan'); cur_ent = float('nan')
        env_has = torch.tensor([len(per_env_decisions[b]) for b in range(B)], device=model.device)
        if len(batch_idx) > 0:
            next_states = stp['next_state'].to(model.device)  # [B,S]
            with torch.no_grad():
                v_next_env = model.value_net(next_states).squeeze(-1)  # [B]
            rewards_env = stp['step_reward'].to(model.device)  # [B]
            returns_list = []
            for b in range(B):
                k = int(env_has[b].item())
                if k > 0:
                    ret_b = torch.full((k,), float(rewards_env[b].item()), dtype=torch.float32, device=model.device) + gamma * v_next_env[b]
                    returns_list.append(ret_b)
            returns = torch.cat(returns_list, dim=0) if len(returns_list) > 0 else torch.empty((0,), device=model.device)

            batch_states_t = torch.cat(batch_states, dim=0)
            batch_tf_t = torch.cat(batch_tf, dim=0)
            batch_nid_t = torch.cat(batch_nid, dim=0)
            batch_mask_t = torch.cat(batch_mask, dim=0)
            batch_idx_t = torch.cat(batch_idx, dim=0)
            with torch.no_grad():
                v = model.value_net(batch_states_t).squeeze(-1)
            adv = returns - v
            loss_dict = model.compute_manager_loss(batch_states_t, batch_tf_t, batch_nid_t, batch_idx_t, adv, returns,
                                                   task_mask=batch_mask_t, entropy_coef=entropy_coef)
            optim.zero_grad(); loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            optim.step()
            cur_loss = float(loss_dict['total_loss'].item())
            cur_pl = float(loss_dict['policy_loss'].item())
            cur_vl = float(loss_dict['value_loss'].item())
            cur_el = float(loss_dict['entropy_loss'].item())
            cur_ent = float(loss_dict['entropy'].item())

        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step)
            loss_log.append(cur_loss)
            step_rew = float(stp['step_reward'].mean().item()) if isinstance(stp, dict) else 0.0
            reward_log.append(step_rew)
            pol_log.append(cur_pl); val_log.append(cur_vl); entL_log.append(cur_el); ent_log.append(cur_ent)
            env_steps_log.append(int((step + 1) * used_envs))
        # Do not coerce NaN to 0 in display; show 'nan' if not finite
        disp_loss = cur_loss if np.isfinite(cur_loss) else float('nan')
        try:
            env_steps = (step + 1) * used_envs
            pbar.set_postfix(envs=used_envs, env_steps=env_steps, rew=f"{step_rew:.2f}", loss=f"{disp_loss:.3f}")
        except Exception:
            pass

    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'NL-HMARL')
            os.makedirs(out_dir, exist_ok=True)
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.DataFrame({
                'step': steps_log,
                'env_steps': env_steps_log,
                'loss': loss_log,
                'step_reward': reward_log,
                'policy_loss': pol_log,
                'value_loss': val_log,
                'entropy_loss': entL_log,
                'entropy': ent_log,
            })
            df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            if len(steps_log) > 0:
                plt.figure(figsize=(7, 4))
                plt.plot(steps_log, pol_log, label='policy_loss')
                plt.plot(steps_log, val_log, label='value_loss')
                plt.plot(steps_log, entL_log, label='entropy_loss')
                plt.legend(fontsize=8)
                plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{metrics_tag} Manager Loss Components (tensorvec)')
                plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'manager_losses.png'))
                plt.close()
        except Exception as e:
            print(f"[warn] Failed to save NL-HMARL (tensorvec) metrics: {e}")
    try:
        vec.close()
    except Exception:
        pass
    return model


# Override: batched-tensor collated subproc trainer (manager+workers)
def train_nl_hmarl_ac_subproc(
    *,
    env_config: dict,
    training_steps: int = 5000,
    hidden_dim: int = 256,
    lr_manager: float = 1e-3,
    lr_workers: float = 1e-3,
    max_tasks: int = 20,
    gamma: float = 0.99,
    entropy_coef_manager: float = 0.01,
    entropy_coef_workers: float = 0.01,
    # NL manager structure
    n_nests: int = 4,
    learn_eta: bool = False,
    eta_init: float = 1.0,
    device: str = 'cpu',
    n_envs: int = 4,
    # logging
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'NL-HMARL-AC',
):
    _enable_torch_perf()
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL

    used_envs = int(max(1, n_envs))
    vec = SubprocVecEnv(used_envs, env_config, max_tasks=max_tasks)
    f0 = vec.get_features_tensor(device=device)
    state_dim = int(f0['state'].shape[1])
    worker_obs_dim = 45
    worker_action_dim = 7
    n_agents = int(env_config.get('n_pickers', 1))

    model = NLHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,
        n_nests=n_nests,
        worker_obs_dim=worker_obs_dim,
        worker_action_dim=worker_action_dim,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        device=device,
        learn_eta=learn_eta,
        eta_init=eta_init,
    )
    opt_manager = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr_manager)
    opt_workers = torch.optim.Adam(model.workers.parameters(), lr=lr_workers)

    steps_log, m_loss_log, w_loss_log, reward_log = [], [], [], []
    m_pl_log, m_vl_log, m_entL_log, m_ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL-AC (subproc)', ncols=100)

    for step in pbar:
        feats = vec.get_features_tensor(device=model.device)
        state = feats['state']
        task_feats = feats['task_feats']
        nest_ids = feats['nest_ids']
        task_mask = feats['task_mask']
        free_mask = feats['free_mask']

        B, N = free_mask.shape
        # Manager batched decisions
        per_env_decisions: List[List[int]] = [[] for _ in range(B)]
        batch_states, batch_tf, batch_nid, batch_mask, batch_idx = [], [], [], [], []
        free_lists = [torch.nonzero(free_mask[b], as_tuple=False).squeeze(1).tolist() for b in range(B)]
        next_ptr = [0 for _ in range(B)]
        local_mask = task_mask.clone()
        max_free = int(max((len(fl) for fl in free_lists), default=0))
        for _it in range(max_free):
            active_envs = [b for b in range(B) if next_ptr[b] < len(free_lists[b]) and bool(local_mask[b].any().item())]
            if not active_envs:
                break
            s_act = state[active_envs]
            tf_act = task_feats[active_envs]
            nid_act = nest_ids[active_envs]
            m_act = local_mask[active_envs]
            with torch.no_grad():
                sel, _ = model.select_tasks(s_act, tf_act, nid_act, m_act, deterministic=False)
            for j, b in enumerate(active_envs):
                idx = int(sel[j].item())
                if idx < 0 or not bool(m_act[j, idx].item()):
                    continue
                pid = int(free_lists[b][next_ptr[b]])
                per_env_decisions[b].append((pid, idx))
                batch_states.append(s_act[j:j+1])
                batch_tf.append(tf_act[j:j+1])
                batch_nid.append(nid_act[j:j+1])
                batch_mask.append(m_act[j:j+1].clone())
                batch_idx.append(torch.tensor([idx], dtype=torch.long, device=model.device))
                local_mask[b, idx] = False
                next_ptr[b] += 1

        # Worker obs/actions batched
        obs = vec.get_worker_obs_tensor(include_global=True, device=model.device)  # [B,N,45]
        obs_b = obs.view(B * N, worker_obs_dim)
        with torch.no_grad():
            w_out = model.workers(obs_b)
            a_idx = torch.multinomial(torch.clamp(w_out['action_probs'], min=1e-8), num_samples=1).squeeze(1)
        a_mat = a_idx.view(B, N)
        actions_per_env = [[int(a_mat[b, i].item()) for i in range(N)] for b in range(B)]

        # Step envs (tensor collation)
        stp = vec.step_with_decisions_and_actions_tensor(per_env_decisions, actions_per_env, device=model.device)

        # Manager update
        cur_m_loss = float('nan'); cur_m_pl = float('nan'); cur_m_vl = float('nan'); cur_m_el = float('nan'); cur_m_ent = float('nan')
        env_has = torch.tensor([len(per_env_decisions[b]) for b in range(B)], device=model.device)
        if len(batch_idx) > 0:
            next_states = stp['next_state']
            with torch.no_grad():
                v_next_env = model.value_net(next_states).squeeze(-1)
            rewards_env = stp['step_reward']
            returns_list = []
            for b in range(B):
                k = int(env_has[b].item())
                if k > 0:
                    ret_b = torch.full((k,), float(rewards_env[b].item()), dtype=torch.float32, device=model.device) + gamma * v_next_env[b]
                    returns_list.append(ret_b)
            returns = torch.cat(returns_list, dim=0) if len(returns_list) > 0 else torch.empty((0,), device=model.device)
            batch_states_t = torch.cat(batch_states, dim=0)
            batch_tf_t = torch.cat(batch_tf, dim=0)
            batch_nid_t = torch.cat(batch_nid, dim=0)
            batch_mask_t = torch.cat(batch_mask, dim=0)
            batch_idx_t = torch.cat(batch_idx, dim=0)
            with torch.no_grad():
                v = model.value_net(batch_states_t).squeeze(-1)
            adv = returns - v
            m_losses = model.compute_manager_loss(batch_states_t, batch_tf_t, batch_nid_t, batch_idx_t, adv, returns,
                                                  task_mask=batch_mask_t, entropy_coef=entropy_coef_manager)
            opt_manager.zero_grad(); m_losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            opt_manager.step()
            cur_m_loss = float(m_losses['total_loss'].item())
            cur_m_pl = float(m_losses['policy_loss'].item())
            cur_m_vl = float(m_losses['value_loss'].item())
            cur_m_el = float(m_losses['entropy_loss'].item())
            cur_m_ent = float(m_losses['entropy'].item())

        # Worker update
        next_obs = vec.get_worker_obs_tensor(include_global=True, device=model.device)
        next_obs_b = next_obs.view(B * N, worker_obs_dim)
        with torch.no_grad():
            next_vals = model.workers(next_obs_b)['value']
        r_vec = stp['rewards_vec'].reshape(B * N)
        d_vec = stp['dones_vec'].reshape(B * N)
        out2 = model.workers(obs_b)
        log_probs_all = torch.log(torch.clamp(out2['action_probs'], min=1e-8))
        act_logp = log_probs_all.gather(1, a_idx.unsqueeze(1)).squeeze(1)
        returns_w = r_vec + gamma * next_vals * (1.0 - d_vec)
        adv_w = returns_w - out2['value']
        policy_loss = -(adv_w.detach() * act_logp).mean()
        value_loss = torch.nn.functional.mse_loss(out2['value'], returns_w.detach())
        entropy = -(out2['action_probs'] * torch.log(torch.clamp(out2['action_probs'], min=1e-8))).sum(dim=1).mean()
        total_w_loss = policy_loss + value_loss - entropy_coef_workers * entropy
        opt_workers.zero_grad(); total_w_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.workers.parameters(), 1.0)
        opt_workers.step()
        cur_w_loss = float(total_w_loss.item())

        # Logging
        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step)
            m_loss_log.append(cur_m_loss)
            w_loss_log.append(cur_w_loss)
            reward_log.append(float(stp['step_reward'].mean().item()))
            m_pl_log.append(cur_m_pl)
            m_vl_log.append(cur_m_vl)
            m_entL_log.append(cur_m_el)
            m_ent_log.append(cur_m_ent)
        disp_m = cur_m_loss if np.isfinite(cur_m_loss) else float('nan')
        try:
            pbar.set_postfix(rew=f"{float(stp['step_reward'].mean().item()):.2f}", mL=f"{disp_m:.3f}", wL=f"{cur_w_loss:.3f}")
        except Exception:
            pass

    # Save metrics
    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', (metrics_tag or 'NL-HMARL-AC') + '_subproc')
            os.makedirs(out_dir, exist_ok=True)
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.DataFrame({
                'step': steps_log,
                'manager_loss': m_loss_log,
                'worker_loss': w_loss_log,
                'step_reward': reward_log,
                'policy_loss': m_pl_log,
                'value_loss': m_vl_log,
                'entropy_loss': m_entL_log,
                'entropy': m_ent_log,
            })
            df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            if len(steps_log) > 0:
                plt.figure(figsize=(7, 4))
                plt.plot(steps_log, m_pl_log, label='policy_loss')
                plt.plot(steps_log, m_vl_log, label='value_loss')
                plt.plot(steps_log, m_entL_log, label='entropy_loss')
                plt.legend(fontsize=8)
                plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{metrics_tag} Manager Loss Components (subproc)')
                plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'manager_losses.png')); plt.close()
        except Exception as e:
            print(f"[warn] Failed to save NL-HMARL-AC (subproc) metrics: {e}")
    try:
        vec.close()
    except Exception:
        pass
    return model


def train_nl_hmarl_ac_tensorvec(
    *,
    env_config: dict,
    training_steps: int = 5000,
    hidden_dim: int = 256,
    lr_manager: float = 1e-3,
    lr_workers: float = 1e-3,
    max_tasks: int = 20,
    gamma: float = 0.99,
    entropy_coef_manager: float = 0.01,
    entropy_coef_workers: float = 0.01,
    n_nests: int = 4,
    learn_eta: bool = False,
    eta_init: float = 1.0,
    device: str = 'cuda',
    n_envs: int = 32,
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'NL-HMARL-AC',
):
    from exp.vecenv_tensor import TensorVecEnv
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL
    from exp.obs import get_agent_observation  # placeholder if needed later

    vec = TensorVecEnv(env_config, max_tasks=max_tasks, n_envs=int(max(1, n_envs)), device=device)
    f0 = vec.get_features()[0]
    state_dim = int(np.array(f0['state_vec'], dtype=np.float32).shape[0])
    worker_obs_dim = 45
    worker_action_dim = 7
    n_agents = int(env_config.get('n_pickers', 1))

    model = NLHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,
        n_nests=n_nests,
        worker_obs_dim=worker_obs_dim,
        worker_action_dim=worker_action_dim,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        device=device,
        learn_eta=learn_eta,
        eta_init=eta_init,
    )
    opt_manager = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr_manager)
    opt_workers = torch.optim.Adam(model.workers.parameters(), lr=lr_workers)

    steps_log, m_loss_log, w_loss_log, reward_log = [], [], [], []
    m_pl_log, m_vl_log, m_entL_log, m_ent_log = [], [], [], []
    # Warmup: spawn tasks before first decision
    try:
        B_warm = int(max(1, n_envs)); N_warm = int(env_config.get('n_pickers', 1))
        _ = vec.step_with_decisions_and_actions_tensor([[] for _ in range(B_warm)], [[4] * N_warm for _ in range(B_warm)])
    except Exception:
        pass
    used_envs = int(max(1, n_envs))
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL-AC (tensorvec)', ncols=100)
    for step in pbar:
        feats_list = vec.get_features()
        per_env_decisions = [[] for _ in range(len(feats_list))]
        decisions_all = []
        returns_all = []
        for ei, f in enumerate(feats_list):
            state_vec = np.array(f['state_vec'], dtype=np.float32)
            task_feats = np.array(f['task_feats'], dtype=np.float32)
            task_ids = np.array(f['task_ids'], dtype=np.int64)
            requires = np.array(f['requires'], dtype=np.bool_)
            free_pids = list(np.array(f['free_pids'], dtype=np.int64))
            T = int(task_feats.shape[0])
            nest_ids = np.zeros((T,), dtype=np.int64)
            nest_ids[:len(requires)] = requires.astype(np.int64)
            mask = np.zeros((T,), dtype=bool)
            mask[:len(task_ids)] = True
            local_mask = mask.copy()
            for pid in free_pids:
                if not local_mask.any():
                    break
                s = torch.tensor(state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)
                tf = torch.tensor(task_feats, dtype=torch.float32, device=model.device).unsqueeze(0)
                nid = torch.tensor(nest_ids, dtype=torch.long, device=model.device).unsqueeze(0)
                m = torch.tensor(local_mask, dtype=torch.bool, device=model.device).unsqueeze(0)
                with torch.no_grad():
                    sel, _ = model.select_tasks(s, tf, nid, m, deterministic=False)
                idx = int(sel.item())
                if idx < 0 or idx >= len(task_ids) or not local_mask[idx]:
                    continue
                per_env_decisions[ei].append((int(pid), int(task_ids[idx])))
                local_mask[idx] = False
                decisions_all.append((s, tf, nid, m, torch.tensor(idx, dtype=torch.long, device=model.device)))
        # Workers: build obs and sample actions
        obs_list = vec.get_worker_obs(include_global=True)
        actions_per_env: List[List[int]] = []
        obs_all = []
        for ei, ob in enumerate(obs_list):
            obs = np.array(ob.get('obs'), dtype=np.float32)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device)
            out = model.workers(obs_tensor)
            with torch.no_grad():
                a_idx = torch.multinomial(torch.clamp(out['action_probs'], min=1e-8), num_samples=1).squeeze(1)
            actions = [int(a_idx[i].item()) for i in range(obs.shape[0])]
            actions_per_env.append(actions)
            obs_all.append(obs_tensor)

        outs = vec.step_with_decisions_and_actions(per_env_decisions, actions_per_env)
        # Manager update
        cur_m_loss = float('nan'); cur_m_pl = float('nan'); cur_m_vl = float('nan'); cur_m_el = float('nan'); cur_m_ent = float('nan')
        for ei, out in enumerate(outs):
            if len(per_env_decisions[ei]) == 0:
                continue
            r = float(out.get('step_reward', 0.0))
            nsv = np.array(out.get('next_state_vec'), dtype=np.float32)
            with torch.no_grad():
                v_next = model.value_net(torch.tensor(nsv, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0)
            returns_all.append(torch.full((len(per_env_decisions[ei]),), r, dtype=torch.float32, device=model.device) + gamma * v_next.item())
        if len(decisions_all) > 0:
            batch_states = torch.cat([d[0] for d in decisions_all], dim=0)
            batch_tf = torch.cat([d[1] for d in decisions_all], dim=0)
            batch_nid = torch.cat([d[2] for d in decisions_all], dim=0)
            batch_mask = torch.cat([d[3] for d in decisions_all], dim=0)
            batch_idx = torch.stack([d[4] for d in decisions_all], dim=0).to(model.device)
            returns = torch.cat(returns_all, dim=0)
            with torch.no_grad():
                v = model.value_net(batch_states).squeeze(-1)
            adv = returns - v
            m_losses = model.compute_manager_loss(batch_states, batch_tf, batch_nid, batch_idx, adv, returns,
                                                  task_mask=batch_mask, entropy_coef=entropy_coef_manager)
            opt_manager.zero_grad(); m_losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            opt_manager.step()
            cur_m_loss = float(m_losses['total_loss'].item())
            cur_m_pl = float(m_losses['policy_loss'].item())
            cur_m_vl = float(m_losses['value_loss'].item())
            cur_m_el = float(m_losses['entropy_loss'].item())
            cur_m_ent = float(m_losses['entropy'].item())
        # Worker update: A2C one-step with per-agent rewards from tensor vec
        # Collect next observations
        next_obs_list = vec.get_worker_obs(include_global=True)
        obs_b_list = []
        next_obs_b_list = []
        actions_b_list = []
        rewards_b_list = []
        dones_b_list = []
        for ei, ob in enumerate(obs_list):
            obs = np.array(ob.get('obs'), dtype=np.float32)
            obs_b_list.append(torch.tensor(obs, dtype=torch.float32, device=model.device))
            nob = np.array(next_obs_list[ei].get('obs'), dtype=np.float32)
            next_obs_b_list.append(torch.tensor(nob, dtype=torch.float32, device=model.device))
            actions_b_list.append(torch.tensor(actions_per_env[ei], dtype=torch.long, device=model.device))
            r_vec = np.array(outs[ei].get('rewards_vec'), dtype=np.float32)
            d_vec = np.array(outs[ei].get('dones_vec'), dtype=np.float32)
            rewards_b_list.append(torch.tensor(r_vec, dtype=torch.float32, device=model.device))
            dones_b_list.append(torch.tensor(d_vec, dtype=torch.float32, device=model.device))
        if len(obs_b_list) > 0:
            obs_b = torch.cat(obs_b_list, dim=0)
            actions_b = torch.cat(actions_b_list, dim=0)
            with torch.no_grad():
                out2 = model.workers(obs_b)
            # Log probs from fresh forward for stability
            out2 = model.workers(obs_b)
            log_probs_all = torch.log(torch.clamp(out2['action_probs'], min=1e-8))
            act_logp = log_probs_all.gather(1, actions_b.unsqueeze(1)).squeeze(1)
            next_obs_b = torch.cat(next_obs_b_list, dim=0)
            with torch.no_grad():
                next_vals = model.workers(next_obs_b)['value']
            r_vec = torch.cat(rewards_b_list, dim=0)
            d_vec = torch.cat(dones_b_list, dim=0)
            returns_w = r_vec + gamma * next_vals * (1.0 - d_vec)
            adv_w = returns_w - out2['value']
            policy_loss = -(adv_w.detach() * act_logp).mean()
            value_loss = torch.nn.functional.mse_loss(out2['value'], returns_w.detach())
            entropy = -(out2['action_probs'] * torch.log(torch.clamp(out2['action_probs'], min=1e-8))).sum(dim=1).mean()
            total_w_loss = policy_loss + value_loss - entropy_coef_workers * entropy
            opt_workers.zero_grad(); total_w_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.workers.parameters(), 1.0)
            opt_workers.step()
            cur_w_loss = float(total_w_loss.item())
        else:
            cur_w_loss = float('nan')

        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step)
            m_loss_log.append(cur_m_loss)
            w_loss_log.append(cur_w_loss)
            try:
                mean_rew = float(np.mean([float(o.get('step_reward', 0.0)) for o in outs]))
            except Exception:
                mean_rew = 0.0
            reward_log.append(mean_rew)
            m_pl_log.append(cur_m_pl)
            m_vl_log.append(cur_m_vl)
            m_entL_log.append(cur_m_el)
            m_ent_log.append(cur_m_ent)
        disp_m = cur_m_loss if np.isfinite(cur_m_loss) else float('nan')
        try:
            pbar.set_postfix(rew=f"{mean_rew:.2f}", mL=f"{disp_m:.3f}")
        except Exception:
            pass

    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', (metrics_tag or 'NL-HMARL-AC') + '_tensorvec')
            os.makedirs(out_dir, exist_ok=True)
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.DataFrame({
                'step': steps_log,
                'manager_loss': m_loss_log,
                'worker_loss': w_loss_log,
                'step_reward': reward_log,
                'policy_loss': m_pl_log,
                'value_loss': m_vl_log,
                'entropy_loss': m_entL_log,
                'entropy': m_ent_log,
            })
            df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            if len(steps_log) > 0:
                plt.figure(figsize=(7, 4))
                plt.plot(steps_log, m_pl_log, label='policy_loss')
                plt.plot(steps_log, m_vl_log, label='value_loss')
                plt.plot(steps_log, m_entL_log, label='entropy_loss')
                plt.legend(fontsize=8)
                plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{metrics_tag} Manager Loss Components (tensorvec)')
                plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'manager_losses.png'))
                plt.close()
        except Exception as e:
            print(f"[warn] Failed to save NL-HMARL-AC (tensorvec) metrics: {e}")
    try:
        vec.close()
    except Exception:
        pass
    return model


# New batched-tensor version overriding the earlier definition above.
def train_nl_hmarl_ac_tensorvec(
    *,
    env_config: dict,
    training_steps: int = 5000,
    hidden_dim: int = 256,
    lr_manager: float = 1e-3,
    lr_workers: float = 1e-3,
    max_tasks: int = 20,
    gamma: float = 0.99,
    entropy_coef_manager: float = 0.01,
    entropy_coef_workers: float = 0.01,
    n_nests: int = 4,
    learn_eta: bool = False,
    eta_init: float = 1.0,
    device: str = 'cuda',
    n_envs: int = 32,
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'NL-HMARL-AC',
):
    from exp.vecenv_tensor import TensorVecEnv
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL

    vec = TensorVecEnv(env_config, max_tasks=max_tasks, n_envs=int(max(1, n_envs)), device=device)
    f = vec.get_features()
    state_dim = int(f['state'].shape[1])
    worker_obs_dim = 45
    worker_action_dim = 7
    n_agents = int(env_config.get('n_pickers', 1))

    model = NLHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,
        n_nests=n_nests,
        worker_obs_dim=worker_obs_dim,
        worker_action_dim=worker_action_dim,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        device=device,
        learn_eta=learn_eta,
        eta_init=eta_init,
    )
    opt_manager = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr_manager)
    opt_workers = torch.optim.Adam(model.workers.parameters(), lr=lr_workers)

    steps_log, m_loss_log, w_loss_log, reward_log = [], [], [], []
    m_pl_log, m_vl_log, m_entL_log, m_ent_log = [], [], [], []
    env_steps_log = []
    used_envs = int(max(1, n_envs))
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL-AC (tensorvec)', ncols=100)
    for step in pbar:
        feats = vec.get_features()
        state = feats['state'].to(model.device)
        task_feats = feats['task_feats'].to(model.device)
        nest_ids = feats['nest_ids'].to(model.device)
        task_mask = feats['task_mask'].to(model.device)
        free_mask = feats['free_mask'].to(model.device)

        B, N = free_mask.shape
        T = task_mask.shape[1]
        free_lists = [torch.nonzero(free_mask[b], as_tuple=False).squeeze(1).tolist() for b in range(B)]
        next_ptr = [0 for _ in range(B)]
        per_env_decisions = [[] for _ in range(B)]
        batch_states, batch_tf, batch_nid, batch_mask, batch_idx = [], [], [], [], []
        local_mask = task_mask.clone()
        max_free = int(max((len(fl) for fl in free_lists), default=0))
        for it in range(max_free):
            active_envs = [b for b in range(B) if next_ptr[b] < len(free_lists[b]) and bool(local_mask[b].any().item())]
            if len(active_envs) == 0:
                break
            s_act = state[active_envs]
            tf_act = task_feats[active_envs]
            nid_act = nest_ids[active_envs]
            m_act = local_mask[active_envs]
            with torch.no_grad():
                sel, _ = model.select_tasks(s_act, tf_act, nid_act, m_act, deterministic=False)
            for j, b in enumerate(active_envs):
                idx = int(sel[j].item())
                if idx < 0 or not bool(m_act[j, idx].item()):
                    continue
                pid = int(free_lists[b][next_ptr[b]])
                per_env_decisions[b].append((pid, idx))
                batch_states.append(s_act[j:j+1])
                batch_tf.append(tf_act[j:j+1])
                batch_nid.append(nid_act[j:j+1])
                batch_mask.append(m_act[j:j+1].clone())
                batch_idx.append(torch.tensor([idx], dtype=torch.long, device=model.device))
                local_mask[b, idx] = False
                next_ptr[b] += 1

        # Worker obs -> actions for all B*N
        obs = vec.get_worker_obs(include_global=True).to(model.device)  # [B,N,45]
        obs_b = obs.view(B * N, worker_obs_dim)
        with torch.no_grad():
            out_workers = model.workers(obs_b)
            a_idx = torch.multinomial(torch.clamp(out_workers['action_probs'], min=1e-8), num_samples=1).squeeze(1)
        a_mat = a_idx.view(B, N)
        actions_per_env = [[int(a_mat[b, i].item()) for i in range(N)] for b in range(B)]

        # Step envs (tensor outputs)
        stp = vec.step_with_decisions_and_actions_tensor(per_env_decisions, actions_per_env)

        # Manager update
        cur_m_loss = float('nan'); cur_m_pl = float('nan'); cur_m_vl = float('nan'); cur_m_el = float('nan'); cur_m_ent = float('nan')
        env_has = torch.tensor([len(per_env_decisions[b]) for b in range(B)], device=model.device)
        if len(batch_idx) > 0:
            next_states = stp['next_state'].to(model.device)  # [B,S]
            with torch.no_grad():
                v_next_env = model.value_net(next_states).squeeze(-1)  # [B]
            rewards_env = stp['step_reward'].to(model.device)  # [B]
            returns_list = []
            for b in range(B):
                k = int(env_has[b].item())
                if k > 0:
                    ret_b = torch.full((k,), float(rewards_env[b].item()), dtype=torch.float32, device=model.device) + gamma * v_next_env[b]
                    returns_list.append(ret_b)
            returns = torch.cat(returns_list, dim=0) if len(returns_list) > 0 else torch.empty((0,), device=model.device)

            batch_states_t = torch.cat(batch_states, dim=0)
            batch_tf_t = torch.cat(batch_tf, dim=0)
            batch_nid_t = torch.cat(batch_nid, dim=0)
            batch_mask_t = torch.cat(batch_mask, dim=0)
            batch_idx_t = torch.cat(batch_idx, dim=0)
            with torch.no_grad():
                v = model.value_net(batch_states_t).squeeze(-1)
            adv = returns - v
            m_losses = model.compute_manager_loss(batch_states_t, batch_tf_t, batch_nid_t, batch_idx_t, adv, returns,
                                                  task_mask=batch_mask_t, entropy_coef=entropy_coef_manager)
            opt_manager.zero_grad(); m_losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            opt_manager.step()
            cur_m_loss = float(m_losses['total_loss'].item())
            cur_m_pl = float(m_losses['policy_loss'].item())
            cur_m_vl = float(m_losses['value_loss'].item())
            cur_m_el = float(m_losses['entropy_loss'].item())
            cur_m_ent = float(m_losses['entropy'].item())

        # Worker update
        next_obs = vec.get_worker_obs(include_global=True).to(model.device)
        next_obs_b = next_obs.view(B * N, worker_obs_dim)
        with torch.no_grad():
            next_vals = model.workers(next_obs_b)['value']
        r_vec = stp['rewards_vec'].to(model.device).reshape(B * N)
        d_vec = stp['dones_vec'].to(model.device).reshape(B * N)
        out2 = model.workers(obs_b)
        log_probs_all = torch.log(torch.clamp(out2['action_probs'], min=1e-8))
        act_logp = log_probs_all.gather(1, a_idx.unsqueeze(1)).squeeze(1)
        returns_w = r_vec + gamma * next_vals * (1.0 - d_vec)
        adv_w = returns_w - out2['value']
        policy_loss = -(adv_w.detach() * act_logp).mean()
        value_loss = torch.nn.functional.mse_loss(out2['value'], returns_w.detach())
        entropy = -(out2['action_probs'] * torch.log(torch.clamp(out2['action_probs'], min=1e-8))).sum(dim=1).mean()
        total_w_loss = policy_loss + value_loss - entropy_coef_workers * entropy
        opt_workers.zero_grad(); total_w_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.workers.parameters(), 1.0)
        opt_workers.step()
        cur_w_loss = float(total_w_loss.item())

        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step)
            m_loss_log.append(cur_m_loss)
            w_loss_log.append(cur_w_loss)
            mean_rew = float(stp['step_reward'].mean().item())
            reward_log.append(mean_rew)
            m_pl_log.append(cur_m_pl)
            m_vl_log.append(cur_m_vl)
            m_entL_log.append(cur_m_el)
            m_ent_log.append(cur_m_ent)
            env_steps_log.append(int((step + 1) * used_envs))
        try:
            env_steps = (step + 1) * used_envs
            pbar.set_postfix(envs=used_envs, env_steps=env_steps, rew=f"{mean_rew:.2f}", mL=f"{0 if not np.isfinite(cur_m_loss) else cur_m_loss:.3f}", wL=f"{cur_w_loss:.3f}")
        except Exception:
            pass

    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', (metrics_tag or 'NL-HMARL-AC') + '_tensorvec')
            os.makedirs(out_dir, exist_ok=True)
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.DataFrame({
                'step': steps_log,
                'env_steps': env_steps_log,
                'manager_loss': m_loss_log,
                'worker_loss': w_loss_log,
                'step_reward': reward_log,
                'policy_loss': m_pl_log,
                'value_loss': m_vl_log,
                'entropy_loss': m_entL_log,
                'entropy': m_ent_log,
            })
            df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            if len(steps_log) > 0:
                plt.figure(figsize=(7, 4))
                plt.plot(steps_log, m_pl_log, label='policy_loss')
                plt.plot(steps_log, m_vl_log, label='value_loss')
                plt.plot(steps_log, m_entL_log, label='entropy_loss')
                plt.legend(fontsize=8)
                plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'{metrics_tag} Manager Loss Components (tensorvec)')
                plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'manager_losses.png'))
                plt.close()
        except Exception as e:
            print(f"[warn] Failed to save NL-HMARL-AC (tensorvec) metrics: {e}")
    try:
        vec.close()
    except Exception:
        pass
    return model
