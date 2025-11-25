from typing import Dict, Optional, Callable
import numpy as np
import torch
from tqdm import tqdm

from baselines.flat_marl_dqn import FlatMARLDQN
from exp.obs import get_agent_observation
from exp.actions import get_valid_actions, convert_to_dynamic_actions
from exp.assigners import assign_tasks_dynamic
from exp.obs import get_global_state, get_task_features

# Soft deadline filtering hyper-parameters (hours)
SOFT_DEADLINE_RATIO = 1.15
SOFT_DEADLINE_MIN_SLACK = 0.05


def _apply_soft_deadline_filter(env, picker, task_list, mask,
                                ratio: float = SOFT_DEADLINE_RATIO,
                                min_slack: float = SOFT_DEADLINE_MIN_SLACK):
    """Filter tasks whose remaining slack is insufficient for the picker."""
    if mask is None or not mask.any():
        return mask


def _compute_high_value_risk(env,
                             task,
                             slack_threshold: float = 0.2,
                             value_threshold: float = 80.0,
                             risk_coef: float = 0.5) -> float:
    """估算高价值任务被忽视时的潜在损失（越紧迫越大）。"""
    try:
        slack = float(getattr(task, 'deadline', float('inf')) - env.current_time)
    except Exception:
        slack = float('inf')
    if not np.isfinite(slack):
        slack = float('inf')
    try:
        dec_val = float(env.get_task_decayed_value(task, at_time=env.current_time))
    except Exception:
        dec_val = float(getattr(task, 'base_value', 0.0))
    if dec_val < value_threshold or slack >= slack_threshold:
        return 0.0
    urgency = 1.0 - max(0.0, slack) / max(slack_threshold, 1e-6)
    reward_cfg = getattr(env, 'reward_config', {}) or {}
    late_penalty = abs(float(reward_cfg.get('late_penalty', 0.0))) if isinstance(reward_cfg, dict) else 0.0
    return risk_coef * (dec_val + late_penalty) * urgency
    if picker is None:
        return mask
    eta_fn = getattr(env, 'estimate_completion_time', None)
    if eta_fn is None:
        return mask
    filtered = mask.copy()
    pruned = False
    try:
        for idx, task in enumerate(task_list):
            if idx >= len(filtered):
                break
            if not filtered[idx]:
                continue
            slack = float(getattr(task, 'deadline', float('inf')) - env.current_time)
            if slack <= 0.0:
                filtered[idx] = False
                pruned = True
                continue
            eta = float(eta_fn(picker, task))
            if eta <= 0.0 or not np.isfinite(eta):
                continue
            threshold = max(min_slack, eta * ratio)
            if slack < threshold:
                filtered[idx] = False
                pruned = True
        if filtered.any():
            return filtered
        return mask
    except Exception:
        return mask


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
    pbar = tqdm(range(training_steps), desc='Train DQN', ncols=100, disable=True)
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
            # 仅在 loss 非零/有效时输出一行进度
            try:
                if last_loss is not None and np.isfinite(last_loss) and abs(float(last_loss)) > 0.0:
                    pct = (step + 1) * 100.0 / max(1, int(training_steps))
                    from tqdm import tqdm as _tqdm
                    _tqdm.write(f"step {step+1}/{training_steps} ({pct:.2f}%) loss={last_loss:.3f} avgR={avg_reward_ema:.2f}")
            except Exception:
                pass

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
    # Deprecated: multi-env training removed. Use train_flat_dqn instead.
    raise RuntimeError('Subproc training removed; use train_flat_dqn (single-env)')
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
    vec = SubprocVecEnv(int(max(2, n_envs)), env_config, max_tasks=20)

    obs_dim = 40 if pure_learning else 45
    action_dim = 7
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
    used_envs = int(max(2, n_envs))
    total_agents = used_envs * n_pickers
    import time as _time
    _t0 = _time.time()
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
            q_vals = model.q_network(obs_tensor).detach().cpu().numpy()  # (E*N,7)
        q_vals[masks_all == 0] = -np.inf
        # Epsilon per-current step across all agents
        eps_start = getattr(model, 'epsilon_start', 1.0)
        eps_end = getattr(model, 'epsilon_end', 0.05)
        eps_decay = max(1, int(getattr(model, 'epsilon_decay', 100000)))
        cur_eps = eps_end + (eps_start - eps_end) * np.exp(-float(model.steps_done) / float(eps_decay))
        explore = (np.random.rand(obs_all.shape[0]) < cur_eps)
        chosen = np.full((obs_all.shape[0],), 4, dtype=np.int64)
        for i in range(obs_all.shape[0]):
            if explore[i]:
                valid_idx = np.where(masks_all[i] == 1)[0]
                if len(valid_idx) == 0:
                    chosen[i] = int(np.argmax(q_vals[i])) if np.all(np.isfinite(q_vals[i])) else 4
                else:
                    chosen[i] = int(np.random.choice(valid_idx))
            else:
                chosen[i] = int(np.nanargmax(q_vals[i])) if np.any(np.isfinite(q_vals[i])) else 4
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
        if step % update_freq == 0:
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
    n_nests: int = 8,
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
    """Train NL-HMARL manager with a simple A2C objective; workers use heuristic navigation during training.

    Notes:
    - Global state built via exp.obs.get_global_state
    - Task features via exp.obs.get_task_features (5-dim per task)
    - Nests are task.zone * 2 + urgency (0..7)
    - Manager reward uses sum of env step rewards (global) per decision step
    - Uses 1-step return: R = r + gamma * V(s') and advantage A = R - V(s)
    """
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL
    from exp.actions import smart_navigate, find_adjacent_accessible_position
    from env.dynamic_warehouse_env import TaskStatus, PickerType

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
    task_feat_dim = int(get_task_features(envs[0], max_tasks=max_tasks, pending_only=True).shape[1])
    worker_obs_dim = 45  # reuse agent obs with include_global=True
    worker_action_dim = 7
    n_agents = envs[0].n_pickers
    n_nests = 8

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
        task_feature_dim=task_feat_dim,
    )
    optim = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr)

    # Logging buffers
    steps_log, loss_log, reward_log = [], [], []
    pol_log, val_log, entL_log, ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL', ncols=100, disable=True)
    value_coef = 2.0
    penalty_coef = 1.0
    risk_value_threshold = 80.0
    risk_slack_threshold = 0.2  # 小于约12分钟视为高风险
    risk_coef = 0.5
    for step in pbar:
        # === Vectorized pass across envs ===
        decisions_all = []
        returns_all = []
        # 累计本轮各环境的 manager 差分回报（用于日志/心跳）
        manager_r_acc = 0.0
        cur_loss = float('nan'); cur_pl = float('nan'); cur_vl = float('nan'); cur_el = float('nan'); cur_ent = float('nan')
        for env in envs:
            state_vec = get_global_state(env)
            task_feats = get_task_features(env, max_tasks=max_tasks, pending_only=True)
            nest_ids = np.full((max_tasks,), -1, dtype=np.int64)
            mask = np.zeros((max_tasks,), dtype=np.bool_)
            pending_tasks = [t for t in env.task_pool if t.status == TaskStatus.PENDING][:max_tasks]
            for i, t in enumerate(pending_tasks):
                # 以 Zone * 2 + Urgency 作为巢标识 (8 nests)
                try:
                    nid = int(getattr(t, 'zone', 0))
                except Exception:
                    nid = 0
                nid = max(0, min(3, nid))

                is_urgent = 0
                try:
                    # Urgency check: priority > 0.7 or close to deadline (< 0.15h approx 9 min)
                    rem = t.deadline - env.current_time
                    if t.priority > 0.7 or rem < 0.15:
                        is_urgent = 1
                except Exception:
                    pass

                nest_ids[i] = nid * 2 + is_urgent
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
                # Capability-aware task mask per picker
                comp_mask = local_mask.copy()
                picker = None
                try:
                    picker = env.pickers[pid]
                except Exception:
                    picker = None
                if picker is not None:
                    for ii, tt in enumerate(pending_tasks):
                        if comp_mask[ii] and bool(getattr(tt, 'requires_car', False)) and picker.type != PickerType.FORKLIFT:
                            comp_mask[ii] = False
                    filtered_mask = _apply_soft_deadline_filter(env, picker, pending_tasks, comp_mask)
                    if filtered_mask is not None:
                        comp_mask = filtered_mask
                if not comp_mask.any():
                    continue
                m = torch.tensor(comp_mask, dtype=torch.bool, device=model.device).unsqueeze(0)
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
            # Manager 差分回报：仅关注"完成价值增量 − 迟到罚没增量"以降低噪声
            prev_val = float(getattr(env, 'total_value_completed', 0.0))
            prev_pen = float(getattr(env, 'total_value_penalty', 0.0))

            # 记录执行前每个picker到目标的距离(用于进度奖励)
            prev_distances = {}
            for i, p in enumerate(env.pickers):
                t = getattr(p, 'current_task', None)
                if t is not None:
                    if len(p.carrying_items) == 0 and t.shelf_id is not None and t.shelf_id < len(env.shelves):
                        # 前往货架
                        sh = env.shelves[t.shelf_id]
                        prev_distances[p.id] = abs(p.x - sh['x']) + abs(p.y - sh['y'])
                    elif len(p.carrying_items) > 0 and t.station_id is not None and t.station_id < len(env.stations):
                        # 前往站点
                        st = env.stations[t.station_id]
                        prev_distances[p.id] = abs(p.x - st['x']) + abs(p.y - st['y'])

            env_actions = convert_to_dynamic_actions(actions, env, input_space='env')
            _, _, _, _ = env.step(env_actions)
            val_now = float(getattr(env, 'total_value_completed', 0.0))
            pen_now = float(getattr(env, 'total_value_penalty', 0.0))

            # 计算进度奖励：接近目标获得正奖励
            progress_reward = 0.0
            for i, p in enumerate(env.pickers):
                if p.id in prev_distances:
                    t = getattr(p, 'current_task', None)
                    if t is not None:
                        curr_dist = 0
                        if len(p.carrying_items) == 0 and t.shelf_id is not None and t.shelf_id < len(env.shelves):
                            sh = env.shelves[t.shelf_id]
                            curr_dist = abs(p.x - sh['x']) + abs(p.y - sh['y'])
                        elif len(p.carrying_items) > 0 and t.station_id is not None and t.station_id < len(env.stations):
                            st = env.stations[t.station_id]
                            curr_dist = abs(p.x - st['x']) + abs(p.y - st['y'])

                        # 距离缩短 = 正奖励, 距离增加 = 负奖励
                        distance_delta = prev_distances[p.id] - curr_dist
                        progress_reward += distance_delta * 0.5  # 每缩短1格距离 = +0.5奖励

            # 忽视高价值任务的风险惩罚：衡量仍未分配的【贵且快到期】任务
            risk_penalty = 0.0
            if pending_tasks:
                for idx, keep in enumerate(local_mask):
                    if not keep or idx >= len(pending_tasks):
                        continue
                    risk_penalty += _compute_high_value_risk(
                        env,
                        pending_tasks[idx],
                        slack_threshold=risk_slack_threshold,
                        value_threshold=risk_value_threshold,
                        risk_coef=risk_coef
                    )
            
            # 拥堵惩罚：计算每个Zone的负载 (Zone 0-3)
            zone_loads = [0] * 4
            for p in env.pickers:
                # 将picker位置映射到Zone
                zx = int(p.x / (env.width / 2))
                zy = int(p.y / (env.height / 2))
                z_idx = zy * 2 + zx
                if 0 <= z_idx < 4:
                    zone_loads[z_idx] += 1
            
            # 差分回报并做幅度裁剪，稳定训练
            value_gain = (val_now - prev_val)
            penalty_gain = (pen_now - prev_pen)
            manager_r = value_coef * value_gain - penalty_coef * penalty_gain - risk_penalty + progress_reward
            
            # 为每个决策施加特定Zone的拥堵惩罚
            # 如果分配的任务在拥堵区，给一个额外负反馈
            congestion_penalty = 0.0
            for d_idx, (_, _, nid_tensor, _, action_tensor) in enumerate(decisions):
                try:
                    # nid 是 nest_id (0-7), zone_id = nid // 2
                    task_nest = nid_tensor.item() if nid_tensor.numel() == 1 else nid_tensor[0].item()
                    task_zone = int(task_nest // 2)
                    if 0 <= task_zone < 4:
                        # 如果该区人数 > 3 (根据总人数15平均每区3.75)，则视为拥堵
                        if zone_loads[task_zone] > 4:
                             # 拥堵度线性惩罚: (load - 4) * 2.0
                             congestion_penalty -= (zone_loads[task_zone] - 4) * 2.0
                except Exception:
                    pass
            
            # 将总拥堵惩罚平摊到当前步的 reward 中 (或者仅针对特定决策，但这里是全局共享reward)
            # 为简单起见，将拥堵惩罚加到全局 manager_r，但这会惩罚所有决策
            # 更好的做法是：让 Critic 学习到这种状态价值低。
            # 这里我们直接加到 manager_r，作为全局指导信号。
            if decisions:
                manager_r += congestion_penalty / max(1, len(decisions))

            manager_r = float(np.clip(manager_r, -50.0, 50.0))
            manager_r_acc += manager_r
            if decisions:
                next_state_vec = get_global_state(env)
                with torch.no_grad():
                    v_next = model.value_net(torch.tensor(next_state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0).item()
                    # FIXED: Clip v_next to prevent value explosion
                    v_next = float(np.clip(v_next, -100.0, 100.0))
                r = torch.full((len(decisions),), manager_r, dtype=torch.float32, device=model.device)
                # FIXED: Clip returns to prevent value explosion
                returns_tensor = r + gamma * v_next
                returns_tensor = torch.clamp(returns_tensor, -100.0, 100.0)
                # Accumulate
                decisions_all.extend(decisions)
                returns_all.append(returns_tensor)
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
            # 优势标准化（无偏差修正，避免N=1产生NaN）；若std过小则仅去均值
            std = adv.std(unbiased=False)
            if torch.isfinite(std) and float(std.item()) > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = adv - adv.mean()
            # 熵退火：从 entropy_coef 线性退火至 1/4 * entropy_coef
            _ec_start = float(entropy_coef)
            _ec_end = float(max(0.0, _ec_start * 0.25))
            _cur_ec = _ec_start + (_ec_end - _ec_start) * (float(step) / max(1.0, float(training_steps - 1)))
            loss_dict = model.compute_manager_loss(
                batch_states, batch_tf, batch_nid, batch_idx, adv, returns,
                task_mask=batch_mask, entropy_coef=_cur_ec
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
        # 平均 manager 差分回报用于日志
        step_reward = float(manager_r_acc / max(1, len(envs)))
        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step)
            loss_log.append(cur_loss)
            reward_log.append(step_reward)
            pol_log.append(cur_pl)
            val_log.append(cur_vl)
            entL_log.append(cur_el)
            ent_log.append(cur_ent)
        # 心跳：每500步写一行到 heartbeat.log（不刷屏），便于跟踪
        try:
            if step % 500 == 0:
                import os
                hb_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'NL-HMARL')
                os.makedirs(hb_dir, exist_ok=True)
                with open(os.path.join(hb_dir, 'heartbeat.log'), 'a') as f:
                    f.write(f"step {step}/{training_steps} mL={cur_loss:.3f} rew={step_reward:.2f}\n")
            # 控制台心跳：每1000步打印一次，避免长时间静默
            if step % 1000 == 0:
                from tqdm import tqdm as _tqdm
                pct = (step + 1) * 100.0 / max(1, int(training_steps))
                _tqdm.write(f"HB {metrics_tag}: {step+1}/{training_steps} ({pct:.2f}%) mL={0.0 if not np.isfinite(cur_loss) else cur_loss:.3f} rew={step_reward:.2f}")
        except Exception:
            pass
        # 仅在 mL (manager loss) 非零/有效时输出一行进度
        try:
            if np.isfinite(cur_loss) and abs(float(cur_loss)) > 0.0:
                pct = (step + 1) * 100.0 / max(1, int(training_steps))
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"step {step+1}/{training_steps} ({pct:.2f}%) mL={cur_loss:.3f} rew={step_reward:.2f}")
        except Exception:
            pass

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

    # Save model checkpoint
    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'NL-HMARL')
            os.makedirs(out_dir, exist_ok=True)
            checkpoint_path = os.path.join(out_dir, 'model_final.pth')
            model.save(checkpoint_path)
            print(f"✅ Model checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            print(f"[warn] Failed to save model checkpoint: {e}")

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
    n_nests: int = 8,  # Changed from 4 to 8 for zone×urgency
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
    # Deprecated: multi-env training removed. Use train_nl_hmarl instead.
    raise RuntimeError('Subproc training removed; use train_nl_hmarl (single-env)')
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL

    used_envs = int(max(2, n_envs))
    vec = SubprocVecEnv(used_envs, env_config, max_tasks=max_tasks)
    f0 = vec.get_features()[0]
    state_dim = int(np.array(f0['state_vec'], dtype=np.float32).shape[0])

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
        feats_list = vec.get_features()
        per_env_decisions = [[] for _ in range(len(feats_list))]
        decisions_all = []
        returns_all = []
        # Build decisions
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
                # Capability-aware mask per picker: regulars skip forklift-only tasks
                comp_mask = local_mask.copy()
                try:
                    p = env.pickers[pid]
                    for ii, tt in enumerate(pending_tasks):
                        if comp_mask[ii] and bool(getattr(tt, 'requires_car', False)) and p.type != PickerType.FORKLIFT:
                            comp_mask[ii] = False
                    filtered_mask = _apply_soft_deadline_filter(env, p, pending_tasks, comp_mask)
                    if filtered_mask is not None:
                        comp_mask = filtered_mask
                except Exception:
                    pass
                if not comp_mask.any():
                    continue
                m = torch.tensor(comp_mask, dtype=torch.bool, device=model.device).unsqueeze(0)
                with torch.no_grad():
                    sel, _ = model.select_tasks(s, tf, nid, m, deterministic=False)
                idx = int(sel.item())
                if idx < 0 or idx >= len(task_ids) or not local_mask[idx]:
                    continue
                per_env_decisions[ei].append((int(pid), int(task_ids[idx])))
                local_mask[idx] = False
                decisions_all.append((s, tf, nid, m, torch.tensor(idx, dtype=torch.long, device=model.device)))
        # Step envs
        outs = vec.step_with_decisions(per_env_decisions)
        # Prepare returns
        for ei, out in enumerate(outs):
            if len(per_env_decisions[ei]) == 0:
                continue
            r = float(out.get('step_reward', 0.0))
            nsv = np.array(out.get('next_state_vec'), dtype=np.float32)
            with torch.no_grad():
                v_next = model.value_net(torch.tensor(nsv, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0)
                # FIXED: Clip v_next to prevent value explosion
                v_next_val = float(np.clip(v_next.item(), -100.0, 100.0))
            # FIXED: Clip returns to prevent value explosion
            returns_tensor = torch.full((len(per_env_decisions[ei]),), r, dtype=torch.float32, device=model.device) + gamma * v_next_val
            returns_tensor = torch.clamp(returns_tensor, -100.0, 100.0)
            returns_all.append(returns_tensor)
        # Update manager
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
                                                   task_mask=batch_mask, entropy_coef=entropy_coef)
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
            try:
                mean_rew = float(np.mean([float(o.get('step_reward', 0.0)) for o in outs]))
            except Exception:
                mean_rew = 0.0
            reward_log.append(mean_rew)
            pol_log.append(cur_pl); val_log.append(cur_vl); entL_log.append(cur_el); ent_log.append(cur_ent)
        try:
            elapsed = max(1e-6, (_time.time() - _t0))
            env_steps = (step + 1) * used_envs
            sps = env_steps / elapsed
            pbar.set_postfix(envs=used_envs, env_steps=env_steps, sps=f"{sps:.1f}", rew=f"{mean_rew:.2f}", loss=f"{(cur_loss if np.isfinite(cur_loss) else 0):.3f}")
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
    n_nests: int = 8,
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
    """Train NL-HMARL with Actor-Critic workers.

    - Manager: same A2C as train_nl_hmarl
    - Workers: per-step A2C with shared parameters across agents
    """
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL
    from exp.actions import smart_navigate, find_adjacent_accessible_position, convert_to_dynamic_actions, get_valid_actions
    from exp.obs import get_agent_observation
    from env.dynamic_warehouse_env import TaskStatus, PickerType

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
    task_feat_dim = int(get_task_features(envs[0], max_tasks=max_tasks, pending_only=True).shape[1])
    n_nests = 8

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
        task_feature_dim=task_feat_dim,
    )
    opt_manager = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr_manager)
    opt_workers = torch.optim.Adam(model.workers.parameters(), lr=lr_workers)

    steps_log, m_loss_log, w_loss_log, reward_log = [], [], [], []
    m_pl_log, m_vl_log, m_entL_log, m_ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL-AC', ncols=100, disable=True)
    value_coef = 2.0
    penalty_coef = 1.0
    risk_value_threshold = 80.0
    risk_slack_threshold = 0.2
    risk_coef = 0.5

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
                # 以 Zone * 2 + Urgency 作为巢标识 (8 nests)
                try:
                    nid = int(getattr(t, 'zone', 0))
                except Exception:
                    nid = 0
                nid = max(0, min(3, nid))

                is_urgent = 0
                try:
                    # Urgency check: priority > 0.7 or close to deadline (< 0.15h approx 9 min)
                    rem = t.deadline - env.current_time
                    if t.priority > 0.7 or rem < 0.15:
                        is_urgent = 1
                except Exception:
                    pass

                nest_ids[i] = nid * 2 + is_urgent
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
                comp_mask = local_mask.copy()
                picker = None
                try:
                    picker = env.pickers[pid]
                except Exception:
                    picker = None
                if picker is not None:
                    for ii, tt in enumerate(t_list):
                        if comp_mask[ii] and bool(getattr(tt, 'requires_car', False)) and picker.type != PickerType.FORKLIFT:
                            comp_mask[ii] = False
                    filtered_mask = _apply_soft_deadline_filter(env, picker, t_list, comp_mask)
                    if filtered_mask is not None:
                        comp_mask = filtered_mask
                if not comp_mask.any():
                    continue
                m = torch.tensor(comp_mask, dtype=torch.bool, device=model.device).unsqueeze(0)
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
            # Mask invalid actions (PICK/DROP only when adjacent; keep movement/idle)
            try:
                vm = np.vstack([np.array(get_valid_actions(env, p), dtype=np.float32) for p in env.pickers])
                vm_t = torch.tensor(vm, dtype=torch.float32, device=model.device)
                probs = torch.clamp(out['action_probs'], min=1e-8) * vm_t
                sums = probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
                probs = probs / sums
            except Exception:
                probs = torch.clamp(out['action_probs'], min=1e-8)
            with torch.no_grad():
                actions_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            actions = {}
            for i, p in enumerate(env.pickers):
                a = int(actions_idx[i].item())
                if a in (0, 1, 2, 3):
                    dd = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[a]
                    nx, ny = p.x + dd[0], p.y + dd[1]
                    invalid = not (0 <= nx < env.width and 0 <= ny < env.height) or (env.grid[ny, nx] == 2)
                    if invalid:
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
            # Manager 差分回报：完成价值 − 迟到罚没的增量 + 任务完成数奖励
            prev_val = float(getattr(env, 'total_value_completed', 0.0))
            prev_pen = float(getattr(env, 'total_value_penalty', 0.0))
            prev_tasks_done = int(getattr(env, 'tasks_completed', 0))
            env_actions = convert_to_dynamic_actions(actions, env, input_space='env')
            _, rewards, dones, _ = env.step(env_actions)
            val_now = float(getattr(env, 'total_value_completed', 0.0))
            pen_now = float(getattr(env, 'total_value_penalty', 0.0))
            tasks_done_now = int(getattr(env, 'tasks_completed', 0))

            risk_penalty = 0.0
            if t_list:
                for idx, keep in enumerate(local_mask):
                    if not keep or idx >= len(t_list):
                        continue
                    risk_penalty += _compute_high_value_risk(
                        env,
                        t_list[idx],
                        slack_threshold=risk_slack_threshold,
                        value_threshold=risk_value_threshold,
                        risk_coef=risk_coef
                    )
            
            # 拥堵惩罚 (AC版)
            zone_loads = [0] * 4
            for p in env.pickers:
                zx = int(p.x / (env.width / 2))
                zy = int(p.y / (env.height / 2))
                z_idx = zy * 2 + zx
                if 0 <= z_idx < 4:
                    zone_loads[z_idx] += 1
            
            congestion_penalty = 0.0
            for d_idx, (_, _, nid_tensor, _, action_tensor) in enumerate(decisions):
                try:
                    task_nest = nid_tensor.item() if nid_tensor.numel() == 1 else nid_tensor[0].item()
                    task_zone = int(task_nest // 2)
                    if 0 <= task_zone < 4:
                        if zone_loads[task_zone] > 4:
                             congestion_penalty -= (zone_loads[task_zone] - 4) * 2.0
                except Exception:
                    pass

            value_gain = (val_now - prev_val)
            penalty_gain = (pen_now - prev_pen)
            tasks_gain = (tasks_done_now - prev_tasks_done)

            # Enhanced manager reward (BALANCED):
            # - Moderate boost on value (2.5 vs 2.0)
            # - Balanced bonus for completing tasks (+3 per task, not +10)
            # - Reduced risk penalty weight (0.5 vs 1.0)
            # - Small time efficiency bonus
            task_completion_bonus = tasks_gain * 3.0  # FIXED: Reduced from 10.0
            time_efficiency_bonus = 0.0
            if decisions:
                # Reward assigning tasks to more agents (encourages parallelism)
                time_efficiency_bonus = len(decisions) * 0.2  # FIXED: Reduced from 0.5

            manager_r = (2.5 * value_gain - 0.9 * penalty_gain +
                        task_completion_bonus + time_efficiency_bonus -
                        0.5 * risk_penalty)
            if decisions:
                manager_r += congestion_penalty / max(1, len(decisions))

            manager_r = float(np.clip(manager_r, -50.0, 50.0))  # FIXED: Tighter clip
            # Accumulate buffers for batched updates
            if decisions:
                next_state_vec = get_global_state(env)
                with torch.no_grad():
                    v_next = model.value_net(torch.tensor(next_state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0).item()
                    # FIXED: Clip v_next to prevent explosion
                    v_next = float(np.clip(v_next, -100.0, 100.0))
                r = torch.full((len(decisions),), manager_r, dtype=torch.float32, device=model.device)
                returns_tensor = r + gamma * v_next
                # FIXED: Clip returns to prevent value explosion
                returns_tensor = torch.clamp(returns_tensor, -100.0, 100.0)
                decisions_all.extend(decisions)
                returns_all.append(returns_tensor)
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
            std = adv.std(unbiased=False)
            if torch.isfinite(std) and float(std.item()) > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = adv - adv.mean()
            # 熵退火：从起始到四分之一
            _ec_start = float(entropy_coef_manager)
            _ec_end = float(max(0.0, _ec_start * 0.25))
            _cur_ec = _ec_start + (_ec_end - _ec_start) * (float(step) / max(1.0, float(training_steps - 1)))
            m_losses = model.compute_manager_loss(batch_states, batch_tf, batch_nid, batch_idx, adv, returns,
                                                 task_mask=batch_mask, entropy_coef=_cur_ec)
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
            # Clip returns to prevent value explosion
            returns_w = torch.clamp(returns_w, -100.0, 100.0)
            adv_w = returns_w - out2['value']
            policy_loss = -(adv_w.detach() * act_logp).mean()
            # Use Huber loss instead of MSE for robustness against outliers
            value_loss = torch.nn.functional.huber_loss(out2['value'], returns_w.detach(), delta=10.0)
            entropy = -(out2['action_probs'] * torch.log(torch.clamp(out2['action_probs'], min=1e-8))).sum(dim=1).mean()
            total_w_loss = policy_loss + value_loss - entropy_coef_workers * entropy
            opt_workers.zero_grad(); total_w_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.workers.parameters(), 1.0)
            opt_workers.step()
            cur_w_loss = float(total_w_loss.item())
        else:
            cur_w_loss = float('nan')

        # 平均即时奖励（用于日志）
        try:
            if rewards_all:
                step_rew = float(np.mean([float(r.mean().item() if hasattr(r, 'item') else r.mean()) for r in rewards_all]))
            else:
                step_rew = 0.0
        except Exception:
            step_rew = 0.0
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
        # 心跳：每500步写入文件 + 每1000步控制台心跳
        try:
            if step % 500 == 0:
                import os
                hb_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'NL-HMARL-AC')
                os.makedirs(hb_dir, exist_ok=True)
                with open(os.path.join(hb_dir, 'heartbeat.log'), 'a') as f:
                    f.write(f"step {step}/{training_steps} mL={cur_m_loss:.3f} wL={cur_w_loss:.3f} rew={step_rew:.2f}\n")
            if step % 1000 == 0:
                from tqdm import tqdm as _tqdm
                pct = (step + 1) * 100.0 / max(1, int(training_steps))
                _tqdm.write(f"HB {metrics_tag}: {step+1}/{training_steps} ({pct:.2f}%) mL={0.0 if not np.isfinite(cur_m_loss) else cur_m_loss:.3f} wL={0.0 if not np.isfinite(cur_w_loss) else cur_w_loss:.3f} rew={step_rew:.2f}")
        except Exception:
            pass
        # Only when mL is non-zero/finite, print a one-line progress with mL and reward
        try:
            if np.isfinite(cur_m_loss) and abs(float(cur_m_loss)) > 0.0:
                _pct = (step + 1) * 100.0 / max(1, int(training_steps))
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"step {step+1}/{training_steps} ({_pct:.2f}%) mL={cur_m_loss:.3f} rew={step_rew:.2f}")
        except Exception:
            pass

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


def train_softmax_hmarl(
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
    device: str = 'cpu',
    speed_function=None,
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'Softmax-HMARL',
    n_envs: int = 1,
):
    """Train Softmax-HMARL manager (categorical) with heuristic workers.

    Mirrors train_nl_hmarl but uses SoftmaxHMARL model and same masks/stabilizations.
    """
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.softmax_hmarl import SoftmaxHMARL
    from exp.actions import smart_navigate, find_adjacent_accessible_position
    from env.dynamic_warehouse_env import TaskStatus, PickerType

    if speed_function is None:
        def speed_function(e):
            return {p.id: float(getattr(p, 'speed', 1.0)) for p in e.pickers}
    envs = [env_ctor(dict(env_config)) for _ in range(max(1, int(n_envs)))]
    for ev in envs:
        ev.set_speed_function(speed_function)
        ev.reset()

    state_dim = int(get_global_state(envs[0]).shape[0])
    worker_obs_dim = 45
    worker_action_dim = 7
    n_agents = envs[0].n_pickers
    task_feat_dim = int(get_task_features(envs[0], max_tasks=max_tasks, pending_only=True).shape[1])

    model = SoftmaxHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,
        n_agents=n_agents,
        worker_obs_dim=worker_obs_dim,
        worker_action_dim=worker_action_dim,
        hidden_dim=hidden_dim,
        device=device,
        task_feature_dim=task_feat_dim,
    )
    optim = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr)

    steps_log, loss_log, reward_log = [], [], []
    pol_log, val_log, entL_log, ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train Softmax-HMARL', ncols=100, disable=True)

    for step in pbar:
        decisions_all, returns_all = [], []
        cur_loss = float('nan'); cur_pl = float('nan'); cur_vl = float('nan'); cur_el = float('nan'); cur_ent = float('nan')
        for env in envs:
            state_vec = get_global_state(env)
            task_feats = get_task_features(env, max_tasks=max_tasks, pending_only=True)
            nest_ids = np.full((max_tasks,), -1, dtype=np.int64)
            mask = np.zeros((max_tasks,), dtype=np.bool_)
            pending_tasks = [t for t in env.task_pool if t.status == TaskStatus.PENDING][:max_tasks]
            for i, t in enumerate(pending_tasks):
                try:
                    nid = int(getattr(t, 'zone', 0))
                except Exception:
                    nid = 0
                nest_ids[i] = max(0, min(3, nid))
                mask[i] = (t.status == TaskStatus.PENDING)
            free_pids = [i for i, p in enumerate(env.pickers) if p.current_task is None and len(p.carrying_items) == 0]
            local_mask = mask.copy()
            for pid in free_pids:
                if not local_mask.any():
                    break
                s = torch.tensor(state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)
                tf = torch.tensor(task_feats, dtype=torch.float32, device=model.device).unsqueeze(0)
                nid = torch.tensor(nest_ids, dtype=torch.long, device=model.device).unsqueeze(0)
                # Capability-aware mask per picker + deadline过滤
                comp_mask = local_mask.copy()
                picker = None
                try:
                    picker = env.pickers[pid]
                    for ii, tt in enumerate(pending_tasks):
                        if comp_mask[ii] and bool(getattr(tt, 'requires_car', False)) and picker.type != PickerType.FORKLIFT:
                            comp_mask[ii] = False
                except Exception:
                    picker = None
                if picker is not None:
                    filtered_mask = _apply_soft_deadline_filter(env, picker, pending_tasks, comp_mask)
                    if filtered_mask is not None:
                        comp_mask = filtered_mask
                if not comp_mask.any():
                    continue
                m = torch.tensor(comp_mask, dtype=torch.bool, device=model.device).unsqueeze(0)
                with torch.no_grad():
                    sel, _ = model.select_tasks(s, tf, m, deterministic=False)
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
                decisions_all.append((s, tf, torch.tensor(idx, dtype=torch.long, device=model.device), m))
            # Heuristic actions
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
            # Per-agent mean reward
            if isinstance(rewards, dict):
                step_reward = float(sum(rewards.values())) / max(1, len(env.pickers))
            else:
                step_reward = float(rewards) / max(1, len(env.pickers))
            if decisions_all:
                next_state_vec = get_global_state(env)
                with torch.no_grad():
                    v_next = model.value_net(torch.tensor(next_state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0).item()
                    # FIXED: Clip v_next to prevent value explosion
                    v_next = float(np.clip(v_next, -100.0, 100.0))
                r = torch.full((len(decisions_all),), step_reward, dtype=torch.float32, device=model.device)
                # FIXED: Clip returns to prevent value explosion
                returns_tensor = r + gamma * v_next
                returns_tensor = torch.clamp(returns_tensor, -100.0, 100.0)
                returns_all.append(returns_tensor)
        # Single batch update
        if len(decisions_all) > 0 and len(returns_all) > 0:
            states = torch.cat([d[0] for d in decisions_all], dim=0)
            tfs = torch.cat([d[1] for d in decisions_all], dim=0)
            idx = torch.stack([d[2] for d in decisions_all], dim=0)
            mask = torch.cat([d[3] for d in decisions_all], dim=0)
            returns = torch.cat(returns_all, dim=0)
            with torch.no_grad():
                v = model.value_net(states).squeeze(-1)
            adv = returns - v
            std = adv.std(unbiased=False)
            if torch.isfinite(std) and float(std.item()) > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = adv - adv.mean()
            loss_dict = model.compute_manager_loss(states, tfs, idx, adv, returns, task_mask=mask, entropy_coef=entropy_coef)
            optim.zero_grad(); loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            optim.step()
            cur_loss = float(loss_dict['total_loss'].item())
            cur_pl = float(loss_dict['policy_loss'].item())
            cur_vl = float(loss_dict['value_loss'].item())
            cur_el = float(loss_dict['entropy_loss'].item())
            cur_ent = float(loss_dict['entropy'].item())

        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step); loss_log.append(cur_loss); reward_log.append(step_reward)
            pol_log.append(cur_pl); val_log.append(cur_vl); entL_log.append(cur_el); ent_log.append(cur_ent)
        # 心跳：每500步 + 控制台每1000步
        try:
            if step % 500 == 0:
                import os
                hb_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'Softmax-HMARL')
                os.makedirs(hb_dir, exist_ok=True)
                with open(os.path.join(hb_dir, 'heartbeat.log'), 'a') as f:
                    f.write(f"step {step}/{training_steps} mL={cur_loss:.3f} rew={step_reward if 'step_reward' in locals() else 0.0:.2f}\n")
            if step % 1000 == 0:
                from tqdm import tqdm as _tqdm
                pct = (step + 1) * 100.0 / max(1, int(training_steps))
                _tqdm.write(f"HB {metrics_tag}: {step+1}/{training_steps} ({pct:.2f}%) mL={0.0 if not np.isfinite(cur_loss) else cur_loss:.3f} rew={(step_reward if 'step_reward' in locals() else 0.0):.2f}")
        except Exception:
            pass

    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'Softmax-HMARL')
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
        except Exception:
            pass

    # Save model checkpoint
    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'Softmax-HMARL')
            os.makedirs(out_dir, exist_ok=True)
            checkpoint_path = os.path.join(out_dir, 'model_final.pth')
            model.save(checkpoint_path)
            print(f"✅ Model checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            print(f"[warn] Failed to save model checkpoint: {e}")

    return model


def train_softmax_hmarl_ac(
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
    device: str = 'cpu',
    speed_function=None,
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'Softmax-HMARL-AC',
    n_envs: int = 1,
):
    """Train Softmax-HMARL with Actor-Critic workers.

    Mirrors train_nl_hmarl_ac with same masks and stabilizations.
    """
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.softmax_hmarl import SoftmaxHMARL
    from exp.actions import smart_navigate, find_adjacent_accessible_position
    from exp.obs import get_agent_observation
    from env.dynamic_warehouse_env import TaskStatus, PickerType

    if speed_function is None:
        def speed_function(e):
            return {p.id: float(getattr(e, 'speed', 1.0)) for p in e.pickers}
    envs = [env_ctor(dict(env_config)) for _ in range(max(1, int(n_envs)))]
    for ev in envs:
        ev.set_speed_function(speed_function)
        ev.reset()

    state_dim = int(get_global_state(envs[0]).shape[0])
    task_feat_dim = int(get_task_features(envs[0], max_tasks=max_tasks, pending_only=True).shape[1])
    worker_obs_dim = 45
    worker_action_dim = 7
    n_agents = envs[0].n_pickers

    model = SoftmaxHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,
        n_agents=n_agents,
        worker_obs_dim=worker_obs_dim,
        worker_action_dim=worker_action_dim,
        hidden_dim=hidden_dim,
        device=device,
        task_feature_dim=task_feat_dim,
    )
    opt_manager = torch.optim.Adam(list(model.manager.parameters()) + list(model.value_net.parameters()), lr=lr_manager)
    opt_workers = torch.optim.Adam(model.workers.parameters(), lr=lr_workers)

    steps_log, m_loss_log, w_loss_log, reward_log = [], [], [], []
    m_pl_log, m_vl_log, m_entL_log, m_ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train Softmax-HMARL-AC', ncols=100, disable=True)

    for step in pbar:
        decisions_all, returns_all = [], []
        obs_all, actions_all, rewards_all, dones_all = [], [], [], []
        for env in envs:
            state_vec = get_global_state(env)
            task_feats = get_task_features(env, max_tasks=max_tasks, pending_only=True)
            nest_ids = np.full((max_tasks,), -1, dtype=np.int64)
            mask = np.zeros((max_tasks,), dtype=np.bool_)
            t_list = [t for t in env.task_pool if t.status == TaskStatus.PENDING][:max_tasks]
            for i, t in enumerate(t_list):
                try:
                    nid = int(getattr(t, 'zone', 0))
                except Exception:
                    nid = 0
                nest_ids[i] = max(0, min(3, nid))
                mask[i] = (t.status == TaskStatus.PENDING)
            free_pids = [i for i, p in enumerate(env.pickers) if p.current_task is None and len(p.carrying_items) == 0]
            local_mask = mask.copy()
            for pid in free_pids:
                if not local_mask.any():
                    break
                s = torch.tensor(state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)
                tf = torch.tensor(task_feats, dtype=torch.float32, device=model.device).unsqueeze(0)
                # capability-aware mask per picker + deadline过滤
                comp_mask = local_mask.copy()
                picker = None
                try:
                    picker = env.pickers[pid]
                    for ii, tt in enumerate(t_list):
                        if comp_mask[ii] and bool(getattr(tt, 'requires_car', False)) and picker.type != PickerType.FORKLIFT:
                            comp_mask[ii] = False
                except Exception:
                    picker = None
                if picker is not None:
                    filtered_mask = _apply_soft_deadline_filter(env, picker, t_list, comp_mask)
                    if filtered_mask is not None:
                        comp_mask = filtered_mask
                if not comp_mask.any():
                    continue
                m = torch.tensor(comp_mask, dtype=torch.bool, device=model.device).unsqueeze(0)
                with torch.no_grad():
                    sel, _ = model.select_tasks(s, tf, m, deterministic=False)
                idx = int(sel.item())
                if not local_mask[idx] or idx >= len(t_list):
                    continue
                t = t_list[idx]
                if not (t.status == TaskStatus.PENDING):
                    continue
                t.status = TaskStatus.ASSIGNED
                t.assigned_picker = pid
                env.pickers[pid].current_task = t
                local_mask[idx] = False
                decisions_all.append((s, tf, torch.tensor(idx, dtype=torch.long, device=model.device), m))
            # Worker actions with mask
            obs_batch = [get_agent_observation(env, p, include_global=True) for p in env.pickers]
            obs_tensor = torch.tensor(np.vstack(obs_batch), dtype=torch.float32, device=model.device)
            out = model.workers(obs_tensor)
            try:
                vm = np.vstack([np.array(get_valid_actions(env, p), dtype=np.float32) for p in env.pickers])
                vm_t = torch.tensor(vm, dtype=torch.float32, device=model.device)
                probs = torch.clamp(out['action_probs'], min=1e-8) * vm_t
                sums = probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
                probs = probs / sums
            except Exception:
                probs = torch.clamp(out['action_probs'], min=1e-8)
            with torch.no_grad():
                actions_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            actions = {i: int(actions_idx[i].item()) for i in range(n_agents)}
            actions_all.append(actions_idx.to(model.device))
            env_actions = convert_to_dynamic_actions(actions, env, input_space='env')
            _, rewards, dones, _ = env.step(env_actions)
            # Per-agent vectors
            if isinstance(rewards, dict):
                r_vec = torch.tensor([float(rewards.get(i, 0.0)) for i in range(n_agents)], dtype=torch.float32, device=model.device)
                d_vec = torch.tensor([1.0 if dones.get(i, False) else 0.0 for i in range(n_agents)], dtype=torch.float32, device=model.device)
                step_rew = float(sum(rewards.values())) / max(1, n_agents)
            else:
                step_rew = float(rewards) / max(1, n_agents)
                r_vec = torch.full((n_agents,), step_rew, dtype=torch.float32, device=model.device)
                d_vec = torch.zeros((n_agents,), dtype=torch.float32, device=model.device)
            rewards_all.append(r_vec); dones_all.append(d_vec); obs_all.append(obs_tensor)
            if decisions_all:
                nsv = get_global_state(env)
                with torch.no_grad():
                    v_next = model.value_net(torch.tensor(nsv, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0)
                    # FIXED: Clip v_next to prevent value explosion
                    v_next_val = float(np.clip(v_next.item(), -100.0, 100.0))
                # FIXED: Clip returns to prevent value explosion
                returns_tensor = torch.full((len(decisions_all),), step_rew, dtype=torch.float32, device=model.device) + gamma * v_next_val
                returns_tensor = torch.clamp(returns_tensor, -100.0, 100.0)
                returns_all.append(returns_tensor)
        # Manager update
        if len(decisions_all) > 0 and len(returns_all) > 0:
            states = torch.cat([d[0] for d in decisions_all], dim=0)
            tfs = torch.cat([d[1] for d in decisions_all], dim=0)
            idx = torch.stack([d[2] for d in decisions_all], dim=0)
            mask = torch.cat([d[3] for d in decisions_all], dim=0)
            returns = torch.cat(returns_all, dim=0)
            with torch.no_grad():
                v = model.value_net(states).squeeze(-1)
            adv = returns - v
            std = adv.std(unbiased=False)
            if torch.isfinite(std) and float(std.item()) > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = adv - adv.mean()
            loss_dict = model.compute_manager_loss(states, tfs, idx, adv, returns, task_mask=mask, entropy_coef=entropy_coef_manager)
            opt_manager.zero_grad(); loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            opt_manager.step()
            cur_m_loss = float(loss_dict['total_loss'].item())
            cur_m_pl = float(loss_dict['policy_loss'].item())
            cur_m_vl = float(loss_dict['value_loss'].item())
            cur_m_el = float(loss_dict['entropy_loss'].item())
            cur_m_ent = float(loss_dict['entropy'].item())
        else:
            cur_m_loss = float('nan'); cur_m_pl = float('nan'); cur_m_vl = float('nan'); cur_m_el = float('nan'); cur_m_ent = float('nan')
        # Worker A2C update
        if len(obs_all) > 0 and len(actions_all) > 0:
            obs_b = torch.cat(obs_all, dim=0)
            actions_b = torch.cat(actions_all, dim=0)
            out2 = model.workers(obs_b)
            log_probs_all = torch.log(torch.clamp(out2['action_probs'], min=1e-8))
            act_logp = log_probs_all.gather(1, actions_b.unsqueeze(1)).squeeze(1)
            next_obs_all = []
            for env in envs:
                next_obs_all.extend([get_agent_observation(env, p, include_global=True) for p in env.pickers])
            next_obs_tensor = torch.tensor(np.vstack(next_obs_all), dtype=torch.float32, device=model.device)
            with torch.no_grad():
                next_vals = model.workers(next_obs_tensor)['value']
            r_vec = torch.cat(rewards_all, dim=0)
            d_vec = torch.cat(dones_all, dim=0)
            returns_w = r_vec + gamma * next_vals * (1.0 - d_vec)
            # Clip returns to prevent value explosion
            returns_w = torch.clamp(returns_w, -100.0, 100.0)
            adv_w = returns_w - out2['value']
            policy_loss = -(adv_w.detach() * act_logp).mean()
            # Use Huber loss instead of MSE for robustness against outliers
            value_loss = torch.nn.functional.huber_loss(out2['value'], returns_w.detach(), delta=10.0)
            entropy = -(out2['action_probs'] * torch.log(torch.clamp(out2['action_probs'], min=1e-8))).sum(dim=1).mean()
            total_w_loss = policy_loss + value_loss - entropy_coef_workers * entropy
            opt_workers.zero_grad(); total_w_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.workers.parameters(), 1.0)
            opt_workers.step()
            cur_w_loss = float(total_w_loss.item())
        else:
            cur_w_loss = float('nan')
        # Logging
        try:
            step_rew = float(np.mean([float(r.mean().item()) if hasattr(r, 'mean') else float(r.mean()) for r in rewards_all])) if rewards_all else 0.0
        except Exception:
            step_rew = 0.0
        if log_metrics and (step % max(1, log_every) == 0):
            steps_log.append(step); m_loss_log.append(cur_m_loss); w_loss_log.append(cur_w_loss)
            reward_log.append(step_rew); m_pl_log.append(cur_m_pl); m_vl_log.append(cur_m_vl); m_entL_log.append(cur_m_el); m_ent_log.append(cur_m_ent)
        # 心跳：每500步 + 控制台每1000步
        try:
            if step % 500 == 0:
                import os
                hb_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'Softmax-HMARL-AC')
                os.makedirs(hb_dir, exist_ok=True)
                with open(os.path.join(hb_dir, 'heartbeat.log'), 'a') as f:
                    f.write(f"step {step}/{training_steps} mL={cur_m_loss:.3f} wL={cur_w_loss:.3f} rew={step_rew:.2f}\n")
            if step % 1000 == 0:
                from tqdm import tqdm as _tqdm
                pct = (step + 1) * 100.0 / max(1, int(training_steps))
                _tqdm.write(f"HB {metrics_tag}: {step+1}/{training_steps} ({pct:.2f}%) mL={0.0 if not np.isfinite(cur_m_loss) else cur_m_loss:.3f} wL={0.0 if not np.isfinite(cur_w_loss) else cur_w_loss:.3f} rew={step_rew:.2f}")
        except Exception:
            pass

    if log_metrics:
        try:
            out_dir = os.path.join(metrics_dir or 'results/train_metrics', metrics_tag or 'Softmax-HMARL-AC')
            os.makedirs(out_dir, exist_ok=True)
            import pandas as pd
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
        except Exception:
            pass
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
    # Deprecated: multi-env training removed. Use train_nl_hmarl_ac instead.
    raise RuntimeError('Subproc training removed; use train_nl_hmarl_ac (single-env)')
    import os
    import torch
    import numpy as np
    from tqdm import tqdm
    from baselines.nl_hmarl import NLHMARL

    used_envs = int(max(2, n_envs))
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
        task_feature_dim=task_feat_dim,
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
                # FIXED: Clip v_next to prevent value explosion
                v_next_val = float(np.clip(v_next.item(), -100.0, 100.0))
            # FIXED: Clip returns to prevent value explosion
            returns_tensor = torch.full((len(per_env_decisions[ei]),), r, dtype=torch.float32, device=model.device) + gamma * v_next_val
            returns_tensor = torch.clamp(returns_tensor, -100.0, 100.0)
            returns_all.append(returns_tensor)
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
            # 优势标准化（无偏差修正，避免N=1产生NaN）；若std过小则仅去均值
            std = adv.std(unbiased=False)
            if torch.isfinite(std) and float(std.item()) > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = adv - adv.mean()
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
            # Clip returns to prevent value explosion
            returns_w = torch.clamp(returns_w, -100.0, 100.0)
            adv_w = returns_w - out2['value']
            # 优势标准化（无偏差修正，避免N=1产生NaN）；若std过小则仅去均值
            std_w = adv_w.std(unbiased=False)
            if torch.isfinite(std_w) and float(std_w.item()) > 1e-8:
                adv_w = (adv_w - adv_w.mean()) / (std_w + 1e-8)
            else:
                adv_w = adv_w - adv_w.mean()
            policy_loss = -(adv_w.detach() * act_logp).mean()
            # Use Huber loss instead of MSE for robustness against outliers
            value_loss = torch.nn.functional.huber_loss(out2['value'], returns_w.detach(), delta=10.0)
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
