from typing import Dict, Optional, Callable
import numpy as np
import torch
from tqdm import tqdm

from baselines.flat_marl_dqn import FlatMARLDQN
from exp.obs import get_agent_observation
from exp.actions import get_valid_actions, convert_to_dynamic_actions
from exp.assigners import assign_tasks_dynamic
from exp.obs import get_global_state, get_task_features


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
    pbar = tqdm(range(training_steps), desc='Train DQN', ncols=100)
    avg_reward_ema = None
    last_loss: Optional[float] = None

    # Metrics buffers
    steps_log, eps_log, loss_log, stepR_log, avgR_log = [], [], [], [], []
    q_logs = []  # list of length-7 arrays (mean Q per action)
    for step in pbar:
        # 分配任务（关键）：为空闲拣货员分配可执行任务，避免永远没有拣/投事件
        assign_tasks_dynamic(env)
        # 收集观测
        obs_batch = [get_agent_observation(env, p, include_global=not pure_learning) for p in env.pickers]
        # 选动作（与环境一致的索引空间: 0..3=UP/DOWN/LEFT/RIGHT, 4=IDLE, 5/6=PICK/DROP）
        chosen_actions: Dict[int, int] = {}
        for i, p in enumerate(env.pickers):
            mask = np.array(get_valid_actions(env, p))
            a = model.select_action(obs_batch[i], deterministic=False, action_mask=mask)
            chosen_actions[i] = a
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
):
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

    # Build training env
    env = env_ctor(dict(env_config))
    if speed_function is None:
        def speed_function(e):
            return {p.id: float(getattr(p, 'speed', 1.0)) for p in e.pickers}
    env.set_speed_function(speed_function)
    _ = env.reset()

    # Dimensions
    state_dim = int(get_global_state(env).shape[0])
    worker_obs_dim = 45  # reuse agent obs with include_global=True
    worker_action_dim = 7
    n_agents = env.n_pickers
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
    pol_log, val_log, entL_log, ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL', ncols=100)
    for step in pbar:
        # Assign tasks to free pickers using current manager policy
        state_vec = get_global_state(env)
        task_feats = get_task_features(env, max_tasks=max_tasks, pending_only=True)
        # Build nest ids and mask for current tasks
        nest_ids = np.full((max_tasks,), -1, dtype=np.int64)
        mask = np.zeros((max_tasks,), dtype=np.bool_)
        pending_tasks = [t for t in env.task_pool if t.status == TaskStatus.PENDING][:max_tasks]
        for i, t in enumerate(pending_tasks):
            # Nest by forklift need: 1 if requires_car else 0
            nest_ids[i] = 1 if bool(getattr(t, 'requires_car', False)) else 0
            mask[i] = (t.status == TaskStatus.PENDING)

        # One decision per free picker (greedy without collision)
        free_pids = [i for i, p in enumerate(env.pickers) if p.current_task is None and len(p.carrying_items) == 0]
        decisions = []  # tuples of (state, task_feats, nest_ids, mask, chosen_idx)
        # Keep a local copy of mask to avoid picking the same task twice
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
            if not local_mask[idx]:
                continue
            # Map index to task object (pending slice)
            t_list = [t for t in env.task_pool if t.status == TaskStatus.PENDING][:max_tasks]
            if idx >= len(t_list):
                continue
            t = t_list[idx]
            if not (t.status == TaskStatus.PENDING):
                continue
            # Assign
            t.status = TaskStatus.ASSIGNED
            t.assigned_picker = pid
            env.pickers[pid].current_task = t
            local_mask[idx] = False
            decisions.append((s, tf, nid, m, torch.tensor(idx, dtype=torch.long, device=model.device)))

        # Build worker actions via simple navigation heuristic
        actions = {}
        for i, p in enumerate(env.pickers):
            t = getattr(p, 'current_task', None)
            if t is None:
                actions[i] = 4
                continue
            if len(p.carrying_items) == 0:
                if t.shelf_id is None or t.shelf_id >= len(env.shelves):
                    actions[i] = 4
                else:
                    sh = env.shelves[t.shelf_id]
                    adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
                    if adj is None:
                        actions[i] = 4
                    elif (p.x, p.y) == adj or (abs(p.x - sh['x']) + abs(p.y - sh['y']) == 1):
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
        # Manager reward: sum of agent rewards
        step_reward = float(sum(rewards.values())) if isinstance(rewards, dict) else float(rewards)

        # TD(0) update for manager on the collected decisions
        if decisions:
            next_state_vec = get_global_state(env)
            with torch.no_grad():
                v_next = model.value_net(torch.tensor(next_state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0)
            batch_states = torch.cat([d[0] for d in decisions], dim=0)
            batch_tf = torch.cat([d[1] for d in decisions], dim=0)
            batch_nid = torch.cat([d[2] for d in decisions], dim=0)
            batch_mask = torch.cat([d[3] for d in decisions], dim=0)
            batch_idx = torch.stack([d[4] for d in decisions], dim=0).to(model.device)
            # Same reward for each manager choice within the step (simple credit assignment)
            r = torch.full((len(decisions),), step_reward, dtype=torch.float32, device=model.device)
            with torch.no_grad():
                v = model.value_net(batch_states).squeeze(-1)
            returns = r + gamma * v_next.item()
            adv = returns - v
            loss_dict = model.compute_manager_loss(
                batch_states, batch_tf, batch_nid, batch_idx, adv, returns,
                task_mask=batch_mask, entropy_coef=entropy_coef
            )
            optim.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            optim.step()
            cur_loss = float(loss_dict['total_loss'].item())
            cur_pl = float(loss_dict['policy_loss'].item())
            cur_vl = float(loss_dict['value_loss'].item())
            cur_el = float(loss_dict['entropy_loss'].item())
            cur_ent = float(loss_dict['entropy'].item())
        else:
            cur_loss = float('nan')
            cur_pl = float('nan')
            cur_vl = float('nan')
            cur_el = float('nan')
            cur_ent = float('nan')

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
    from exp.actions import smart_navigate, find_adjacent_accessible_position, convert_to_dynamic_actions
    from exp.obs import get_agent_observation
    from env.dynamic_warehouse_env import TaskStatus

    env = env_ctor(dict(env_config))
    if speed_function is None:
        def speed_function(e):
            return {p.id: float(getattr(p, 'speed', 1.0)) for p in e.pickers}
    env.set_speed_function(speed_function)
    _ = env.reset()

    # Dimensions
    state_dim = int(get_global_state(env).shape[0])
    worker_obs_dim = 45
    worker_action_dim = 7
    n_agents = env.n_pickers
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
    m_pl_log, m_vl_log, m_entL_log, m_ent_log = [], [], [], []
    pbar = tqdm(range(training_steps), desc='Train NL-HMARL-AC', ncols=100)

    for step in pbar:
        # 1) Manager assigns tasks to free pickers
        state_vec = get_global_state(env)
        task_feats = get_task_features(env, max_tasks=max_tasks, pending_only=True)
        nest_ids = np.full((max_tasks,), -1, dtype=np.int64)
        mask = np.zeros((max_tasks,), dtype=np.bool_)
        t_list = [t for t in env.task_pool if t.status == TaskStatus.PENDING][:max_tasks]
        for i, t in enumerate(t_list):
            # Nest by forklift need: 1 if requires_car else 0
            nest_ids[i] = 1 if bool(getattr(t, 'requires_car', False)) else 0
            mask[i] = (t.status == TaskStatus.PENDING)

        free_pids = [i for i, p in enumerate(env.pickers) if p.current_task is None and len(p.carrying_items) == 0]
        decisions = []
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

        # 2) Workers select actions via shared policy
        obs_batch = [get_agent_observation(env, p, include_global=True) for p in env.pickers]
        obs_tensor = torch.tensor(np.vstack(obs_batch), dtype=torch.float32, device=model.device)
        out = model.workers(obs_tensor)
        action_probs = out['action_probs']  # [N,7]
        values = out['value']               # [N]
        # Sample actions from probs
        with torch.no_grad():
            actions_idx = torch.multinomial(torch.clamp(action_probs, min=1e-8), num_samples=1).squeeze(1)
        # Sanitize invalid moves (avoid shelves/out-of-bounds). If invalid, try to navigate to goal or idle.
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
        env_actions = convert_to_dynamic_actions(actions, env, input_space='env')

        # 3) Step env
        _, rewards, dones, _ = env.step(env_actions)
        step_rew = float(sum(rewards.values())) if isinstance(rewards, dict) else float(rewards)

        # 4) Manager update (TD(0) on aggregated reward)
        if decisions:
            next_state_vec = get_global_state(env)
            with torch.no_grad():
                v_next = model.value_net(torch.tensor(next_state_vec, dtype=torch.float32, device=model.device).unsqueeze(0)).squeeze(0)
            batch_states = torch.cat([d[0] for d in decisions], dim=0)
            batch_tf = torch.cat([d[1] for d in decisions], dim=0)
            batch_nid = torch.cat([d[2] for d in decisions], dim=0)
            batch_mask = torch.cat([d[3] for d in decisions], dim=0)
            batch_idx = torch.stack([d[4] for d in decisions], dim=0).to(model.device)
            r = torch.full((len(decisions),), step_rew, dtype=torch.float32, device=model.device)
            with torch.no_grad():
                v = model.value_net(batch_states).squeeze(-1)
            returns = r + gamma * v_next.item()
            adv = returns - v
            m_losses = model.compute_manager_loss(
                batch_states, batch_tf, batch_nid, batch_idx, adv, returns,
                task_mask=batch_mask, entropy_coef=entropy_coef_manager
            )
            opt_manager.zero_grad()
            m_losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0)
            opt_manager.step()
            cur_m_loss = float(m_losses['total_loss'].item())
            cur_m_pl = float(m_losses['policy_loss'].item())
            cur_m_vl = float(m_losses['value_loss'].item())
            cur_m_el = float(m_losses['entropy_loss'].item())
            cur_m_ent = float(m_losses['entropy'].item())
        else:
            cur_m_loss = float('nan')
            cur_m_pl = float('nan')
            cur_m_vl = float('nan')
            cur_m_el = float('nan')
            cur_m_ent = float('nan')

        # 5) Workers update (A2C)
        next_obs_batch = [get_agent_observation(env, p, include_global=True) for p in env.pickers]
        next_obs_tensor = torch.tensor(np.vstack(next_obs_batch), dtype=torch.float32, device=model.device)
        with torch.no_grad():
            next_vals = model.workers(next_obs_tensor)['value']  # [N]
        # Gather log_probs for executed actions (after sanitization)
        # Recompute forward to align probabilities with executed actions
        out2 = model.workers(obs_tensor)
        log_probs_all = torch.log(torch.clamp(out2['action_probs'], min=1e-8))
        exec_actions = torch.tensor([actions[i] for i in range(n_agents)], dtype=torch.long, device=model.device)
        act_logp = log_probs_all.gather(1, exec_actions.unsqueeze(1)).squeeze(1)
        values2 = out2['value']
        # Build per-agent rewards vector
        if isinstance(rewards, dict):
            r_vec = torch.tensor([float(rewards.get(i, 0.0)) for i in range(n_agents)], dtype=torch.float32, device=model.device)
        else:
            r_avg = float(rewards) / max(1, n_agents)
            r_vec = torch.full((n_agents,), r_avg, dtype=torch.float32, device=model.device)
        done_vec = torch.tensor([1.0 if (isinstance(dones, dict) and dones.get(i, False)) else 0.0 for i in range(n_agents)], dtype=torch.float32, device=model.device)
        returns_w = r_vec + gamma * next_vals * (1.0 - done_vec)
        adv_w = returns_w - values2
        policy_loss = -(adv_w.detach() * act_logp).mean()
        value_loss = torch.nn.functional.mse_loss(values2, returns_w.detach())
        entropy = -(out2['action_probs'] * torch.log(torch.clamp(out2['action_probs'], min=1e-8))).sum(dim=1).mean()
        total_w_loss = policy_loss + value_loss - entropy_coef_workers * entropy
        opt_workers.zero_grad()
        total_w_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.workers.parameters(), 1.0)
        opt_workers.step()
        cur_w_loss = float(total_w_loss.item())

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
