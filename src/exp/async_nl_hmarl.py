"""
NL-HMARL-AC 异步训练适配器
将原有的串行训练逻辑拆分为CPU收集和GPU训练两部分
"""

import numpy as np
import torch
from typing import Dict, List, Any
from exp.obs import get_global_state, get_task_features, get_agent_observation
from exp.actions import get_valid_actions, convert_to_dynamic_actions, smart_navigate
from exp.actions import find_adjacent_accessible_position
from env.dynamic_warehouse_env import TaskStatus


class NLHMARLAsyncAdapter:
    """NL-HMARL-AC异步训练适配器"""

    def __init__(
        self,
        model,
        n_agents: int = 8,
        max_tasks: int = 20,
        gamma: float = 0.99,
        entropy_coef_manager: float = 0.01,
        entropy_coef_worker: float = 0.01,
        device: str = 'cuda',
    ):
        self.model = model
        self.n_agents = n_agents
        self.max_tasks = max_tasks
        self.gamma = gamma
        self.entropy_coef_manager = entropy_coef_manager
        self.entropy_coef_worker = entropy_coef_worker
        self.device = device

        # 训练步数计数器（用于熵退火）
        self.training_step = 0
        self.total_training_steps = 200000  # 默认值

    def create_collect_fn(self):
        """创建CPU收集函数（在收集线程中运行）"""

        def collect_fn(env, local_model):
            """
            单步环境交互，收集经验

            Args:
                env: 环境实例
                local_model: 本地模型副本（CPU线程用）

            Returns:
                experience: 经验字典，包含manager和worker的数据
            """
            try:
                local_model.eval()  # 评估模式
                experiences = []

                # ===== Manager: 任务分配 =====
                state_vec = get_global_state(env)
                task_feats = get_task_features(env, max_tasks=self.max_tasks, pending_only=True)
                nest_ids = np.full((self.max_tasks,), -1, dtype=np.int64)
                mask = np.zeros((self.max_tasks,), dtype=np.bool_)

                t_list = [t for t in env.task_pool if t.status == TaskStatus.PENDING][:self.max_tasks]
                for i, t in enumerate(t_list):
                    try:
                        nid = int(getattr(t, 'zone', 0))
                    except Exception:
                        nid = 0
                    nest_ids[i] = max(0, min(3, nid))
                    mask[i] = (t.status == TaskStatus.PENDING)

                free_pids = [i for i, p in enumerate(env.pickers)
                            if p.current_task is None and len(p.carrying_items) == 0]

                local_mask = mask.copy()
                manager_decisions = []

                # Manager分配任务
                for pid in free_pids:
                    if not local_mask.any():
                        break

                    with torch.no_grad():
                        s = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                        tf = torch.tensor(task_feats, dtype=torch.float32).unsqueeze(0)
                        nid = torch.tensor(nest_ids, dtype=torch.long).unsqueeze(0)
                        m = torch.tensor(local_mask, dtype=torch.bool).unsqueeze(0)

                        sel, _ = local_model.select_tasks(s, tf, nid, m, deterministic=False)

                    idx = int(sel.item())
                    if not local_mask[idx] or idx >= len(t_list):
                        continue

                    t = t_list[idx]
                    if t.status != TaskStatus.PENDING:
                        continue

                    # 分配任务
                    t.status = TaskStatus.ASSIGNED
                    t.assigned_picker = pid
                    env.pickers[pid].current_task = t
                    local_mask[idx] = False

                    # 保存manager决策数据（转为numpy，避免GPU tensor）
                    manager_decisions.append({
                        'state': state_vec.copy(),
                        'task_feats': task_feats.copy(),
                        'nest_ids': nest_ids.copy(),
                        'mask': mask.copy(),
                        'action_idx': idx,
                    })

                # ===== Worker: 导航动作 =====
                obs_batch = [get_agent_observation(env, p, include_global=True) for p in env.pickers]
                obs_array = np.vstack(obs_batch)

                with torch.no_grad():
                    obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
                    out = local_model.workers(obs_tensor)

                    # 动作掩码
                    vm = np.vstack([np.array(get_valid_actions(env, p), dtype=np.float32)
                                   for p in env.pickers])
                    vm_t = torch.tensor(vm, dtype=torch.float32)
                    probs = torch.clamp(out['action_probs'], min=1e-8) * vm_t
                    sums = probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    probs = probs / sums

                    actions_idx = torch.multinomial(probs, num_samples=1).squeeze(1)

                # 构造动作字典
                actions = {}
                for i, p in enumerate(env.pickers):
                    a = int(actions_idx[i].item())

                    # 处理无效移动
                    if a in (0, 1, 2, 3):
                        dd = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[a]
                        nx, ny = p.x + dd[0], p.y + dd[1]
                        invalid = (not (0 <= nx < env.width and 0 <= ny < env.height) or
                                 (env.grid[ny, nx] == 2))

                        if invalid:
                            # 使用启发式导航
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
                                a = 4  # IDLE

                    actions[i] = a

                # 记录执行前的价值
                prev_val = float(getattr(env, 'total_value_completed', 0.0))
                prev_pen = float(getattr(env, 'total_value_penalty', 0.0))

                # 执行环境step
                env_actions = convert_to_dynamic_actions(actions, env, input_space='env')
                _, rewards, dones, _ = env.step(env_actions)

                # Manager奖励
                val_now = float(getattr(env, 'total_value_completed', 0.0))
                pen_now = float(getattr(env, 'total_value_penalty', 0.0))
                manager_r = (val_now - prev_val) - (pen_now - prev_pen)
                manager_r = float(np.clip(manager_r, -50.0, 50.0))

                # 下一状态
                next_state_vec = get_global_state(env)
                next_obs_batch = [get_agent_observation(env, p, include_global=True) for p in env.pickers]
                next_obs_array = np.vstack(next_obs_batch)

                # Worker奖励
                if isinstance(rewards, dict):
                    r_vec = np.array([float(rewards.get(i, 0.0)) for i in range(self.n_agents)])
                    d_vec = np.array([1.0 if dones.get(i, False) else 0.0 for i in range(self.n_agents)])
                else:
                    r_avg = float(rewards) / max(1, self.n_agents)
                    r_vec = np.full((self.n_agents,), r_avg)
                    d_vec = np.zeros((self.n_agents,))

                # 构造经验
                experience = {
                    # Manager数据
                    'manager_decisions': manager_decisions,
                    'manager_reward': manager_r,
                    'next_state': next_state_vec,

                    # Worker数据
                    'worker_obs': obs_array,
                    'worker_actions': np.array([actions[i] for i in range(self.n_agents)]),
                    'worker_rewards': r_vec,
                    'worker_dones': d_vec,
                    'worker_next_obs': next_obs_array,
                }

                return [experience]  # 返回列表

            except Exception as e:
                print(f"⚠️ collect_fn错误: {e}")
                import traceback
                traceback.print_exc()
                return []

        return collect_fn

    def create_train_fn(self, opt_manager, opt_workers):
        """创建GPU训练函数（在训练线程中运行）"""

        def train_fn(model, batch):
            """
            GPU训练步骤

            Args:
                model: 神经网络模型
                batch: 经验批次（List[experience_dict]）

            Returns:
                loss: 总损失
                metrics: 训练指标字典
            """
            try:
                model.train()

                # ===== 收集Manager数据 =====
                all_manager_decisions = []
                all_manager_rewards = []
                all_next_states = []

                for exp in batch:
                    all_manager_decisions.extend(exp['manager_decisions'])
                    if exp['manager_decisions']:
                        all_manager_rewards.append(exp['manager_reward'])
                        all_next_states.append(exp['next_state'])

                # ===== Manager训练 =====
                manager_loss = 0.0
                manager_metrics = {}

                if all_manager_decisions:
                    # 构造批量tensor
                    batch_states = torch.tensor(
                        np.vstack([d['state'] for d in all_manager_decisions]),
                        dtype=torch.float32, device=self.device
                    )
                    batch_tf = torch.tensor(
                        np.vstack([d['task_feats'] for d in all_manager_decisions]),
                        dtype=torch.float32, device=self.device
                    )
                    batch_nid = torch.tensor(
                        np.vstack([d['nest_ids'] for d in all_manager_decisions]),
                        dtype=torch.long, device=self.device
                    )
                    batch_mask = torch.tensor(
                        np.vstack([d['mask'] for d in all_manager_decisions]),
                        dtype=torch.bool, device=self.device
                    )
                    batch_idx = torch.tensor(
                        [d['action_idx'] for d in all_manager_decisions],
                        dtype=torch.long, device=self.device
                    )

                    # 计算returns
                    returns_list = []
                    for i, mr in enumerate(all_manager_rewards):
                        next_state = torch.tensor(all_next_states[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                        with torch.no_grad():
                            v_next = model.value_net(next_state).squeeze().item()

                        # 每个决策都用相同的reward
                        n_decisions_in_exp = sum(1 for d in all_manager_decisions if np.array_equal(d['state'], all_next_states[i]))
                        returns_list.extend([mr + self.gamma * v_next] * n_decisions_in_exp)

                    returns = torch.tensor(returns_list, dtype=torch.float32, device=self.device)

                    # 计算优势
                    with torch.no_grad():
                        v = model.value_net(batch_states).squeeze(-1)
                    adv = returns - v
                    std = adv.std(unbiased=False)
                    if torch.isfinite(std) and float(std.item()) > 1e-8:
                        adv = (adv - adv.mean()) / (std + 1e-8)
                    else:
                        adv = adv - adv.mean()

                    # 熵退火
                    ec_start = float(self.entropy_coef_manager)
                    ec_end = float(max(0.0, ec_start * 0.25))
                    progress = float(self.training_step) / max(1.0, float(self.total_training_steps - 1))
                    cur_ec = ec_start + (ec_end - ec_start) * progress

                    # Manager损失
                    m_losses = model.compute_manager_loss(
                        batch_states, batch_tf, batch_nid, batch_idx, adv, returns,
                        task_mask=batch_mask, entropy_coef=cur_ec
                    )

                    opt_manager.zero_grad()
                    m_losses['total_loss'].backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(model.manager.parameters()) + list(model.value_net.parameters()), 1.0
                    )
                    opt_manager.step()

                    manager_loss = float(m_losses['total_loss'].item())
                    manager_metrics = {
                        'policy_loss': float(m_losses['policy_loss'].item()),
                        'value_loss': float(m_losses['value_loss'].item()),
                        'entropy': float(m_losses['entropy'].item()),
                    }

                # ===== Worker训练 =====
                worker_loss = 0.0
                worker_metrics = {}

                # 收集Worker数据
                all_obs = []
                all_actions = []
                all_rewards = []
                all_dones = []
                all_next_obs = []

                for exp in batch:
                    all_obs.append(exp['worker_obs'])
                    all_actions.append(exp['worker_actions'])
                    all_rewards.append(exp['worker_rewards'])
                    all_dones.append(exp['worker_dones'])
                    all_next_obs.append(exp['worker_next_obs'])

                if all_obs:
                    obs_b = torch.tensor(np.vstack(all_obs), dtype=torch.float32, device=self.device)
                    actions_b = torch.tensor(np.concatenate(all_actions), dtype=torch.long, device=self.device)
                    rewards_b = torch.tensor(np.concatenate(all_rewards), dtype=torch.float32, device=self.device)
                    dones_b = torch.tensor(np.concatenate(all_dones), dtype=torch.float32, device=self.device)
                    next_obs_b = torch.tensor(np.vstack(all_next_obs), dtype=torch.float32, device=self.device)

                    # Worker前向传播
                    out = model.workers(obs_b)
                    log_probs = torch.log(torch.clamp(out['action_probs'], min=1e-8))
                    act_logp = log_probs.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                    # 计算returns
                    with torch.no_grad():
                        next_vals = model.workers(next_obs_b)['value']
                    targets = rewards_b + self.gamma * next_vals * (1.0 - dones_b)

                    # Worker优势
                    w_adv = targets - out['value']
                    std_w = w_adv.std(unbiased=False)
                    if torch.isfinite(std_w) and float(std_w.item()) > 1e-8:
                        w_adv = (w_adv - w_adv.mean()) / (std_w + 1e-8)
                    else:
                        w_adv = w_adv - w_adv.mean()

                    # Worker损失
                    w_policy_loss = -(act_logp * w_adv.detach()).mean()
                    w_value_loss = torch.nn.functional.mse_loss(out['value'], targets.detach())
                    w_entropy = -(out['action_probs'] * log_probs).sum(dim=1).mean()
                    w_total_loss = w_policy_loss + 0.5 * w_value_loss - self.entropy_coef_worker * w_entropy

                    opt_workers.zero_grad()
                    w_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.workers.parameters(), 1.0)
                    opt_workers.step()

                    worker_loss = float(w_total_loss.item())
                    worker_metrics = {
                        'policy_loss': float(w_policy_loss.item()),
                        'value_loss': float(w_value_loss.item()),
                        'entropy': float(w_entropy.item()),
                    }

                # 更新训练步数
                self.training_step += 1

                # 总损失和指标
                total_loss = manager_loss + worker_loss
                metrics = {
                    'manager_loss': manager_loss,
                    'worker_loss': worker_loss,
                    **{f'manager_{k}': v for k, v in manager_metrics.items()},
                    **{f'worker_{k}': v for k, v in worker_metrics.items()},
                }

                return total_loss, metrics

            except Exception as e:
                print(f"⚠️ train_fn错误: {e}")
                import traceback
                traceback.print_exc()
                return 0.0, {}

        return train_fn
