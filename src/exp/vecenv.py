import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Tuple
import numpy as np


def _worker(remote: Connection, env_config: Dict[str, Any], max_tasks: int) -> None:
    """Subprocess worker: owns a DynamicWarehouseEnv and executes commands.

    Commands:
    - 'init': initialize env (idempotent)
    - 'features': return dict with state/task features/pending task ids/requires flags/free pids
    - 'assign_and_step', decisions: List[Tuple[int,pid,int,task_id]] for this env
      Performs assignment, computes heuristic actions and steps env once.
      Returns dict with 'step_reward' and 'next_state_vec'.
    - 'close': terminate
    """
    try:
        from env.dynamic_warehouse_env import DynamicWarehouseEnv, TaskStatus
        from exp.obs import get_global_state, get_task_features
        from exp.actions import smart_navigate, find_adjacent_accessible_position, convert_to_dynamic_actions
    except Exception as e:
        remote.send({'error': f'import_failed: {e}'})
        remote.close(); return

    env = None

    def _ensure_env():
        nonlocal env
        if env is None:
            env = DynamicWarehouseEnv(dict(env_config))
            # Default speed function mimicking run_experiments.speed_fn
            def _speed_fn(e):
                speeds = {}
                for p in e.pickers:
                    t = getattr(p, 'current_task', None)
                    base = float(getattr(p, 'speed', 1.0))
                    if getattr(p.type, 'name', '') == 'FORKLIFT':
                        speeds[p.id] = base
                    else:
                        if p.carrying_items and t is not None:
                            try:
                                eff = float(e._compute_movement_efficiency(p, t))
                            except Exception:
                                eff = 1.0
                        else:
                            eff = 1.0
                        speeds[p.id] = base * eff
                return speeds
            env.set_speed_function(_speed_fn)
            env.reset()

    while True:
        cmd, data = remote.recv()
        if cmd == 'init':
            try:
                _ensure_env()
                remote.send({'ok': True})
            except Exception as e:
                remote.send({'error': str(e)})
        elif cmd == 'features':
            try:
                _ensure_env()
                state_vec = get_global_state(env)
                task_feats = get_task_features(env, max_tasks=max_tasks, pending_only=True)
                pending = [t for t in env.task_pool if getattr(t, 'status', None) == TaskStatus.PENDING][:max_tasks]
                task_ids = [getattr(t, 'task_id', -1) for t in pending]
                requires = [bool(getattr(t, 'requires_car', False)) for t in pending]
                free_pids = [i for i, p in enumerate(env.pickers) if getattr(p, 'current_task', None) is None and len(p.carrying_items) == 0]
                remote.send({
                    'state_vec': np.array(state_vec, dtype=np.float32),
                    'task_feats': np.array(task_feats, dtype=np.float32),
                    'task_ids': np.array(task_ids, dtype=np.int64),
                    'requires': np.array(requires, dtype=np.bool_),
                    'free_pids': np.array(free_pids, dtype=np.int64),
                })
            except Exception as e:
                remote.send({'error': str(e)})
        elif cmd == 'assign_and_step':
            try:
                _ensure_env()
                # decisions: List[Tuple[pid, task_id]]
                decisions: List[Tuple[int, int]] = data or []
                # Apply assignments
                if decisions:
                    # Map task_id -> task
                    pending_map = {getattr(t, 'task_id', -1): t for t in env.task_pool if getattr(t, 'status', None) == TaskStatus.PENDING}
                    for pid, tid in decisions:
                        t = pending_map.get(int(tid))
                        if t is None:
                            continue
                        if not (0 <= pid < len(env.pickers)):
                            continue
                        t.status = TaskStatus.ASSIGNED
                        t.assigned_picker = int(pid)
                        env.pickers[int(pid)].current_task = t
                # Heuristic actions identical to trainers
                actions: Dict[int, int] = {}
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
                next_state_vec = get_global_state(env)
                remote.send({
                    'step_reward': float(step_reward),
                    'next_state_vec': np.array(next_state_vec, dtype=np.float32),
                })
            except Exception as e:
                remote.send({'error': str(e)})
        elif cmd == 'close':
            try:
                remote.send({'ok': True})
            except Exception:
                pass
            break
        else:
            remote.send({'error': f'unknown_cmd:{cmd}'})
    remote.close()


class SubprocVecEnv:
    def __init__(self, n_envs: int, env_config: Dict[str, Any], max_tasks: int):
        self.n = int(max(1, n_envs))
        self.max_tasks = int(max_tasks)
        self.parent_conns: List[Connection] = []
        self.procs: List[mp.Process] = []
        ctx = mp.get_context('spawn')
        for _ in range(self.n):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child_conn, env_config, self.max_tasks), daemon=True)
            p.start()
            self.parent_conns.append(parent_conn)
            self.procs.append(p)
            parent_conn.send(('init', None))
            _ = parent_conn.recv()

    def get_features(self) -> List[Dict[str, Any]]:
        for pc in self.parent_conns:
            pc.send(('features', None))
        outs = [pc.recv() for pc in self.parent_conns]
        return outs

    def step_with_decisions(self, decisions: List[List[Tuple[int, int]]]) -> List[Dict[str, Any]]:
        # decisions[i] applies to env i
        for i, pc in enumerate(self.parent_conns):
            pc.send(('assign_and_step', decisions[i] if i < len(decisions) else []))
        outs = [pc.recv() for pc in self.parent_conns]
        return outs

    def close(self):
        for pc in self.parent_conns:
            try:
                pc.send(('close', None))
            except Exception:
                pass
        for p in self.procs:
            try:
                p.join(timeout=0.1)
            except Exception:
                pass
