import numpy as np


def get_agent_observation(env, picker, include_global=False):
    """构造单体观测（与旧版保持近似结构）。
    include_global=True 时追加全局摘要维度。
    返回 np.float32 向量。
    """
    obs = []
    # 自身
    has_task = getattr(picker, 'current_task', None) is not None
    obs.extend([
        picker.x / env.width,
        picker.y / env.height,
        len(picker.carrying_items) / 10.0,
        1.0 if has_task else 0.0,
        picker.battery / 100.0
    ])
    # 局部 5x5
    R = 2
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            x, y = picker.x + dx, picker.y + dy
            if 0 <= x < env.width and 0 <= y < env.height:
                val = env.grid[y, x] / 4.0
            else:
                val = -1.0
            obs.append(val)
    # 目标相对位移（4维）
    if has_task:
        t = picker.current_task
        if not picker.carrying_items and t.shelf_id is not None and t.shelf_id < len(env.shelves):
            sh = env.shelves[t.shelf_id]
            obs.extend([(sh['x'] - picker.x) / env.width, (sh['y'] - picker.y) / env.height, 1.0, 1.0])
        elif picker.carrying_items and t.station_id is not None and t.station_id < len(env.stations):
            st = env.stations[t.station_id]
            obs.extend([(st['x'] - picker.x) / env.width, (st['y'] - picker.y) / env.height, 1.0, 1.0])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])
    else:
        obs.extend([0.0, 0.0, 0.0, 0.0])

    # 最近 3 个其他拣货员位置
    others = []
    for other in env.pickers:
        if other is picker:
            continue
        d = abs(other.x - picker.x) + abs(other.y - picker.y)
        others.append((d, other))
    others.sort(key=lambda x: x[0])
    for i in range(3):
        if i < len(others):
            _, o = others[i]
            obs.extend([(o.x - picker.x) / env.width, (o.y - picker.y) / env.height])
        else:
            obs.extend([0.0, 0.0])

    if include_global:
        # 简要全局信息（5维）
        obs.extend([
            len(env.task_pool) / 20.0,
            getattr(env, 'total_tasks_completed', 0) / 100.0,
            env.current_time / 1000.0,
            sum(1 for p in env.pickers if p.current_task is not None) / max(1, len(env.pickers)),
            0.0
        ])

    expected = 45 if include_global else 40
    while len(obs) < expected:
        obs.append(0.0)
    return np.array(obs[:expected], dtype=np.float32)


def get_global_state(env):
    """全局状态向量（用于 Manager 或分析）。"""
    state = []
    # 时间
    if hasattr(env, 'order_generator'):
        hour = env.current_time % 24
        state.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            env.order_generator.get_arrival_rate(env.current_time) / 100.0,
        ])
    else:
        state.extend([0.0, 0.0, 0.0])
    # 任务池
    state.append(len(env.task_pool) / 100.0)
    for w in ['forklift_only', 'heavy', 'medium', 'light']:
        state.append(len([t for t in env.task_pool if getattr(t, 'weight_class', None) == w and t.status.name == 'PENDING']) / 50.0)
    # 区域×重量摘要
    for zone in range(4):
        for w in ['forklift_only', 'heavy', 'medium', 'light']:
            ts = [t for t in env.task_pool if t.zone == zone and getattr(t, 'weight_class', None) == w and t.status.name == 'PENDING']
            cnt = len(ts)
            avg_pri = float(np.mean([t.priority for t in ts])) if ts else 0.0
            avg_rem = float(np.mean([t.deadline - env.current_time for t in ts])) if ts else 0.0
            state.extend([cnt / 20.0, avg_pri, 0.5, avg_rem])
    # 拣货员统计
    busy = sum(1 for p in env.pickers if p.current_task is not None)
    forklifts = sum(1 for p in env.pickers if getattr(p.type, 'name', '') == 'FORKLIFT')
    forklifts_busy = sum(1 for p in env.pickers if getattr(p.type, 'name', '') == 'FORKLIFT' and p.current_task is not None)
    state.extend([
        busy / max(1, len(env.pickers)),
        np.mean([p.battery for p in env.pickers]) / 100 if env.pickers else 1.0,
        forklifts_busy / max(1, forklifts) if forklifts > 0 else 0.0,
    ])
    # 区域负载占比
    total = sum(env.zone_loads)
    for z in range(4):
        state.append(env.zone_loads[z] / max(1, total))
    # 绩效
    state.extend([
        getattr(env, 'total_orders_completed', 0) / max(1, getattr(env, 'total_orders_received', 1)),
        getattr(env, 'on_time_completions', 0) / max(1, getattr(env, 'total_tasks_completed', 1)),
    ])
    return np.array(state, dtype=np.float32)


def get_task_features(env, max_tasks=20, pending_only: bool = True):
    """将任务池编码成固定维度特征。

    - pending_only=True 时，仅取 PENDING 任务；否则按池顺序截取前 max_tasks 个。
    - 输出 shape = [max_tasks, 5]，不足用零填充。
    """
    feats = []
    if pending_only:
        try:
            from env.dynamic_warehouse_env import TaskStatus  # type: ignore
            tasks = [t for t in env.task_pool if getattr(t, 'status', None) == TaskStatus.PENDING][:max_tasks]
        except Exception:
            tasks = list(env.task_pool)[:max_tasks]
    else:
        tasks = list(env.task_pool)[:max_tasks]
    for t in tasks:
        shx, shy = 0.5, 0.5
        if t.shelf_id is not None and t.shelf_id < len(env.shelves):
            sh = env.shelves[t.shelf_id]
            shx, shy = sh['x'] / env.width, sh['y'] / env.height
        feats.append([
            shx, shy,
            1.0,  # 大小占位
            t.priority,
            max(0.0, t.deadline - env.current_time)
        ])
    while len(feats) < max_tasks:
        feats.append([0.0, 0.0, 0.0, 0.0, 0.0])
    return np.array(feats[:max_tasks], dtype=np.float32)
