from typing import List, Tuple, Dict, Optional, Any
from collections import namedtuple
import numpy as np
from baselines.rule_based import RuleBasedAssigner, RoutingStrategy
from exp.actions import aisle_distance, find_adjacent_accessible_position


Assignment = namedtuple('Assignment', ['picker_id', 'order_id', 'route', 'estimated_time'])


class SimpleReturnAssigner(RuleBasedAssigner):
    """简化 Return 策略的分配器：按回字形思路生成路径（占位）。"""
    def __init__(self):
        super().__init__(strategy=RoutingStrategy.RETURN)

    def _return_route(self, start_pos, items, warehouse_grid):
        if not items:
            return []
        return list(items)


class FixedOptimalAssigner:
    """简单贪心：按距离最近的任务进行分配（用于动态环境演示）。"""
    def assign_orders(self, pickers: List[Dict], orders: List[Dict], warehouse_grid=None):
        if not pickers or not orders:
            return []
        ass = []
        used_p, used_o = set(), set()
        while len(ass) < min(len(pickers), len(orders)):
            best = (1e9, None, None)
            for i, p in enumerate(pickers):
                if p['picker_id'] in used_p:
                    continue
                for j, o in enumerate(orders):
                    if o['order_id'] in used_o or not o['items']:
                        continue
                    x0, y0 = p['x'], p['y']
                    x1, y1 = o['items'][0]
                    d = abs(x1 - x0) + abs(y1 - y0)
                    if d < best[0]:
                        best = (d, i, j)
            if best[1] is None:
                break
            pi, oj = best[1], best[2]
            pick = pickers[pi]
            ordx = orders[oj]
            ass.append(Assignment(pick['picker_id'], ordx['order_id'], list(ordx['items']), best[0] * 2))
            used_p.add(pick['picker_id'])
            used_o.add(ordx['order_id'])
        return ass


def assign_tasks_dynamic(env, forklift_heavy_bonus: float = 5.0):
    """为动态环境进行一次启发式任务分配：
    - 空闲且未携货的拣货员：从 PENDING 任务中挑一个可执行的（heavy→仅叉车）
    - 评分：距离(拣货员→货架) + 2 * 紧急度(1/剩余时间)
    - 将选中的任务标记为 ASSIGNED，并写入 picker.current_task

    返回：分配数量。
    """
    from env.dynamic_warehouse_env import TaskStatus, PickerType
    assigned = 0
    for pid, p in enumerate(env.pickers):
        # 仅给空闲且未携货的拣货员分配
        if getattr(p, 'current_task', None) is not None or len(p.carrying_items) > 0:
            continue
        best = None
        best_score = 1e9
        for t in list(getattr(env, 'task_pool', [])):
            if getattr(t, 'status', None) != TaskStatus.PENDING:
                continue
            if getattr(t, 'requires_car', False) and p.type != PickerType.FORKLIFT:
                continue
            if t.shelf_id is None or t.shelf_id >= len(env.shelves):
                continue
            sh = env.shelves[t.shelf_id]
            # 距离改为通道内到货架相邻可达格的BFS距离
            adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
            dist = aisle_distance(env, (p.x, p.y), adj) if adj is not None else 10**9
            rem = max(1e-3, (t.deadline - getattr(env, 'current_time', 0.0)))
            # 高 value 优先：score 减去 value 权重
            # 使用当前衰减后的任务价值进行打分（越大越优，因此作为负项）
            try:
                tval = float(getattr(env, 'get_task_decayed_value')(t)) if hasattr(env, 'get_task_decayed_value') else float(getattr(t, 'items', [{}])[0].get('value', 0))
            except Exception:
                tval = 0.0
            score = dist + 2.0 * (1.0 / rem) - 0.05 * tval
            # 叉车优先重单（heavy）
            if p.type.name == 'FORKLIFT' and getattr(t, 'requires_car', False):
                score -= forklift_heavy_bonus
            if score < best_score:
                best_score = score
                best = t
        if best is not None:
            best.status = TaskStatus.ASSIGNED
            best.assigned_picker = pid
            p.current_task = best
            assigned += 1
    return assigned


def assign_tasks_dynamic_nl_hmarl(
    env,
    *,
    value_weight: float = 0.05,
    dist_weight: float = 1.0,
    urg_weight: float = 2.0,
    forklift_heavy_bonus: float = 5.0,
    eta_by_nest: Optional[Dict[int, float]] = None,
) -> int:
    """Nested-Logit style manager for dynamic assignment.

    - Nest definition: by task.zone (0..3) computed by the environment layout
    - Per-task utility v = value_weight * decayed_value - dist_weight * distance - urg_weight * (1/rem)
      If picker is a forklift and task.requires_car, add forklift_heavy_bonus to v
    - For each free picker, compute nest probabilities and within-nest probabilities (nested logit)
      using eta per nest (default 1.0), then pick the task with highest P(nest)*P(task|nest)

    Returns number of assignments made this step.
    """
    from env.dynamic_warehouse_env import TaskStatus, PickerType

    def _logsumexp(vals, scale: float = 1.0) -> float:
        if not vals:
            return float('-inf')
        m = max(vals)
        if not np.isfinite(m):
            return m
        s = sum(np.exp((v - m) / max(1e-6, scale)) for v in vals)
        if s <= 0:
            return float('-inf')
        return m / max(1e-6, scale) + np.log(s)

    free_ids = _free_picker_ids(env)
    if not free_ids:
        return 0

    tasks_all = _pending_tasks(env)
    if not tasks_all:
        return 0

    # Per picker greedy assignment using NL probabilities
    assigned = 0
    # Ensure eta per nest
    default_eta = 1.0
    eta_by_nest = eta_by_nest or {}

    for pid in free_ids:
        p = env.pickers[pid]

        # Filter tasks still pending and feasible for this picker
        tasks = [t for t in tasks_all if t.status == TaskStatus.PENDING and (not t.requires_car or p.type == PickerType.FORKLIFT)]
        if not tasks:
            continue

        # Utility per task for this picker
        task_utils: Dict[int, float] = {}
        tasks_by_nest: Dict[int, List[Any]] = {}
        for t in tasks:
            try:
                sh = env.shelves[t.shelf_id]
            except Exception:
                continue
            # BFS-based aisle distance to an adjacent accessible cell
            adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
            dist = aisle_distance(env, (p.x, p.y), adj) if adj is not None else 10**6
            # decayed value and urgency
            try:
                tval = float(env.get_task_decayed_value(t))
            except Exception:
                tval = float(getattr(t, 'items', [{}])[0].get('value', 0))
            rem = max(1e-3, (t.deadline - getattr(env, 'current_time', 0.0)))
            util = value_weight * tval - dist_weight * float(dist) - urg_weight * (1.0 / rem)
            if p.type == PickerType.FORKLIFT and getattr(t, 'requires_car', False):
                util += forklift_heavy_bonus
            task_utils[getattr(t, 'task_id', id(t))] = util
            # Nest by forklift requirement: 1 if requires_car else 0
            nid = 1 if bool(getattr(t, 'requires_car', False)) else 0
            tasks_by_nest.setdefault(nid, []).append(t)

        if not task_utils:
            continue

        # Compute nested-logit probabilities per nest, then within nest
        nest_scores: Dict[int, float] = {}
        nest_exp: Dict[int, float] = {}
        for nest_id, ts in tasks_by_nest.items():
            eta = float(eta_by_nest.get(nest_id, default_eta))
            vals = [task_utils.get(getattr(t, 'task_id', id(t)), -1e9) for t in ts]
            # U_m = eta * log sum_j exp(v_j / eta)
            lse = _logsumexp(vals, scale=eta)
            U_m = eta * lse if np.isfinite(lse) else -1e9
            nest_scores[nest_id] = U_m
        # Upper-level softmax over nests
        # Stabilize
        if not nest_scores:
            continue
        max_um = max(nest_scores.values())
        denom_nest = sum(np.exp(Um - max_um) for Um in nest_scores.values())
        if denom_nest <= 0:
            continue
        for nid, Um in nest_scores.items():
            nest_exp[nid] = np.exp(Um - max_um) / denom_nest

        # For each task, compute P(task) = P(nest) * P(task|nest)
        task_probs: Dict[int, float] = {}
        for nid, ts in tasks_by_nest.items():
            eta = float(eta_by_nest.get(nid, default_eta))
            vals = [task_utils.get(getattr(t, 'task_id', id(t)), -1e9) for t in ts]
            if not vals:
                continue
            mval = max(vals)
            exps = [np.exp((v - mval) / max(1e-6, eta)) for v in vals]
            denom = sum(exps)
            if denom <= 0:
                continue
            for t, ev in zip(ts, exps):
                tid = getattr(t, 'task_id', id(t))
                p_in_nest = ev / denom
                task_probs[tid] = nest_exp.get(nid, 0.0) * p_in_nest

        if not task_probs:
            continue

        # Pick best remaining task for this picker
        # Ensure we do not select tasks already assigned in this loop
        best_tid = max(task_probs.items(), key=lambda kv: kv[1])[0]
        # Find the actual task object
        chosen = None
        for t in tasks_all:
            if getattr(t, 'task_id', id(t)) == best_tid and t.status == TaskStatus.PENDING:
                chosen = t
                break
        if chosen is None:
            continue
        # Assign
        chosen.status = TaskStatus.ASSIGNED
        chosen.assigned_picker = pid
        p.current_task = chosen
        assigned += 1
        # Remove from local view to avoid duplicate assignment
        try:
            tasks_all.remove(chosen)
        except ValueError:
            pass

    return assigned


def _free_picker_ids(env):
    ids = []
    for pid, p in enumerate(env.pickers):
        if getattr(p, 'current_task', None) is None and len(p.carrying_items) == 0:
            ids.append(pid)
    return ids


def _pending_tasks(env):
    from env.dynamic_warehouse_env import TaskStatus
    return [t for t in getattr(env, 'task_pool', []) if getattr(t, 'status', None) == TaskStatus.PENDING]


def assign_tasks_dynamic_s_shape(env, forklift_heavy_first: bool = True, value_weight: float = 0.05):
    """按 S-Shape 序（行蛇形）对任务排序后为空闲拣货员依次分配。
    - heavy→仅叉车
    - 简化实现：按 (row, x 或 -x) 排序全局任务队列，再顺序分配给空闲拣货员
    返回：分配数量
    """
    from env.dynamic_warehouse_env import TaskStatus, PickerType
    free_ids = _free_picker_ids(env)
    if not free_ids:
        return 0
    # 生成 S-Shape key
    def s_key(t):
        sh = env.shelves[t.shelf_id]
        y, x = sh['y'], sh['x']
        serp = x if (y % 2 == 1) else -x
        return (y, serp)
    tasks = sorted(_pending_tasks(env), key=s_key)
    assigned = 0
    for pid in free_ids:
        p = env.pickers[pid]
        chosen = None
        if forklift_heavy_first and p.type == PickerType.FORKLIFT:
            # 叉车优先选择 heavy 任务中 value 最大的（仍保持 S-Shape 排序的基础上）
            best_val = -1.0
            for t in tasks:
                if t.status != TaskStatus.PENDING:
                    continue
                if not getattr(t, 'requires_car', False):
                    continue
                try:
                    tval = float(env.get_task_decayed_value(t)) if hasattr(env, 'get_task_decayed_value') else float(getattr(t, 'items', [{}])[0].get('value', 0))
                except Exception:
                    tval = 0.0
                if tval > best_val:
                    best_val = tval
                    chosen = t
        if chosen is None:
            # 回退：按 S-Shape 顺序选择第一个可执行任务
            best_score = -1e9
            for t in tasks:
                if t.status != TaskStatus.PENDING:
                    continue
                if t.requires_car and p.type != PickerType.FORKLIFT:
                    continue
                try:
                    tval = float(env.get_task_decayed_value(t)) if hasattr(env, 'get_task_decayed_value') else float(getattr(t, 'items', [{}])[0].get('value', 0))
                except Exception:
                    tval = 0.0
                # 以 value 为主的打分（也可加上负距离项）
                score = value_weight * tval
                if score > best_score:
                    best_score = score
                    chosen = t
        if chosen is None:
            continue
        chosen.status = TaskStatus.ASSIGNED
        chosen.assigned_picker = pid
        p.current_task = chosen
        tasks.remove(chosen)
        assigned += 1
    return assigned


def assign_tasks_dynamic_return(env, forklift_heavy_first: bool = True, value_weight: float = 0.05):
    """按 Return 思路：优先小列（x）到大列（x），列内按 y 排序。
    返回：分配数量
    """
    from env.dynamic_warehouse_env import TaskStatus, PickerType
    free_ids = _free_picker_ids(env)
    if not free_ids:
        return 0
    def r_key(t):
        sh = env.shelves[t.shelf_id]
        return (sh['x'], sh['y'])
    tasks = sorted(_pending_tasks(env), key=r_key)
    assigned = 0
    for pid in free_ids:
        p = env.pickers[pid]
        chosen = None
        if forklift_heavy_first and p.type == PickerType.FORKLIFT:
            best_val = -1.0
            for t in tasks:
                if t.status != TaskStatus.PENDING:
                    continue
                if not getattr(t, 'requires_car', False):
                    continue
                try:
                    tval = float(env.get_task_decayed_value(t)) if hasattr(env, 'get_task_decayed_value') else float(getattr(t, 'items', [{}])[0].get('value', 0))
                except Exception:
                    tval = 0.0
                if tval > best_val:
                    best_val = tval
                    chosen = t
        if chosen is None:
            best_score = -1e9
            for t in tasks:
                if t.status != TaskStatus.PENDING:
                    continue
                if t.requires_car and p.type != PickerType.FORKLIFT:
                    continue
                try:
                    tval = float(env.get_task_decayed_value(t)) if hasattr(env, 'get_task_decayed_value') else float(getattr(t, 'items', [{}])[0].get('value', 0))
                except Exception:
                    tval = 0.0
                score = value_weight * tval
                if score > best_score:
                    best_score = score
                    chosen = t
        if chosen is None:
            continue
        chosen.status = TaskStatus.ASSIGNED
        chosen.assigned_picker = pid
        p.current_task = chosen
        tasks.remove(chosen)
        assigned += 1
    return assigned


def assign_tasks_dynamic_optimal(env, value_weight: float = 0.05):
    """近似 Optimal（匈牙利简化：贪心最小距离匹配）。
    返回：分配数量
    """
    from env.dynamic_warehouse_env import TaskStatus, PickerType
    free_ids = _free_picker_ids(env)
    tasks = _pending_tasks(env)
    if not free_ids or not tasks:
        return 0
    # 构造候选对 (pid, tid) 与距离
    pairs = []
    for pid in free_ids:
        p = env.pickers[pid]
        for t in tasks:
            if t.status != TaskStatus.PENDING:
                continue
            if t.requires_car and p.type != PickerType.FORKLIFT:
                continue
            sh = env.shelves[t.shelf_id]
            adj = find_adjacent_accessible_position(env, (sh['x'], sh['y']), (p.x, p.y))
            d = aisle_distance(env, (p.x, p.y), adj) if adj is not None else 10**9
            try:
                tval = float(env.get_task_decayed_value(t)) if hasattr(env, 'get_task_decayed_value') else float(getattr(t, 'items', [{}])[0].get('value', 0))
            except Exception:
                tval = 0.0
            score = d - value_weight * tval
            pairs.append((score, pid, t))
    # 按距离排序，依次拿走未使用的 pid 与任务
    pairs.sort(key=lambda x: x[0])
    used_pid = set()
    used_tid = set()  # store task_id to avoid unhashable Task objects
    assigned = 0
    for d, pid, t in pairs:
        if pid in used_pid or getattr(t, 'task_id', id(t)) in used_tid:
            continue
        t.status = TaskStatus.ASSIGNED
        t.assigned_picker = pid
        env.pickers[pid].current_task = t
        used_pid.add(pid)
        used_tid.add(getattr(t, 'task_id', id(t)))
        assigned += 1
    return assigned
