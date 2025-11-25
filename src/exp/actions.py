import numpy as np


def convert_to_dynamic_actions(actions, env=None, input_space: str = 'env'):
    """将上层（策略/DQN/启发式）动作转换为环境动作索引。

    输入空间说明:
    - input_space='env'：输入已是环境索引（0..3=UP/DOWN/LEFT/RIGHT，4=IDLE，5/6当作IDLE）。
    - input_space='dqn'：输入为DQN索引（[IDLE=0, UP=1, DOWN=2, LEFT=3, RIGHT=4, PICK=5, DROP=6]）。

    输出：环境动作索引（0..3 移动，4=IDLE；5/6 归一到 4 用于在相邻时触发拣/投）。
    """
    dynamic_actions = {}
    if input_space == 'dqn':
        # DQN -> Env 映射
        # 0(IDLE)->4, 1..4(UP..RIGHT)->0..3, 5/6(PICK/DROP)->4
        mapping = {0: 4, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4}
        for pid, a in actions.items():
            a = int(a)
            dynamic_actions[pid] = mapping.get(a, 4)
    else:
        # 视为已是环境动作，唯一步骤：5/6→4
        for pid, a in actions.items():
            a = int(a)
            if a in (5, 6):
                a = 4
            dynamic_actions[pid] = a
    return dynamic_actions


def smart_navigate(picker, target_pos, env):
    """使用BFS的一步导航：找到最短路径的第一步。

    使用简化BFS(限制搜索深度)找到最短路径,返回第一步的方向。
    为避免过慢,限制BFS搜索范围。

    返回: action_int（0..3=UP/DOWN/LEFT/RIGHT，4=IDLE）
    """
    from collections import deque

    start = (picker.x, picker.y)
    goal = target_pos

    if start == goal:
        return 4  # 已到达

    # BFS查找最短路径(限制深度避免太慢)
    queue = deque([(start, [])])  # (位置, 路径)
    visited = {start}
    max_depth = 50  # 限制搜索深度

    while queue:
        (x, y), path = queue.popleft()

        # 限制搜索深度
        if len(path) > max_depth:
            break

        # 尝试4个方向
        for action, (dx, dy) in [(0, (0, -1)), (1, (0, 1)), (2, (-1, 0)), (3, (1, 0))]:
            nx, ny = x + dx, y + dy

            # 边界和货架检查
            if not (0 <= nx < env.width and 0 <= ny < env.height):
                continue
            if env.grid[ny, nx] == 2:  # 货架
                continue
            if (nx, ny) in visited:
                continue

            new_path = path + [action]
            visited.add((nx, ny))

            # 找到目标
            if (nx, ny) == goal:
                # 返回路径的第一步
                return new_path[0] if new_path else 4

            queue.append(((nx, ny), new_path))

    # 无法找到路径,尝试朝目标靠近(降级方案)
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]

    def can_move_to(x, y):
        if not (0 <= x < env.width and 0 <= y < env.height):
            return False
        return env.grid[y, x] != 2

    # 尝试主方向
    candidates = []
    if abs(dx) > abs(dy):
        if dx > 0 and can_move_to(start[0] + 1, start[1]):
            candidates.append(3)  # RIGHT
        elif dx < 0 and can_move_to(start[0] - 1, start[1]):
            candidates.append(2)  # LEFT
        if dy > 0 and can_move_to(start[0], start[1] + 1):
            candidates.append(1)  # DOWN
        elif dy < 0 and can_move_to(start[0], start[1] - 1):
            candidates.append(0)  # UP
    else:
        if dy > 0 and can_move_to(start[0], start[1] + 1):
            candidates.append(1)  # DOWN
        elif dy < 0 and can_move_to(start[0], start[1] - 1):
            candidates.append(0)  # UP
        if dx > 0 and can_move_to(start[0] + 1, start[1]):
            candidates.append(3)  # RIGHT
        elif dx < 0 and can_move_to(start[0] - 1, start[1]):
            candidates.append(2)  # LEFT

    # 尝试任意可行方向
    if not candidates:
        for action, (ddx, ddy) in [(0, (0, -1)), (1, (0, 1)), (2, (-1, 0)), (3, (1, 0))]:
            if can_move_to(start[0] + ddx, start[1] + ddy):
                candidates.append(action)

    return candidates[0] if candidates else 4


def find_adjacent_accessible_position(env, shelf_pos, picker_pos):
    """找到目标周围一个可达的相邻格（非货架即可）。"""
    x, y = shelf_pos
    best = None
    best_dist = 1e9
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < env.width and 0 <= ny < env.height:
            if env.grid[ny, nx] != 2:  # 非货架
                d = aisle_distance(env, picker_pos, (nx, ny))
                if d < best_dist:
                    best = (nx, ny)
                    best_dist = d
    return best


def get_valid_actions(env, picker):
    """返回动作有效性掩码（长度7的0/1数组）。
    - 在动态环境中：PICK/DROP 只有在与货架/站点相邻（曼哈顿=1）时才为1。
    """
    valid = [1, 1, 1, 1, 1, 0, 0]
    # PICK
    can_pick = False
    if getattr(picker, 'current_task', None) is not None and not picker.carrying_items:
        t = picker.current_task
        if t.shelf_id is not None and t.shelf_id < len(env.shelves):
            sh = env.shelves[t.shelf_id]
            if abs(picker.x - sh['x']) + abs(picker.y - sh['y']) == 1:
                can_pick = True
    valid[5] = 1 if can_pick else 0
    # DROP
    can_drop = False
    if picker.carrying_items and getattr(picker, 'current_task', None) is not None:
        t = picker.current_task
        if t.station_id is not None and t.station_id < len(env.stations):
            st = env.stations[t.station_id]
            if abs(picker.x - st['x']) + abs(picker.y - st['y']) == 1:
                can_drop = True
    valid[6] = 1 if can_drop else 0
    return valid


def get_guided_exploration_action(env, picker, epsilon=0.5):
    """引导式探索：相邻时优先尝试 PICK/DROP；否则朝目标移动；其余随机。"""
    valid_actions = get_valid_actions(env, picker)
    # 相邻拣
    if getattr(picker, 'current_task', None) and not picker.carrying_items:
        t = picker.current_task
        if t.shelf_id is not None and t.shelf_id < len(env.shelves):
            sh = env.shelves[t.shelf_id]
            if abs(picker.x - sh['x']) + abs(picker.y - sh['y']) == 1 and valid_actions[5]:
                return 5
    # 携货相邻投
    if picker.carrying_items and getattr(picker, 'current_task', None):
        t = picker.current_task
        if t.station_id is not None and t.station_id < len(env.stations):
            st = env.stations[t.station_id]
            if abs(picker.x - st['x']) + abs(picker.y - st['y']) == 1 and valid_actions[6]:
                return 6
    return None


def aisle_distance(env, start, goal):
    """Manhattan距离近似（快速O(1)计算，忽略货架障碍）。

    原实现：BFS搜索，精确但慢（O(W*H)）
    新实现：Manhattan距离，快速但近似（O(1)）
    根据RL研究，Manhattan距离在网格世界奖励塑形中效果很好
    """
    sx, sy = start
    gx, gy = goal
    # 相同位置
    if (sx, sy) == (gx, gy):
        return 0
    # 边界检查
    W, H = env.width, env.height
    if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
        return 10**9
    # 货架检查（起点或终点在货架上视为不可达）
    if env.grid[sy, sx] == 2:
        return 10**9
    if env.grid[gy, gx] == 2:
        return 10**9
    # Manhattan距离：|x1-x2| + |y1-y2|
    return abs(gx - sx) + abs(gy - sy)
