import numpy as np
from collections import deque


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
    """基于通道最短路的贪心导航：
    - 计算当前位置与目标在通道上的BFS距离，选择能降低该距离的方向。
    - 屏蔽货架格（env.grid==2）。
    返回: action_int（0..3=UP/DOWN/LEFT/RIGHT，4=IDLE）
    """
    dx = target_pos[0] - picker.x
    dy = target_pos[1] - picker.y
    if dx == 0 and dy == 0:
        return 4  # 已在目标位置附近 → IDLE

    possible = []
    # 环境动作: 0=UP,1=DOWN,2=LEFT,3=RIGHT,4=IDLE
    dirs = [
        (0, 0, -1),  # UP
        (1, 0, 1),   # DOWN
        (2, -1, 0),  # LEFT
        (3, 1, 0),   # RIGHT
    ]
    # 仅基于静态障碍（货架）计算BFS距离
    cur_dist = aisle_distance(env, (picker.x, picker.y), target_pos)
    last_dir = getattr(picker, 'last_dir', None)
    opposite = {0:1, 1:0, 2:3, 3:2}
    for action, ddx, ddy in dirs:
        nx, ny = picker.x + ddx, picker.y + ddy
        if not (0 <= nx < env.width and 0 <= ny < env.height):
            continue
        # 货架格不走
        if env.grid[ny, nx] == 2:
            continue
        # 计算经通道到目标的BFS距离
        new_dist = aisle_distance(env, (nx, ny), target_pos)
        if new_dist < cur_dist:
            pri = 0
        elif new_dist == cur_dist:
            pri = 1
        else:
            pri = 2
        # 增加反摆动偏好：
        # - 与上一步相同方向优先（tie-break 更小）
        # - 与上一步相反方向劣后（tie-break 更大）
        tie = 1
        if last_dir is not None:
            if action == last_dir:
                tie = 0
            elif action == opposite.get(last_dir):
                tie = 2
        possible.append((pri, new_dist, tie, action))

    if possible:
        possible.sort(key=lambda x: (x[0], x[1], x[2]))
        return possible[0][3]
    return 4  # 无法更近 → IDLE


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
    """计算在通道（非货架格）内从 start 到 goal 的最短步数（BFS）。
    若不可达，返回一个大数（1e9）。
    """
    sx, sy = start
    gx, gy = goal
    if (sx, sy) == (gx, gy):
        return 0
    W, H = env.width, env.height
    # 不可作为起点/终点的格：仅屏蔽货架（2）；站点等可通过
    if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
        return 10**9
    if env.grid[sy, sx] == 2:
        return 10**9
    if env.grid[gy, gx] == 2:
        # 目标如果是货架格，外部调用应传来相邻的可达格；此处视为不可达
        return 10**9
    q = deque()
    q.append((sx, sy, 0))
    vis = [[False]*W for _ in range(H)]
    vis[sy][sx] = True
    while q:
        x, y, d = q.popleft()
        for dx, dy in ((0,1),(1,0),(0,-1),(-1,0)):
            nx, ny = x+dx, y+dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if vis[ny][nx]:
                continue
            if env.grid[ny, nx] == 2:  # 货架不可走
                continue
            if (nx, ny) == (gx, gy):
                return d+1
            vis[ny][nx] = True
            q.append((nx, ny, d+1))
    return 10**9
