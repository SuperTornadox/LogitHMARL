"""
三维仓库存储 + 多类型订单 + 双代理类型 的动态仓库环境（中文注释版）

【模块作用】
- 提供可交互的仓库仿真：布局（货架/通道/站点/充电桩）、多拣货员（含叉车）、动态订单→任务池、
  以及基于时间/拥堵/时效/能耗的奖励与约束。

【主要接口】
- env = DynamicWarehouseEnv(config)       # 构造
- state = env.reset()                      # 重置并返回全局状态向量（np.ndarray）
- next_state, rewards, dones, info = env.step(actions)
    · actions: Dict[int, int] 仅“动作值”（0..3=移动，4=IDLE 触发拣/投；5/6 等价于 4）
    · rewards: Dict[pid, float]
    · dones: Dict[pid, bool]
    · info: Dict（统计信息，如已完成任务、迟到任务）
- env.get_picker_state(pid)                # 返回单体局部观测
- env.plot(...) / env.render('...')        # 可视化（本文件提供 plot 简便接口）

【关键规则】
- 拣/投需与目标（货架/站点）曼哈顿距离=1（相邻一格），用 IDLE 触发尝试执行。
- 过道允许多个拣货员重叠；货架格不允许重叠（移动到已被他人占据的货架格会被阻止）。
- heavy 任务仅叉车可执行；移动/等待会消耗电量；拥堵/时效相关奖励/惩罚由环境计算。

【设计原则】
- 环境不内置任务分配（不做 Nest/Manager），分配策略放在实验/方法层；
- 通过明确的动作掩码与相邻规则，保障安全可执行；学习/规则只需在合法动作集合内优化。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import heapq
from collections import defaultdict
# 说明：run_experiments 会将 `${repo}/src` 加入 sys.path，
# 因此此处按包路径从 `env` 导入生成器模块即可。
# 兼容多种运行方式的导入：
# - 从项目根运行（run_experiments.py 会把 src 加到 sys.path）：from env.order_generation import ...
# - 作为包模块运行：python -m src.env.dynamic_warehouse_env → from .order_generation import ...
# - 直接运行文件：python src/env/dynamic_warehouse_env.py → 动态添加 src 到 sys.path 再导入
try:
    from env.order_generation import Order, NonHomogeneousPoissonOrderGenerator
except ModuleNotFoundError:
    try:
        from .order_generation import Order, NonHomogeneousPoissonOrderGenerator  # type: ignore
    except Exception:
        import os as _os, sys as _sys
        _SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _SRC_DIR not in _sys.path:
            _sys.path.insert(0, _SRC_DIR)
        from env.order_generation import Order, NonHomogeneousPoissonOrderGenerator  # type: ignore

# 动作空间
class ActionType(Enum):
    """动作类型枚举：0..3=移动，4=IDLE（触发拣/投），5/6=等价于 4（兼容训练侧）"""
    # 注意：为保持与现有脚本兼容，这里的编号保持不变：
    # 0..3 为移动，4 为 IDLE（在动态环境中用于尝试PICK/DROP）
    # 同时补充 5/6 两个枚举，作为 PICK/DROP 的别名（在本环境中等价于 IDLE 尝试执行任务）
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    IDLE = 4
    PICK = 5
    DROP = 6

# 任务状态
class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

# agent 类型(普通拣货员和车辆型拣货员)
class PickerType(Enum):
    """拣货员类型：REGULAR（普通）/ FORKLIFT（叉车）"""
    REGULAR = "regular"     # 普通拣货员（不能处理heavy）
    FORKLIFT = "forklift"   # 车辆型（heavy高效，mid/light也有优势）

@dataclass
class Task:
    """拣货-送货任务（支持三维存储与重量类型）

    字段：
    - task_id/order_id：任务/订单 ID
    - priority/deadline：优先级与截止时间（小时）
    - shelf_id/shelf_level：目标货架及层（层数仅用于层级偏置）
    - station_id：投递站点 ID
    - items：订单项（环境执行主要用位置信息）
    - zone：象限（0/1/2/3）
    - weight: 数值重量（合计，float）
    - weight_class：'heavy'|'medium'|'light'（由重量阈值映射得到）
    - requires_car：按重量阈值判断（heavy→True，仅叉车可拣）
    - status/assigned_picker/start_time/completion_time：状态与元数据
    """
    task_id: int
    order_id: int
    task_type: str                 # 'pick'
    priority: float
    deadline: float
    shelf_id: Optional[int]
    shelf_level: int               # 3D层级：0..levels-1
    station_id: Optional[int]
    items: List[Dict]
    zone: int
    weight: float                  # 数值重量（任务合计重量）
    weight_class: str              # 'heavy'|'medium'|'light'（由 weight 映射）
    requires_car: bool             # heavy需要车
    status: TaskStatus
    assigned_picker: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    base_value: int = 0            # 任务基础价值（value*quantity），用于衰减与罚没

@dataclass
class Picker:
    """拣货员/机器人（车辆型或普通型）

    关键属性：位置(x,y)、类型(type)、电量(battery)、容量(capacity)、
    是否携货(carrying_items)、当前任务(current_task) 与统计信息。
    """
    def __init__(self, picker_id: int, x: int, y: int, picker_type: PickerType):
        self.id = picker_id
        self.x = x
        self.y = y
        self.type = picker_type
        self.carrying_items: List[Dict] = []
        self.current_task: Optional[Task] = None
        self.battery = 100.0
        self.speed = 1.0
        self.capacity = 5 if picker_type == PickerType.FORKLIFT else 3
        self.idle_time = 0
        self.total_distance = 0
        self.completed_tasks = 0
        # 连续位置（用于可视化平滑移动）
        self.fx = float(x)
        self.fy = float(y)

class DynamicWarehouseEnv:
    """三维仓储 + 多类型订单 + 双代理类型 的动态环境实现

    构造参数（config）：
    - width/height：网格大小；levels_per_shelf：货架层数
    - n_pickers：拣货员数量；n_stations / n_charging_pads：站点/充电桩数量
    - time_step（秒）/ episode_duration（小时）：时间设置
    - order_config：订单流配置（base_rate/peak_hours/simulation_hours 等）

    主要方法：reset / step / get_picker_state / plot / stats
    """

    def __init__(self, config: Dict[str, Any]):
        # 基础配置
        self.width = config.get('width', 32)
        self.height = config.get('height', 32)
        self.n_pickers = config.get('n_pickers', 8)
        self.n_shelves = config.get('n_shelves', 64)
        self.n_stations = config.get('n_stations', 4)
        self.n_charging_pads = config.get('n_charging_pads', 2)
        self.levels_per_shelf = config.get('levels_per_shelf', 3)  # 三维层数
        # 叉车比例/最少数量（用于 reset 时确定叉车数量）
        self.forklift_ratio = config.get('forklift_ratio', 0.2)
        self.min_forklifts = config.get('min_forklifts', 0)
        # 速度/拥堵/载重对效率的影响配置
        self.speed_config = config.get('speed_config', {
            'base_speed': {'regular': 1.0, 'forklift': 1.2},  # 每步期望移动单元（可>1）
            'carry_alpha': {'regular': 1.0, 'forklift': 0.5}, # 载重减速强度（按 weight/100 缩放）
            'congestion_mult': 0.7,                           # 拥堵时速度乘子（<1 减速）
        })
        # 重量阈值（用于分类）。
        # - < medium → light
        # - [medium, heavy) → medium
        # - [heavy, forklift_only) → heavy（不强制叉车）
        # - >= forklift_only → forklift_only（仅叉车可拣）
        self.weight_thresholds = config.get('weight_thresholds', {
            'medium': 30.0,
            'heavy': 70.0,
            'forklift_only': 90.0,
        })

        # 时间配置
        self.time_step = config.get('time_step', 2.0)     # 2秒/步
        self.episode_duration = config.get('episode_duration', 1.0)  # 小时
        self.max_steps = int(self.episode_duration * 3600 / self.time_step)

        # 行为开关（保持方法无关）：环境不做分巢与启发式分配
        self.enable_nests = False
        self.allow_heuristic_assignment = False

        # 订单生成配置
        order_config = config.get('order_config', {
            'base_rate': 60,
            'peak_hours': [(9, 12), (14, 17), (19, 21)],
            'peak_multiplier': 1.6,
            'off_peak_multiplier': 0.7,
            'urgent_order_prob': 0.15,
            'bulk_order_prob': 0.10,
            'n_skus': 1000,
        })
        order_config['simulation_hours'] = self.episode_duration
        self.order_generator = NonHomogeneousPoissonOrderGenerator(order_config)

        # 奖励配置（按重量与代理类型调节效率）
        self.reward_config = {
            'idle_penalty': -0.05,
            'move_toward_target': 0.1,
            'congestion_penalty': -0.5,
            'battery_low_penalty': -1.0,
            'on_time_bonus': 5.0,
            'late_penalty': -5.0,
            # 按重量分配PICK/DROP基础奖励
            'pick_base': {'forklift_only': 4.0, 'heavy': 3.0, 'medium': 2.0, 'light': 1.0},
            'drop_base': {'forklift_only': 5.0, 'heavy': 4.0, 'medium': 2.5, 'light': 1.5},
            # 车辆型的效率加成（>1 更高效）
            'forklift_eff': {'forklift_only': 2.0, 'heavy': 1.8, 'medium': 1.2, 'light': 1.1},
            # 普通型的效率（forklift_only 不可行 -> 0）
            'regular_eff': {'forklift_only': 0.0, 'heavy': 1.0, 'medium': 1.0, 'light': 1.0},
            # 区域均衡
            'zone_balance_bonus': 0.3,
        }

        # 初始化布局与统计
        self._init_layout()
        self._init_statistics()

        # 运行期可控：策略可通过 hook/函数 动态调整环境参数（如速度覆盖、动态障碍等）
        self.control_hook = None  # type: Optional[callable]
        self.speed_fn = None      # type: Optional[callable]
        self.picker_speed_overrides: Dict[int, float] = {}
        self.last_speeds: Dict[int, float] = {}
        self.last_speed_components: Dict[int, Dict[str, float]] = {}

        # 网格（仅2D运动，3D在货架层级）
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        for shelf in self.shelves:
            self.grid[shelf['y'], shelf['x']] = 2  # SHELF
        for station in self.stations:
            self.grid[station['y'], station['x']] = 3  # STATION

        self.state_dim = self._calculate_state_dim()

        print(
            f"3D动态仓库初始化: {self.width}x{self.height}, 货架{len(self.shelves)} (levels={self.levels_per_shelf}), "
            f"拣货员{self.n_pickers}, 订单率≈{order_config['base_rate']}/h"
        )

    # ===== 运行期控制接口 =====
    def set_control_hook(self, hook):
        """注册一个在每个 step 调用的控制函数：hook(env, ctx) -> None
        策略可在 hook 内动态修改 env 的参数，如 per‑picker 速度、全局 speed_config 等。
        ctx 包含 'time' 与 'step' 字段。
        """
        self.control_hook = hook

    def set_speed_function(self, fn):
        """注册统一速度计算函数。
        要求 fn 为可调用，签名 fn(env) -> dict{id: speed(float>0)}。
        环境在每个 step 开始时调用它，并据此为所有拣货员设置本步速度覆盖；
        一旦注册，内部的拥堵/载重减速逻辑将不再应用于速度（仅用于奖励效率）。
        """
        if not callable(fn):
            raise TypeError("speed_function must be callable, with signature fn(env) -> {picker_id: speed}.")
        self.speed_fn = fn


    def set_picker_speed(self, picker_id: int, speed: float):
        """覆盖指定拣货员的基础速度（可随时调用，传入正数）。"""
        if 0 <= picker_id < len(self.pickers) and speed > 0:
            self.picker_speed_overrides[picker_id] = float(speed)

    def clear_picker_speed_override(self, picker_id: int = None):
        """清除速度覆盖（某个或全部）。"""
        if picker_id is None:
            self.picker_speed_overrides.clear()
        else:
            self.picker_speed_overrides.pop(picker_id, None)

    def _init_layout(self):
        """初始化静态布局：货架/通道/站点/充电桩

        - 隔行布局：奇数行铺货架、偶数行留通道；最右 3/4 处留一条竖向通道列
        - 最底两行：倒数第二行为通道、最后一行放置站点/充电桩（相互不相邻）
        输出：self.shelves / self.stations / self.charging_pads
        """
        # 货架行优先（row-by-row）生成（带行间过道）：
        # 从上到下依次处理行：奇数行(1-based)铺满货架，偶数行空为过道。
        # 即 y=1,3,5,... 为货架行；y=2,4,6,... 为通道。
        # 仍保留一圈边界 (x=0/width-1, y=0/height-1) 作为外边界过道。
        self.shelves: List[Dict] = []
        sid = 0
        # 垂直过道：在右侧中部生成一列过道（列索引 col_aisle），并保存以用于 Zone 划分与可视化
        col_aisle = max(1, min(self.width - 2, (3 * self.width) // 4))
        self.col_aisle = int(col_aisle)
        # 留出底部两行：y=height-2 为空行、y=height-1 放置站点/充电桩
        max_y_for_shelf = max(1, self.height - 2)
        # 水平中线用于 Zone 上/下划分
        mid_y = self.height // 2
        for y in range(1, max_y_for_shelf):
            # 判断是否为货架行：将 0-based y 转为 1-based 行号，再判断奇偶
            row_1based = y
            is_shelf_row = ((row_1based - 1) % 2 == 0)
            if not is_shelf_row:
                continue  # 此行为过道，跳过
            for x in range(1, max(1, self.width - 1)):
                # 垂直过道列留空
                if x == col_aisle:
                    continue
                # 依据坐标计算象限区号：
                # - 左/右由垂直过道 col_aisle 划分
                # - 上/下由水平中线 height//2 划分
                zone_x = 0 if x < self.col_aisle else 1
                zone_y = 0 if y < mid_y else 1
                zone = zone_y * 2 + zone_x
                self.shelves.append({
                    'id': sid,
                    'x': x,
                    'y': y,
                    'zone': zone,
                    'levels': self.levels_per_shelf,
                })
                sid += 1
        # 站点：全部放在最后一行 y = height - 1，沿宽度等距放置（避开边界列），并避免与充电桩相邻
        self.stations: List[Dict] = []
        if self.n_stations > 0:
            import numpy as _np
            xs = list(_np.linspace(1, max(1, self.width - 2), self.n_stations, dtype=int))
            used_exact = set()
            banned = set()  # 不能使用的位置（包含相邻位置）
            for i, x0 in enumerate(xs):
                x = max(1, min(self.width - 2, int(x0)))
                # 尝试向右寻找可用位，避免与已放置元素相邻（|dx|<1禁止）
                while (x in used_exact) or (x in banned):
                    if x < self.width - 2:
                        x += 1
                    else:
                        break
                # 若到右边界仍不可用，尝试向左
                while (x in used_exact) or (x in banned):
                    if x > 1:
                        x -= 1
                    else:
                        break
                self.stations.append({'id': i, 'x': x, 'y': self.height - 1, 'queue': [], 'throughput': 0})
                used_exact.add(x)
                banned.update({x - 1, x, x + 1})

        # 充电桩：也放在最后一行，且避免与站点及其相邻位置相邻
        self.charging_pads = []
        if self.n_charging_pads > 0:
            import numpy as _np
            xs_c = list(_np.linspace(1, max(1, self.width - 2), self.n_charging_pads, dtype=int))
            # 从 stations 构建初始禁用集合（包含相邻）
            used_exact = {s['x'] for s in self.stations}
            banned = set()
            for x in used_exact:
                banned.update({x - 1, x, x + 1})
            # 放置充电桩并避免相互相邻
            for i, x0 in enumerate(xs_c):
                x = max(1, min(self.width - 2, int(x0)))
                # 找到不与站点/已有充电桩相邻的位置
                while (x in used_exact) or (x in banned):
                    if x < self.width - 2:
                        x += 1
                    else:
                        break
                while (x in used_exact) or (x in banned):
                    if x > 1:
                        x -= 1
                    else:
                        break
                self.charging_pads.append({
                    'id': i,
                    'x': x,
                    'y': self.height - 1,
                    'occupied': False,
                    'charging_picker': None,
                })
                used_exact.add(x)
                banned.update({x - 1, x, x + 1})

    def _init_statistics(self):
        """初始化 episode 统计量（在 reset 中调用）"""
        self.current_time = 0.0
        self.current_step = 0
        self.total_orders_received = 0
        self.total_orders_completed = 0
        self.total_tasks_completed = 0
        # 累计完成价值（用于帧图与评估汇总）
        self.total_value_completed = 0
        # 累计超时销毁罚没（正值，表示被扣除的总价值）
        self.total_value_penalty = 0
        self.on_time_completions = 0
        self.late_completions = 0
        self.zone_loads = [0, 0, 0, 0]
        self.task_waiting_times = []
        self.picker_utilization = defaultdict(float)
        self.weight_stats = {
            'forklift_only': 0,
            'heavy': 0,
            'medium': 0,
            'light': 0,
        }

    # 重置环境
    def reset(self) -> np.ndarray:
        """重置环境，返回全局状态向量

        - 清零统计/时间；
        - 放置拣货员（不与货架/站点/桩/他人重叠，且不在最后一行）；
        - 准备任务池并根据当前时间窗口生成初始订单/任务。
        """
        # 时间与统计
        self._init_statistics()
        # 拣货员初始化（20% 车辆型）
        self.pickers: List[Picker] = []
        # 依据比例与最少数量确定叉车数量
        forklift_count = max(int(self.n_pickers * float(getattr(self, 'forklift_ratio', 0.2))), int(getattr(self, 'min_forklifts', 0)))
        forklift_count = min(self.n_pickers, forklift_count)
        # 不能重叠：收集已占用坐标（货架、站点、充电桩、已放置拣货员）
        occupied = set()
        occupied.update((sh['x'], sh['y']) for sh in self.shelves)
        occupied.update((st['x'], st['y']) for st in self.stations)
        occupied.update((ch['x'], ch['y']) for ch in self.charging_pads)
        # 放置拣货员：不与以上元素/彼此重叠，且不占用最后一行（站点/充电行）
        rng = np.random.default_rng()
        for i in range(self.n_pickers):
            ptype = PickerType.FORKLIFT if i < forklift_count else PickerType.REGULAR
            placed = False
            # 随机尝试若干次
            for _ in range(2000):
                x = int(rng.integers(1, max(2, self.width - 1)))  # 1..width-2
                y = int(rng.integers(1, max(2, self.height - 1)))  # 1..height-2（避免最后一行）
                if (x, y) not in occupied:
                    self.pickers.append(Picker(i, x, y, ptype))
                    occupied.add((x, y))
                    placed = True
                    break
            if not placed:
                # 退路：顺序扫描寻找第一个可用单元
                found = False
                for yy in range(1, max(1, self.height - 1)):
                    for xx in range(1, max(1, self.width - 1)):
                        if (xx, yy) not in occupied:
                            self.pickers.append(Picker(i, xx, yy, ptype))
                            occupied.add((xx, yy))
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # 极端情况下允许放在 (0,0)（理论上不应发生）
                    self.pickers.append(Picker(i, 0, 0, ptype))
                    occupied.add((0, 0))
        # 设置拣货员基础速度
        for p in self.pickers:
            if p.type == PickerType.FORKLIFT:
                p.speed = float(self.speed_config.get('base_speed', {}).get('forklift', 1.2))
            else:
                p.speed = float(self.speed_config.get('base_speed', {}).get('regular', 1.0))
        # 连续坐标初始化到整数位置
        for p in self.pickers:
            p.fx, p.fy = float(p.x), float(p.y)
        # 任务池
        self.task_pool: List[Task] = []
        self.task_queue: List[Tuple[float, int, Task]] = []
        self.task_counter = 0
        self.active_orders: Dict[int, Order] = {}
        # 初始订单（前6分钟内）
        initial_orders = self.order_generator.get_orders_in_window(self.current_time, self.current_time + 0.1)
        for order in initial_orders:
            self._process_new_order(order)
        # 不维护 nests 视图（由实验侧按需构建）。为兼容旧代码，提供空列表属性。
        self.nests: List[Any] = []
        return self._get_global_state()

    # 处理新订单
    def _process_new_order(self, order: Order):
        """将订单拆分为任务并加入任务池（使用订单层级 weight_class）"""
        self.total_orders_received += 1
        self.active_orders[order.order_id] = order
        # 分解为任务：使用物品的数值重量进行分类；forklift_only 需叉车
        for item_group in order.items:
            item_w = float(item_group.get('weight', 0.0))
            total_weight = max(0.0, float(np.round(item_w, 2)))
            # 依据阈值重新分类
            thr_med = float(self.weight_thresholds.get('medium', 30.0))
            thr_hvy = float(self.weight_thresholds.get('heavy', 70.0))
            thr_fk = float(self.weight_thresholds.get('forklift_only', 90.0))
            if total_weight >= thr_fk:
                weight_class = 'forklift_only'
                requires_car = True
            elif total_weight >= thr_hvy:
                weight_class = 'heavy'
                requires_car = False
            elif total_weight >= thr_med:
                weight_class = 'medium'
                requires_car = False
            else:
                weight_class = 'light'
                requires_car = False
            # 层级分配（鲁棒到任意 levels_per_shelf>=1）：
            L = int(max(1, self.levels_per_shelf))  # 总层数（>=1）
            if L == 1:
                level = 0
            elif weight_class in ('heavy', 'forklift_only'):
                # 低层优先：从 [0, floor(L/2)] 里选（high 为排他）
                high = max(1, min(L, L // 2 + 1))
                level = int(np.random.randint(0, high))
            elif weight_class == 'medium':
                # 全范围任选（可按需再加权）
                level = int(np.random.randint(0, L))
            else:  # light
                # 高层优先：从 [max(0,L-2), L-1] 里选
                low = max(0, L - 2)
                level = int(np.random.randint(low, L))
            # 将生成器给出的 shelf/zone 映射到环境实际货架：
            # 优先在同 zone 的货架中随机选择，避免总是落在较小的全局 shelf_id 上导致偏左上聚集。
            _zone = int(item_group.get('zone', 0))
            _cand_ids = [sh['id'] for sh in self.shelves if int(sh.get('zone', 0)) == _zone]
            if _cand_ids:
                mapped_shelf_id = int(np.random.choice(_cand_ids))
            else:
                mapped_shelf_id = int(np.random.randint(0, len(self.shelves)))

            # 基础价值：使用该任务第一件物品的 value*quantity（若缺省则为0）
            try:
                base_val = int(item_group.get('quantity', 1)) * int(item_group.get('value', 0))
            except Exception:
                base_val = 0

            task = Task(
                task_id=self.task_counter,
                order_id=order.order_id,
                task_type='pick',
                priority=order.priority,
                deadline=order.deadline,
                # 将生成器的 shelf_id 限制到现有货架数量范围
                shelf_id=mapped_shelf_id,
                shelf_level=int(level),
                station_id=np.random.randint(0, self.n_stations) if self.n_stations > 0 else None,
                items=[item_group],
                zone=_zone,
                weight=total_weight,
                weight_class=weight_class,
                requires_car=requires_car,
                status=TaskStatus.PENDING,
                base_value=base_val,
            )
            self.task_pool.append(task)
            heapq.heappush(self.task_queue, (task.deadline, task.task_id, task))
            self.zone_loads[task.zone] += 1
            self.task_counter += 1


    def step(self, actions: Dict[int, int]) -> Tuple[np.ndarray, Dict[int, float], Dict[int, bool], Dict]:
        """推进一步仿真

        输入：actions = {picker_id: action_int}
        输出：next_state（np.ndarray）、rewards（按 pid）、dones、info
        """
        rewards: Dict[int, float] = {}
        info = {
            'tasks_completed': [],
            'orders_completed': [],
            # 不再使用 nest 概念
            'late_tasks': [],
        }
        # 先执行外部控制 hook（策略可动态调参）
        if self.control_hook is not None:
            try:
                self.control_hook(self, {'time': self.current_time, 'step': self.current_step})
            except Exception:
                pass
        # 必须提供统一速度函数：为每个拣货员设置本步速度覆盖
        if self.speed_fn is None:
            raise RuntimeError("speed_function is required. Please register via env.set_speed_function(fn).")
        speeds = self.speed_fn(self)
        if not isinstance(speeds, dict):
            raise RuntimeError("speed_function must return a dict {picker_id: speed}.")
        # 清空旧覆盖，并逐一校验/设置
        self.picker_speed_overrides.clear()
        for p in self.pickers:
            if p.id not in speeds:
                raise RuntimeError(f"speed_function missing speed for picker {p.id}.")
            v = float(speeds[p.id])
            if v <= 0:
                raise RuntimeError(f"speed_function returned non-positive speed {v} for picker {p.id}.")
            self.set_picker_speed(p.id, v)
        # 新订单
        new_orders = self.order_generator.get_orders_in_window(self.current_time, self.current_time + self.time_step / 3600)
        for order in new_orders:
            self._process_new_order(order)
        # 执行动作（带占用检查；可在通道上重叠）
        occupied = {(p.x, p.y) for p in self.pickers}
        shelf_cells = {(sh['x'], sh['y']) for sh in self.shelves}
        for pid, act in actions.items():
            if pid >= len(self.pickers):
                continue
            picker = self.pickers[pid]
            rew = self._execute_picker_action(picker, act, info, occupied, shelf_cells)
            rewards[pid] = rew
            if picker.current_task is not None:
                self.picker_utilization[pid] += 1
        # 截止期检查、充电
        self._check_deadlines(info)
        self._update_charging()
        # 统计与时间推进
        self.current_time += self.time_step / 3600
        self.current_step += 1
        done = self.current_step >= self.max_steps
        dones = {i: done for i in range(self.n_pickers)}
        next_state = self._get_global_state()
        # 全局奖励按人头均摊
        global_reward = self._calculate_global_reward(rewards, info)
        if rewards:
            for k in rewards:
                rewards[k] += global_reward / max(1, len(rewards))
        return next_state, rewards, dones, info

    def _execute_picker_action(self, picker: Picker, action: int, info: Dict, occupied: set, shelf_cells: set) -> float:
        """执行单体动作并计算奖励（含移动/碰撞/朝目标前进奖励，及 IDLE 触发的拣/投）。支持基于速度/拥堵/载重的多步移动。"""
        reward = self.reward_config['idle_penalty']
        old_x, old_y = picker.x, picker.y
        moved_cells = 0

        # 计算本步有效速度：必须使用 speed_function 提供的覆盖值
        t = picker.current_task
        if picker.id not in self.picker_speed_overrides:
            raise RuntimeError(f"Missing speed override for picker {picker.id}. Ensure speed_function covers all pickers.")
        base_speed = float(self.picker_speed_overrides[picker.id])
        # 记录用于本步的速度（用于调试/可视化）
        self.last_speeds[picker.id] = base_speed
        # 记录速度分解（用于调试）：基础、覆盖、重量项、拥堵项、是否携货、是否用speed_fn
        wt = float(getattr(t, 'weight', 0.0)) if t is not None else 0.0
        wt_term = self._weight_term(wt)
        cong_red = self._congestion_reduction(picker)
        self.last_speed_components[picker.id] = {
            'picker_base': float(getattr(picker, 'speed', 1.0)),
            'override': float(self.picker_speed_overrides.get(picker.id, float('nan')))
                        if picker.id in self.picker_speed_overrides else float('nan'),
            'weight': wt,
            'weight_term': float(wt_term),
            'congestion_red': float(cong_red),
            'carrying': 1.0 if bool(picker.carrying_items) else 0.0,
            'using_speed_fn': 1.0,
            'base_used': float(base_speed),
        }

        # 计算可移动单元数（向下取整 + 概率补偿）
        whole = int(max(0.0, base_speed))
        frac = max(0.0, base_speed - whole)
        extra = 1 if (frac > 0.0 and np.random.random() < frac) else 0
        steps_to_move = whole + extra

        # 执行动作（连续位置 + 栅格占用）
        if action in (ActionType.UP.value, ActionType.DOWN.value, ActionType.LEFT.value, ActionType.RIGHT.value):
            dd = {
                ActionType.UP.value: (0, -1),
                ActionType.DOWN.value: (0, 1),
                ActionType.LEFT.value: (-1, 0),
                ActionType.RIGHT.value: (1, 0),
            }[action]
            # 记录本步方向用于导航反摆动偏好
            picker.last_dir = action
            # 目标连续位移
            s = float(steps_to_move) + frac
            # 当前连续坐标
            if not hasattr(picker, 'fx'):
                picker.fx, picker.fy = float(picker.x), float(picker.y)
            # 先按整数跨格移动（考虑障碍）
            steps_int = int(s)
            for _ in range(steps_int):
                nx, ny = picker.x + dd[0], picker.y + dd[1]
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    break
                if (nx, ny) in shelf_cells:
                    break
                picker.x, picker.y = nx, ny
                moved_cells += 1
            # 剩余小数位移，仅更新连续坐标（不跨入障碍）
            rem = s - steps_int
            # 连续目标
            tx = picker.x + dd[0] * rem
            ty = picker.y + dd[1] * rem
            # 若下一格是障碍，则保持在栅格中心，不越界
            nx_block, ny_block = picker.x + dd[0], picker.y + dd[1]
            if 0 <= nx_block < self.width and 0 <= ny_block < self.height and (nx_block, ny_block) in shelf_cells:
                picker.fx, picker.fy = float(picker.x), float(picker.y)
            else:
                picker.fx, picker.fy = float(tx), float(ty)
        else:
            # IDLE/PICK/DROP：尝试执行当前任务
            reward += self._try_execute_task(picker, info)
            # 保持连续坐标与栅格一致
            picker.fx, picker.fy = float(picker.x), float(picker.y)

        if moved_cells > 0:
            picker.total_distance += moved_cells
            if picker.current_task:
                reward += self._calculate_movement_reward(picker, old_x, old_y)
        # 电池消耗按移动单元数叠加
        if moved_cells > 0:
            picker.battery -= 0.1 * moved_cells
        else:
            picker.battery -= 0.05
        if picker.battery < 20:
            reward += self.reward_config['battery_low_penalty']
        # 拥堵
        if self._check_congestion(picker):
            reward += self.reward_config['congestion_penalty']
        return reward

    # === 价值衰减相关工具 ===
    def get_task_decayed_value(self, task: Task, at_time: Optional[float] = None) -> int:
        """返回某任务在指定时刻（默认当前时刻）的衰减后价值（非负整数）。

        规则：
        - t <= D: 价值 = base_value
        - D < t < 2D: 线性衰减 = base_value * (2D - t)/D
        - t >= 2D: 价值 = 0
        """
        try:
            base_val = int(getattr(task, 'base_value', 0))
        except Exception:
            base_val = 0
        if base_val <= 0:
            return 0
        tnow = float(self.current_time if at_time is None else at_time)
        try:
            D = float(getattr(task, 'deadline', 0.0))
        except Exception:
            D = 0.0
        if D <= 0.0:
            # 无有效截止期时，不进行衰减
            return max(0, base_val)
        if tnow <= D:
            return max(0, base_val)
        if tnow >= 2.0 * D:
            return 0
        factor = max(0.0, min(1.0, (2.0 * D - tnow) / max(1e-6, D)))
        return int(round(max(0.0, base_val * factor)))

    def get_task_value_components(self, task: Task, at_time: Optional[float] = None) -> Dict[str, int]:
        """返回任务价值构成：{'base': base, 'decayed': decayed, 'decay': base-decayed}。
        decayed 按 get_task_decayed_value 计算；decay 不为负。
        """
        try:
            base_val = int(getattr(task, 'base_value', 0))
        except Exception:
            base_val = 0
        decayed = int(self.get_task_decayed_value(task, at_time=at_time))
        decay = max(0, base_val - decayed)
        return {'base': max(0, base_val), 'decayed': max(0, decayed), 'decay': decay}

    # === 效率相关辅助 ===
    def _weight_term(self, weight: float) -> float:
        # 使用“仅叉车阈值”作为效率基准，确保在未达 forklift_only 前仍保有正速度：
        # eff_weight = (thr_base - weight + 10) / (thr_base + 10)，并裁剪到 [0, 2]
        # 其中 thr_base 优先取 forklift_only 阈值，若缺省则回退 heavy。
        thr_base = float(self.weight_thresholds.get('forklift_only', self.weight_thresholds.get('heavy', 70.0)))
        num = (thr_base - float(weight) + 10.0)
        den = (thr_base + 10.0)
        val = num / den if den > 0 else 1.0
        # 限制在 [0, 2] 避免负速或过大
        return float(np.clip(val, 0.0, 2.0))

    def _congestion_reduction(self, picker: Picker) -> float:
        # 拥堵减速量（0~1）。若不希望考虑拥堵，可在 speed_config 中将该值设为 0。
        if self._check_congestion(picker):
            if 'congestion_reduction' in self.speed_config:
                val = float(self.speed_config.get('congestion_reduction', 0.3))
            else:
                mult = float(self.speed_config.get('congestion_mult', 1.0))
                val = max(0.0, min(1.0, 1.0 - mult))
            return max(0.0, min(1.0, val))
        return 0.0

    def _compute_movement_efficiency(self, picker: Picker, task: Optional[Task]) -> float:
        # 公式：eff = weight_term * (1 - 拥堵减速)
        wt = self._weight_term(getattr(task, 'weight', 0.0))
        cong_red = self._congestion_reduction(picker)
        return max(0.0, wt * (1.0 - cong_red))

    def _compute_pickdrop_efficiency(self, picker: Picker, task: Optional[Task]) -> float:
        # 公式：((最高层数-层数+1)/最高层数) * base_speed * weight_term
        if task is None:
            return 1.0
        L = int(max(1, self.levels_per_shelf))
        level = int(getattr(task, 'shelf_level', 0))
        # 将 0-based level 转为 (L - level)/L 范围 (0,1]
        level_term = max(0.0, min(1.0, (L - level) / max(1, L)))
        wt = self._weight_term(getattr(task, 'weight', 0.0))
        return max(0.0, level_term * wt)

    # 不在环境内进行任务分配（无 nest / 无启发式）。
    def _try_execute_task(self, picker: Picker, info: Dict) -> float:
        """在 IDLE/PICK/DROP 时尝试执行当前任务：相邻拣/投，成功则更新任务状态并返回奖励增量"""
        reward = 0.0
        # 没有任务时：环境不分配任务（由实验/方法侧负责）。
        if picker.current_task is None:
            pass
        t = picker.current_task
        if t is None:
            return reward
        # 拣货（在正确货架）
        shelf = self.shelves[t.shelf_id] if (t.shelf_id is not None and t.shelf_id < len(self.shelves)) else None
        # 必须与货架相邻（曼哈顿距离=1）才可PICK
        if not picker.carrying_items and shelf and (abs(picker.x - shelf['x']) + abs(picker.y - shelf['y']) == 1):
            # 权限与效率
            if t.requires_car and picker.type != PickerType.FORKLIFT:
                # 普通拣货员不能执行heavy
                return reward - 0.5
            base = self.reward_config['pick_base'][t.weight_class]
            eff = self.reward_config['forklift_eff' if picker.type == PickerType.FORKLIFT else 'regular_eff'][t.weight_class]
            eff_pd = self._compute_pickdrop_efficiency(picker, t)
            pick_reward = base * eff * eff_pd
            picker.carrying_items = t.items
            t.status = TaskStatus.IN_PROGRESS
            reward += pick_reward
        # 送货（在站点相邻）
        elif picker.carrying_items and t.station_id is not None and t.station_id < len(self.stations):
            st = self.stations[t.station_id]
            # 放宽：在站点格或相邻格（曼哈顿距离<=1）均可 DROP
            if abs(picker.x - st['x']) + abs(picker.y - st['y']) <= 1:
                base = self.reward_config['drop_base'][t.weight_class]
                eff = self.reward_config['forklift_eff' if picker.type == PickerType.FORKLIFT else 'regular_eff'][t.weight_class]
                eff_pd = self._compute_pickdrop_efficiency(picker, t)
                drop_reward = base * eff * eff_pd
                picker.carrying_items = []
                t.status = TaskStatus.COMPLETED
                t.completion_time = self.current_time
                picker.completed_tasks += 1
                self.total_tasks_completed += 1
                # 衰减后的结算价值：使用统一工具函数
                try:
                    val_add = int(self.get_task_decayed_value(t, at_time=self.current_time))
                    self.total_value_completed += max(0, val_add)
                    # 超过 1x deadline 起，价值降低部分计入累计罚没
                    try:
                        base_val = int(getattr(t, 'base_value', 0))
                    except Exception:
                        base_val = 0
                    if self.current_time > float(getattr(t, 'deadline', float('inf'))):
                        dec = max(0, base_val - max(0, val_add))
                        self.total_value_penalty += dec
                except Exception:
                    pass
                self.weight_stats[t.weight_class] += 1
                # 是否准时
                if self.current_time <= t.deadline:
                    drop_reward += self.reward_config['on_time_bonus']
                    self.on_time_completions += 1
                else:
                    drop_reward += self.reward_config['late_penalty']
                    info['late_tasks'].append(t.task_id)
                reward += drop_reward
                info['tasks_completed'].append(t.task_id)
                picker.current_task = None
        return reward

    def _calculate_movement_reward(self, picker: Picker, old_x: int, old_y: int) -> float:
        # 向目标靠近给予微奖励（携带重物更鼓励）；距离采用通道内BFS距离
        t = picker.current_task
        if t is None:
            return 0.0
        # 选择目标：携货→站点；未携货→货架相邻可达格
        target = None
        if picker.carrying_items:
            st = self.stations[t.station_id] if t.station_id is not None else None
            if st:
                target = (st['x'], st['y'])
        else:
            if t.shelf_id is not None:
                sh = self.shelves[t.shelf_id]
                # 选择距离当前位置最近的货架相邻可达格
                target = self._nearest_adjacent_accessible((sh['x'], sh['y']), (picker.x, picker.y))
        if not target:
            return 0.0
        old_dist = self._aisle_distance((old_x, old_y), target)
        new_dist = self._aisle_distance((picker.x, picker.y), target)
        if new_dist < old_dist:
            base = self.reward_config['move_toward_target']
            if picker.carrying_items and getattr(t, 'weight_class', 'light') in ('heavy', 'forklift_only'):
                base *= 2.0
            return base
        return 0.0

    def _aisle_distance(self, start, goal) -> int:
        """通道内BFS最短路（屏蔽货架格）。不可达返回大数。"""
        from collections import deque
        sx, sy = start
        gx, gy = goal
        if (sx, sy) == (gx, gy):
            return 0
        W, H = self.width, self.height
        if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
            return 10**9
        if self.grid[sy, sx] == 2 or self.grid[gy, gx] == 2:
            return 10**9
        q = deque([(sx, sy, 0)])
        vis = [[False]*W for _ in range(H)]
        vis[sy][sx] = True
        while q:
            x, y, d = q.popleft()
            for dx, dy in ((0,1),(1,0),(0,-1),(-1,0)):
                nx, ny = x+dx, y+dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if vis[ny][nx] or self.grid[ny, nx] == 2:
                    continue
                if (nx, ny) == (gx, gy):
                    return d+1
                vis[ny][nx] = True
                q.append((nx, ny, d+1))
        return 10**9

    def _nearest_adjacent_accessible(self, shelf_pos, from_pos):
        """返回离 from_pos 最近的货架相邻可达格（非货架）。"""
        x, y = shelf_pos
        best = None
        best_d = 10**9
        for dx, dy in ((0,1),(1,0),(0,-1),(-1,0)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] != 2:
                d = self._aisle_distance(from_pos, (nx, ny))
                if d < best_d:
                    best_d = d
                    best = (nx, ny)
        return best

    def _check_deadlines(self, info: Dict):
        """标记超过截止时间但未完成的任务（记录在 info['late_tasks']）"""
        # 过期标记与超期销毁
        to_keep = []
        destroyed_ids = set()
        for dl, tid, t in list(self.task_queue):
            if t.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                if self.current_time > t.deadline:
                    info.setdefault('late_tasks', []).append(t.task_id)
                # 超过 2*deadline 自动销毁并记负值（罚没 base_value）
                if self.current_time >= 2 * t.deadline:
                    self._destroy_task(t, info)
                    destroyed_ids.add(t.task_id)
                    continue
            to_keep.append((dl, tid, t))
        # 重建任务堆，去除销毁的任务
        if destroyed_ids:
            import heapq
            self.task_queue = to_keep
            heapq.heapify(self.task_queue)

    def _update_charging(self):
        # 简化充电逻辑：电量低时停留IDLE即可缓慢恢复（不占桩）
        for p in self.pickers:
            if p.battery < 15 and not p.carrying_items and p.current_task is None:
                p.battery = min(100.0, p.battery + 0.2)

    def _destroy_task(self, t: Task, info: Dict):
        """销毁超期任务：从任务池/队列中移除；若在执行中则解除 picker 绑定；
        并将该任务的基础价值计为负值累加（罚没）。
        """
        # 解除 picker 绑定
        if t.assigned_picker is not None and 0 <= t.assigned_picker < len(self.pickers):
            p = self.pickers[t.assigned_picker]
            if getattr(p, 'current_task', None) is t:
                p.current_task = None
            # 丢弃携带的物品
            if getattr(p, 'carrying_items', None):
                p.carrying_items = []
        # 从任务池移除
        try:
            if t in self.task_pool:
                self.task_pool.remove(t)
        except Exception:
            pass
        # 统计罚没
        try:
            base_val = int(getattr(t, 'base_value', 0))
            pen = max(0, base_val)
            self.total_value_completed -= pen
            self.total_value_penalty += pen
        except Exception:
            pass
        # 标记状态
        t.status = TaskStatus.FAILED
        info.setdefault('destroyed_tasks', []).append(t.task_id)

    def _check_congestion(self, picker: Picker) -> bool:
        """拥堵判定：若在周边8个可通行格子中存在任意其他拣货员，则视为拥堵。
        可通行格子=非货架（grid!=2）。
        """
        x0, y0 = picker.x, picker.y
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x0 + dx, y0 + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                # 仅考虑可通行格
                if self.grid[ny, nx] == 2:
                    continue
                # 若该相邻格被其他拣货员占据 → 拥堵
                for other in self.pickers:
                    if other.id != picker.id and other.x == nx and other.y == ny:
                        return True
        return False

    def _calculate_global_reward(self, rewards: Dict[int, float], info: Dict) -> float:
        """计算全局奖励增量：区域均衡/准时率等（按人头均摊到个体奖励）"""
        global_reward = 0.0
        # 区域负载平衡
        total = sum(self.zone_loads)
        if total > 0:
            std = np.std(self.zone_loads)
            if std < max(1.0, total * 0.1):
                global_reward += self.reward_config['zone_balance_bonus']
        # 准时率奖励
        if info['tasks_completed']:
            on_time_count = sum(1 for tid in info['tasks_completed'] if tid not in info.get('late_tasks', []))
            ratio = on_time_count / max(1, len(info['tasks_completed']))
            global_reward += ratio * 2.0
        return global_reward

    def _get_global_state(self) -> np.ndarray:
        """构造全局状态向量（用于上层 Manager 或评估）"""
        # 1. 时间特征（周期 + 当前到达率）
        hour = self.current_time % 24
        s = [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            self.order_generator.get_arrival_rate(self.current_time) / 100,
        ]
        # 2. 任务池统计（总量/类别）
        pool = self.task_pool
        s.append(len(pool) / 100)
        for w in ['heavy', 'medium', 'light']:
            s.append(len([t for t in pool if t.weight_class == w and t.status == TaskStatus.PENDING]) / 50)
        # 3. 区域×重量类别摘要（共12格，每格4个统计：计数/平均优先级/占位eta/平均剩余时间）
        zone_weight_groups = []
        for zone in range(4):
            for w in ['heavy', 'medium', 'light']:
                tasks = [t for t in self.task_pool if t.zone == zone and t.weight_class == w and t.status == TaskStatus.PENDING]
                count = len(tasks)
                avg_pri = float(np.mean([t.priority for t in tasks])) if tasks else 0.0
                avg_rem = float(np.mean([t.deadline - self.current_time for t in tasks])) if tasks else 0.0
                zone_weight_groups.append((count, avg_pri, 0.5, avg_rem))
        # 若任务不足12组，补零
        while len(zone_weight_groups) < 12:
            zone_weight_groups.append((0, 0.0, 0.5, 0.0))
        for (cnt, pri, eta_placeholder, rem) in zone_weight_groups[:12]:
            s.extend([cnt / 20.0, pri, eta_placeholder, rem])
        # 4. 拣货员总体
        busy = sum(1 for p in self.pickers if p.current_task is not None)
        forklifts = sum(1 for p in self.pickers if p.type == PickerType.FORKLIFT)
        forklifts_busy = sum(1 for p in self.pickers if p.type == PickerType.FORKLIFT and p.current_task is not None)
        s.extend([
            busy / max(1, self.n_pickers),
            np.mean([p.battery for p in self.pickers]) / 100 if self.pickers else 1.0,
            forklifts_busy / max(1, forklifts) if forklifts > 0 else 0.0,
        ])
        # 5. 区域负载占比
        total = sum(self.zone_loads)
        for z in range(4):
            s.append(self.zone_loads[z] / max(1, total))
        # 6. 绩效
        s.extend([
            self.total_orders_completed / max(1, self.total_orders_received),
            self.on_time_completions / max(1, self.total_tasks_completed),
        ])
        return np.array(s, dtype=np.float32)

    def _calculate_state_dim(self) -> int:
        # 3(时间)+1(任务量)+3(按重量)+48(12×4 分组摘要) +3(拣货员) +4(区域) +2(绩效)
        return 3 + 1 + 3 + 48 + 3 + 4 + 2

    @property
    def stats(self):
        """返回统计信息快照（准时率/完成量/待办量/已完重量分布）"""
        on_time_rate = (self.on_time_completions / max(1, self.total_tasks_completed)) * 100
        return {
            'on_time_rate': on_time_rate,
            'completed_orders': self.total_orders_completed,
            'completed_tasks': self.total_tasks_completed,
            'pending_tasks': len(self.task_pool),
            'weight_distribution_completed': dict(self.weight_stats),
        }

    # 不再需要订单类型到重量的映射（weight_class 直接来自订单/物品）

    def get_picker_state(self, picker_id: int) -> np.ndarray:
        """返回单个拣货员的局部观测（位置/电量/是否叉车/载荷/任务信息/邻近密度）"""
        p = self.pickers[picker_id]
        st = [
            p.x / self.width,
            p.y / self.height,
            p.battery / 100,
            1.0 if p.type == PickerType.FORKLIFT else 0.0,
            len(p.carrying_items) / p.capacity,
        ]
        if p.current_task:
            t = p.current_task
            st.extend([
                t.priority,
                (t.deadline - self.current_time) / 10,
                1.0 if p.carrying_items else 0.0,
            ])
            # 目标位置（货架/站点）
            if not p.carrying_items and t.shelf_id is not None:
                sh = self.shelves[t.shelf_id]
                st.extend([sh['x'] / self.width, sh['y'] / self.height])
            elif t.station_id is not None:
                stn = self.stations[t.station_id]
                st.extend([stn['x'] / self.width, stn['y'] / self.height])
            else:
                st.extend([0.5, 0.5])
        else:
            st.extend([0, 0, 0, 0.5, 0.5])
        # 邻近拥堵提示
        nearby = 0
        for other in self.pickers:
            if other.id != picker_id and abs(other.x - p.x) + abs(other.y - p.y) <= 3:
                nearby += 1
        st.append(min(1.0, nearby / 5))
        return np.array(st, dtype=np.float32)

    def plot(self, save_path: str = None, show: bool = True, figsize=(6, 6)):
        """简便接口：仅绘制当前环境布局/实体（等价于 render('plot')）。
        参数:
            save_path: 可选，保存到文件路径
            show: 是否调用 plt.show()
            figsize: 绘图尺寸
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=figsize)
        # 画背景网格
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True, alpha=0.2)
        # 区域(Zone)可视化：按中线划分为4个象限（0/1/2/3），并用半透明背景区分
        try:
            midx, midy = int(getattr(self, 'col_aisle', self.width // 2)), int(self.height // 2)
            zone_colors = {
                0: (0.75, 0.85, 1.00, 0.15),  # 淡蓝
                1: (0.85, 1.00, 0.85, 0.15),  # 淡绿
                2: (1.00, 0.90, 0.80, 0.15),  # 淡橙
                3: (1.00, 0.80, 0.80, 0.15),  # 淡红
            }
            # zone 0: 左上
            ax.add_patch(Rectangle((-0.5, -0.5), midx, midy, facecolor=zone_colors[0], edgecolor='none', zorder=0))
            # zone 1: 右上
            ax.add_patch(Rectangle((midx-0.5, -0.5), self.width - midx + 0.5, midy, facecolor=zone_colors[1], edgecolor='none', zorder=0))
            # zone 2: 左下
            ax.add_patch(Rectangle((-0.5, midy-0.5), midx, self.height - midy + 0.5, facecolor=zone_colors[2], edgecolor='none', zorder=0))
            # zone 3: 右下
            ax.add_patch(Rectangle((midx-0.5, midy-0.5), self.width - midx + 0.5, self.height - midy + 0.5, facecolor=zone_colors[3], edgecolor='none', zorder=0))
            # 中线辅助线
            ax.axvline(midx-0.5, color='k', lw=0.5, ls='--', alpha=0.2)
            ax.axhline(midy-0.5, color='k', lw=0.5, ls='--', alpha=0.2)
            # 标注 zone 文本
            cx = [midx/2, (midx + self.width)/2, midx/2, (midx + self.width)/2]
            cy = [midy/2, midy/2, (midy + self.height)/2, (midy + self.height)/2]
            for zid in range(4):
                ax.text(cx[zid], cy[zid], f"Z{zid}", fontsize=10, ha='center', va='center', alpha=0.6)
        except Exception:
            pass
        # 货架
        xs = [sh['x'] for sh in self.shelves]
        ys = [sh['y'] for sh in self.shelves]
        ax.scatter(xs, ys, c='#888888', marker='s', s=80, label='Shelf')
        # 未完成任务覆盖（按重量着色）
        try:
            pending = [t for t in getattr(self, 'task_pool', []) if getattr(t, 'status', None) != TaskStatus.COMPLETED]
            if pending:
                wcolor = {'heavy': '#d62728', 'medium': '#ff7f0e', 'light': '#2ca02c'}
                for w in ('heavy', 'medium', 'light'):
                    pts = [(self.shelves[t.shelf_id]['x'], self.shelves[t.shelf_id]['y'])
                           for t in pending if getattr(t, 'weight_class', None) == w and t.shelf_id is not None and t.shelf_id < len(self.shelves)]
                    if pts:
                        ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=wcolor[w], s=40, marker='o', alpha=0.9, label=f'Unfinished-{w}')
        except Exception:
            pass
        # 站点
        xs = [st['x'] for st in self.stations]
        ys = [st['y'] for st in self.stations]
        if xs:
            ax.scatter(xs, ys, c='#2ca02c', marker='s', s=120, label='Station')
        # 充电桩
        xs = [ch['x'] for ch in self.charging_pads]
        ys = [ch['y'] for ch in self.charging_pads]
        if xs:
            ax.scatter(xs, ys, c='#ffcc00', marker='s', s=100, label='Charger')
        # 拣货员
        px_f = [((getattr(p, 'fx', p.x)), (getattr(p, 'fy', p.y))) for p in self.pickers if p.type == PickerType.FORKLIFT]
        px_r = [((getattr(p, 'fx', p.x)), (getattr(p, 'fy', p.y))) for p in self.pickers if p.type == PickerType.REGULAR]
        if px_r:
            ax.scatter([p[0] for p in px_r], [p[1] for p in px_r], c='#1f77b4', s=60, label='Picker')
        if px_f:
            ax.scatter([p[0] for p in px_f], [p[1] for p in px_f], c='#d62728', s=60, label='Forklift')
        # 标注拣货员ID与载货（携带价值，显示为 base-decay 并列）
        for p in self.pickers:
            txt = f"{p.id}"
            try:
                if getattr(p, 'carrying_items', None) and getattr(p, 'current_task', None) is not None:
                    comps = self.get_task_value_components(p.current_task, at_time=self.current_time)
                    c_base = int(comps.get('base', 0))
                    c_decay = int(comps.get('decay', 0))
                    if c_base > 0:
                        txt += f"({c_base}-{c_decay})"
            except Exception:
                pass
            ax.text(getattr(p, 'fx', p.x), getattr(p, 'fy', p.y) + 0.2, txt, fontsize=8, ha='center', va='bottom')

        # 方向箭头（展示最近一次移动方向）
        dir_vec = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # UP, DOWN, LEFT, RIGHT
        for p in self.pickers:
            d = dir_vec.get(getattr(p, 'last_dir', None))
            if d is None:
                continue
            fx, fy = float(getattr(p, 'fx', p.x)), float(getattr(p, 'fy', p.y))
            L = 0.45  # 箭头长度（小于1格）
            color = '#d62728' if p.type == PickerType.FORKLIFT else '#1f77b4'
            try:
                ax.arrow(fx, fy, d[0]*L, d[1]*L, head_width=0.18, head_length=0.18, fc=color, ec=color, alpha=0.8, length_includes_head=True)
            except Exception:
                pass
        # 标题与图例：显示完成/未完成任务数量
        ax.invert_yaxis()
        try:
            unfinished = sum(1 for t in getattr(self, 'task_pool', []) if getattr(t, 'status', None) != TaskStatus.COMPLETED)
        except Exception:
            unfinished = len(getattr(self, 'task_pool', []))
        finished = int(getattr(self, 'total_tasks_completed', 0))
        pen = int(getattr(self, 'total_value_penalty', 0))
        ax.set_title(
            f"t={self.current_time:.2f}h, step={self.current_step}, finished={finished}, unfinished={unfinished}, value(decayed)={int(getattr(self,'total_value_completed',0))}, penalty={pen}"
        )
        # 将图例放在图外，避免遮挡
        from matplotlib import pyplot as _plt
        fig.subplots_adjust(right=0.80)
        ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8, borderaxespad=0.)
        _plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return

if __name__ == '__main__':
    import os
    cfg = {
        'width': 32,
        'height': 32,
        'n_pickers': 8,
        'n_shelves': 64,
        'n_stations': 4,
        'n_charging_pads': 2,
        'levels_per_shelf': 4,
        'order_config': {'base_rate': 60, 'simulation_hours': 1}
    }
    _env = DynamicWarehouseEnv(cfg)
    _env.reset()
    os.makedirs('results', exist_ok=True)
    _env.plot(save_path='results/env_view.png', show=False)
    print('保存环境预览到 results/env_view.png')
