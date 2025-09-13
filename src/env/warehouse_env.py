import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import heapq
from collections import defaultdict

class ActionType(Enum):
    IDLE = 0      # Stay in place
    MOVE_UP = 1   # Move up (y-1)
    MOVE_DOWN = 2  # Move down (y+1)
    MOVE_LEFT = 3  # Move left (x-1)
    MOVE_RIGHT = 4 # Move right (x+1)
    PICK = 5      # Pick item from shelf
    DROP = 6      # Drop item at station

class CellType(Enum):
    EMPTY = 0
    WALL = 1
    SHELF = 2
    STATION = 3
    CHARGING = 4

@dataclass
class Order:
    order_id: int
    items: List[Tuple[int, int]]  # List of (shelf_x, shelf_y) positions
    arrival_time: float
    deadline: float
    priority: int = 1
    assigned_picker: Optional[int] = None
    completion_time: Optional[float] = None
    requires_forklift: bool = False  # Whether this order requires forklift equipment
    
@dataclass
class Picker:
    picker_id: int
    x: int
    y: int
    carrying_items: List[Any] = None
    current_order: Optional[int] = None
    battery: float = 100.0
    speed: float = 1.0
    is_forklift: bool = False  # Equipment type constraint
    
    def __post_init__(self):
        if self.carrying_items is None:
            self.carrying_items = []

class WarehouseEnv:
    def __init__(self, config: Dict[str, Any]):
        self.width = config.get('width', 50)
        self.height = config.get('height', 50)
        self.n_pickers = config.get('n_pickers', 64)
        self.n_shelves = config.get('n_shelves', 200)
        self.n_stations = config.get('n_stations', 10)
        self.order_arrival_rate = config.get('order_arrival_rate', 0.5)
        self.max_items_per_order = config.get('max_items_per_order', 10)
        self.decision_interval = config.get('decision_interval', 2.0)  # Manager decision every 2s
        self.forklift_ratio = config.get('forklift_ratio', 0.2)  # 20% are forklifts
        
        # Initialize grid
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.shelves = []
        self.stations = []
        self.pickers = []
        self.orders = []
        self.completed_orders = []
        self.time = 0.0
        self.total_steps = 0
        self.next_order_id = 0  # Global order counter
        
        # Metrics tracking
        self.metrics = {
            'throughput': [],
            'completion_rate': [],
            'mean_wait_time': [],
            'tail_latency_95': [],
            'congestion_time': 0,
            'near_collisions': 0,
            'total_distance': 0,
            'zone_queues': defaultdict(list),  # For per-zone balance
            'congestion_events': []  # Track congestion occurrences
        }
        
        # Define warehouse zones for balance metrics
        self.n_zones = 4  # Divide warehouse into 4 zones
        self.zone_width = self.width // 2
        self.zone_height = self.height // 2
        
        self._setup_warehouse()
        self._initialize_pickers()
        
    def _setup_warehouse(self):
        # Create warehouse layout with aisles and shelves
        # Adjust parameters for smaller grids
        if self.width <= 15 or self.height <= 15:
            # Small grid configuration
            aisle_width = 1
            shelf_block_width = 2
            shelf_block_height = 2
            y_offset = 1
            x_offset = 1
            margin = 1
        else:
            # Normal configuration for larger grids
            aisle_width = 3
            shelf_block_width = 8
            shelf_block_height = 4
            y_offset = 5
            x_offset = 5
            margin = 5
        
        # Place shelves in a grid pattern with aisles
        shelf_id = 0
        initial_x_offset = x_offset
        while y_offset + shelf_block_height < self.height - margin:
            x_offset = initial_x_offset
            while x_offset + shelf_block_width < self.width - margin:
                for dy in range(shelf_block_height):
                    for dx in range(shelf_block_width):
                        y = y_offset + dy
                        x = x_offset + dx
                        if y < self.height and x < self.width:
                            self.grid[y, x] = CellType.SHELF.value
                            self.shelves.append((x, y))
                            shelf_id += 1
                            if shelf_id >= self.n_shelves:
                                break
                    if shelf_id >= self.n_shelves:
                        break
                if shelf_id >= self.n_shelves:
                    break
                x_offset += shelf_block_width + aisle_width
            y_offset += shelf_block_height + aisle_width
            
        # Place stations at the bottom of the warehouse
        station_spacing = self.width // (self.n_stations + 1)
        for i in range(self.n_stations):
            x = station_spacing * (i + 1)
            y = self.height - 2
            self.grid[y, x] = CellType.STATION.value
            self.stations.append((x, y))
            
    def _initialize_pickers(self):
        # Initialize pickers at random empty positions
        for i in range(self.n_pickers):
            attempts = 0
            while attempts < 1000:
                x = np.random.randint(1, self.width - 1)
                y = np.random.randint(1, self.height - 1)
                if self.grid[y, x] == CellType.EMPTY.value:
                    is_forklift = i < int(self.n_pickers * self.forklift_ratio)
                    picker = Picker(
                        picker_id=i,
                        x=x,
                        y=y,
                        is_forklift=is_forklift
                    )
                    self.pickers.append(picker)
                    break
                attempts += 1
            
            if attempts >= 1000:
                print(f"Warning: Could not find empty position for picker {i}")
                    
    def reset(self) -> Dict[str, np.ndarray]:
        self.time = 0.0
        self.total_steps = 0
        self.orders = []
        self.completed_orders = []
        self.next_order_id = 0  # Reset order counter
        
        # Reset pickers
        for picker in self.pickers:
            picker.carrying_items = []
            picker.current_order = None
            picker.battery = 100.0
            
        # Reset metrics
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = []
            elif isinstance(self.metrics[key], defaultdict):
                self.metrics[key] = defaultdict(list)
            elif key == 'zone_queues':
                self.metrics[key] = defaultdict(list)
            else:
                self.metrics[key] = 0
                
        return self.get_observation()
        
    def get_observation(self) -> Dict[str, np.ndarray]:
        # Global state for manager
        global_grid = np.copy(self.grid)
        
        # Mark picker positions
        picker_positions = np.zeros((self.height, self.width))
        for picker in self.pickers:
            picker_positions[picker.y, picker.x] = 1
            
        # Order queue information
        order_features = []
        for order in self.orders[:20]:  # Limit to 20 most recent orders
            features = [
                order.arrival_time,
                order.deadline,
                order.priority,
                len(order.items),
                1 if order.assigned_picker is not None else 0
            ]
            order_features.append(features)
            
        if len(order_features) < 20:
            order_features.extend([[0] * 5] * (20 - len(order_features)))
            
        order_features = np.array(order_features, dtype=np.float32)
        
        # Picker states
        picker_states = []
        for picker in self.pickers:
            state = [
                picker.x / self.width,
                picker.y / self.height,
                len(picker.carrying_items) / self.max_items_per_order,
                picker.battery / 100.0,
                1 if picker.current_order is not None else 0,
                1 if picker.is_forklift else 0
            ]
            picker_states.append(state)
            
        picker_states = np.array(picker_states, dtype=np.float32)
        
        return {
            'global_grid': global_grid,
            'picker_positions': picker_positions,
            'order_features': order_features,
            'picker_states': picker_states,
            'time': np.array([self.time], dtype=np.float32)
        }
        
    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict[int, float], Dict[int, bool], Dict]:
        rewards = {}
        dones = {}
        
        # Check for congestion before actions
        congestion_detected = self._detect_congestion()
        if congestion_detected:
            self.metrics['congestion_time'] += 0.1  # Add step duration to congestion time
        
        # Execute picker actions
        for picker_id, action in actions.items():
            if picker_id < len(self.pickers):
                reward = self._execute_picker_action(picker_id, action)
                rewards[picker_id] = reward
                dones[picker_id] = False
                
        # Generate new orders
        if np.random.random() < self.order_arrival_rate:
            self._generate_order()
            
        # Update time and metrics
        self.time += 0.1  # 100ms per step
        self.total_steps += 1
        
        # Check for completed orders
        self._check_order_completions()
        
        # Calculate global metrics
        self._update_metrics()
        
        obs = self.get_observation()
        info = {'metrics': self.metrics.copy()}
        
        return obs, rewards, dones, info
        
    def _execute_picker_action(self, picker_id: int, action: int) -> float:
        picker = self.pickers[picker_id]
        # 基础奖励：区分有无订单
        if picker.current_order is None:
            # 无订单时IDLE不惩罚，其他动作轻微惩罚
            reward = 0.0 if action == ActionType.IDLE.value else -0.005
        else:
            # 有订单时的时间惩罚
            reward = -0.01
        
        if action == ActionType.IDLE.value:
            pass  # No movement
            
        elif action == ActionType.MOVE_UP.value:
            old_x, old_y = picker.x, picker.y
            new_x, new_y = picker.x, picker.y - 1
            if self._is_valid_position(new_x, new_y, picker_id):
                picker.x = new_x
                picker.y = new_y
                self.metrics['total_distance'] += 1
                # 添加距离奖励
                if picker.current_order is not None:
                    reward += self._calculate_distance_reward(picker, old_x, old_y, new_x, new_y)
            else:
                reward -= 0.2  # 增大碰撞惩罚
                
        elif action == ActionType.MOVE_DOWN.value:
            old_x, old_y = picker.x, picker.y
            new_x, new_y = picker.x, picker.y + 1
            if self._is_valid_position(new_x, new_y, picker_id):
                picker.x = new_x
                picker.y = new_y
                self.metrics['total_distance'] += 1
                # 添加距离奖励
                if picker.current_order is not None:
                    reward += self._calculate_distance_reward(picker, old_x, old_y, new_x, new_y)
            else:
                reward -= 0.2  # 增大碰撞惩罚
                
        elif action == ActionType.MOVE_LEFT.value:
            old_x, old_y = picker.x, picker.y
            new_x, new_y = picker.x - 1, picker.y
            if self._is_valid_position(new_x, new_y, picker_id):
                picker.x = new_x
                picker.y = new_y
                self.metrics['total_distance'] += 1
                # 添加距离奖励
                if picker.current_order is not None:
                    reward += self._calculate_distance_reward(picker, old_x, old_y, new_x, new_y)
            else:
                reward -= 0.2  # 增大碰撞惩罚
                
        elif action == ActionType.MOVE_RIGHT.value:
            old_x, old_y = picker.x, picker.y
            new_x, new_y = picker.x + 1, picker.y
            if self._is_valid_position(new_x, new_y, picker_id):
                picker.x = new_x
                picker.y = new_y
                self.metrics['total_distance'] += 1
                # 添加距离奖励
                if picker.current_order is not None:
                    reward += self._calculate_distance_reward(picker, old_x, old_y, new_x, new_y)
            else:
                reward -= 0.2  # 增大碰撞惩罚
            
        elif action == ActionType.PICK.value:
            if picker.current_order is None:
                # 没有订单时尝试PICK是错误的
                reward -= 0.3
            elif self._can_pick(picker):
                # Pick item from shelf
                picked_successfully = False
                # 找到当前订单
                current_order = None
                for order in self.orders:
                    if order.order_id == picker.current_order:
                        current_order = order
                        break
                
                if current_order:
                    # 检查当前位置是否有订单需要的物品
                    for shelf_pos in current_order.items:
                        # 必须在货架旁边（曼哈顿距离=1）
                        if abs(picker.x - shelf_pos[0]) + abs(picker.y - shelf_pos[1]) == 1:
                            if shelf_pos not in picker.carrying_items:
                                # 成功拾取物品
                                picker.carrying_items.append(shelf_pos)
                                picked_successfully = True
                                # 根据拾取进度给予递增奖励（大幅增加）
                                progress = len(picker.carrying_items) / len(current_order.items)
                                reward = 10.0 + 10.0 * progress  # 10-20的奖励（原来0.5-1.0）
                                # 如果拾取了所有物品，额外奖励
                                if len(picker.carrying_items) == len(current_order.items):
                                    reward += 10.0  # 额外10分（原来0.5）
                                break
                
                if not picked_successfully:
                    # 在错误位置尝试PICK
                    reward -= 0.3
            else:
                # 不能PICK时尝试
                reward -= 0.3
                
        elif action == ActionType.DROP.value:
            if len(picker.carrying_items) == 0:
                # 没有物品时尝试DROP是错误的
                reward -= 0.3
            elif not self._at_station(picker):
                # 不在站点时尝试DROP是错误的
                reward -= 0.3
            else:
                # 在站点且有物品
                if picker.current_order is not None:
                    # 找到当前订单
                    for order in self.orders:
                        if order.order_id == picker.current_order:
                            # 检查是否收集了所有物品
                            if len(picker.carrying_items) >= len(order.items):
                                # 完成订单！最大奖励（大幅增加）
                                reward = 100.0  # 原来5.0 -> 100.0
                                # 清空携带物品
                                picker.carrying_items = []
                                # 标记订单完成
                                order.completion_time = self.time
                                self.completed_orders.append(order)
                                picker.current_order = None
                                # 额外奖励：基于完成速度
                                if self.time - order.arrival_time < 50:
                                    reward += 20.0  # 快速完成额外奖励（原来1.0 -> 20.0）
                            else:
                                # 物品不全就DROP，惩罚
                                reward -= 0.5
                            break
                else:
                    # 没有订单也DROP
                    reward += 0.1 * len(picker.carrying_items)
                    picker.carrying_items = []
                
        # Battery consumption
        picker.battery -= 0.01
        if picker.battery < 20:
            reward -= 0.5  # Low battery penalty
            
        return reward
        
        
    def _is_valid_position(self, x: int, y: int, picker_id: int) -> bool:
        # Check boundaries
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
            
        # Check walls and shelves
        if self.grid[y, x] in [CellType.WALL.value, CellType.SHELF.value]:
            return False
            
        # Check other pickers
        for i, other in enumerate(self.pickers):
            if i != picker_id and other.x == x and other.y == y:
                self.metrics['near_collisions'] += 1
                return False
                
        return True
        
    def _can_pick(self, picker: Picker) -> bool:
        # Check if picker is at or adjacent to a shelf
        # 检查当前位置和相邻位置
        positions_to_check = [(picker.x, picker.y)]  # 包括当前位置
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = picker.x + dx, picker.y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                positions_to_check.append((nx, ny))
        
        # 检查这些位置是否是货架
        for pos in positions_to_check:
            if pos in self.shelves:  # shelves存储的是货架位置列表
                return len(picker.carrying_items) < self.max_items_per_order
        return False
        
    def _at_station(self, picker: Picker) -> bool:
        return (picker.x, picker.y) in self.stations
    
    def _calculate_distance_reward(self, picker: Picker, old_x: int, old_y: int, new_x: int, new_y: int) -> float:
        """计算移动后的距离奖励"""
        # 找到当前订单
        current_order = None
        for order in self.orders:
            if order.order_id == picker.current_order:
                current_order = order
                break
        
        if not current_order:
            return 0.0
        
        # 计算目标位置
        if len(picker.carrying_items) == 0:
            # 还没拿货，目标是最近的待拾取物品
            remaining_items = [item for item in current_order.items 
                             if item not in picker.carrying_items]
            if remaining_items:
                target = min(remaining_items, 
                           key=lambda i: abs(i[0]-old_x) + abs(i[1]-old_y))
                
                # 计算距离变化
                old_dist = abs(target[0] - old_x) + abs(target[1] - old_y)
                new_dist = abs(target[0] - new_x) + abs(target[1] - new_y)
                
                if new_dist < old_dist:
                    return 0.02  # 靠近目标货架
                elif new_dist > old_dist:
                    return -0.01  # 远离目标货架
        else:
            # 已经拿货，目标是最近的站点
            if self.stations:
                target = min(self.stations,
                           key=lambda s: abs(s[0]-old_x) + abs(s[1]-old_y))
                
                old_dist = abs(target[0] - old_x) + abs(target[1] - old_y)
                new_dist = abs(target[0] - new_x) + abs(target[1] - new_y)
                
                if new_dist < old_dist:
                    return 0.03  # 靠近站点的奖励更大
                elif new_dist > old_dist:
                    return -0.01  # 远离站点
        
        return 0.0
        
    def _generate_order(self):
        # Check if shelves are available
        if len(self.shelves) == 0:
            return  # No shelves, cannot generate orders
            
        n_items = np.random.randint(1, min(self.max_items_per_order + 1, len(self.shelves) + 1))
        items = []
        for _ in range(n_items):
            shelf_idx = np.random.randint(0, len(self.shelves))
            items.append(self.shelves[shelf_idx])
            
        order = Order(
            order_id=self.next_order_id,
            items=items,
            arrival_time=self.time,
            deadline=self.time + np.random.uniform(60, 300),  # 1-5 minutes deadline
            priority=np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1]),
            requires_forklift=np.random.random() < 0.2  # 20% of orders require forklift
        )
        self.next_order_id += 1  # Increment global counter
        self.orders.append(order)
        
    def _check_order_completions(self):
        # 注释掉自动完成机制 - 必须通过DROP动作完成订单
        # for order in self.orders:
        #     if order.completion_time is None and order.assigned_picker is not None:
        #         picker = self.pickers[order.assigned_picker]
        #         if self._at_station(picker) and len(picker.carrying_items) >= len(order.items):
        #             order.completion_time = self.time
        #             self.completed_orders.append(order)
        #             picker.current_order = None
        pass  # 不自动完成，必须通过DROP动作
                    
    def _update_metrics(self):
        if len(self.completed_orders) > 0:
            wait_times = [o.completion_time - o.arrival_time for o in self.completed_orders]
            self.metrics['mean_wait_time'] = np.mean(wait_times)
            self.metrics['tail_latency_95'] = np.percentile(wait_times, 95)
            self.metrics['completion_rate'] = len(self.completed_orders) / max(1, len(self.orders))
            
        # Throughput (orders per minute)
        if self.time > 0:
            self.metrics['throughput'] = len(self.completed_orders) / (self.time / 60)
            # Normalize near_collisions to per hour rate
            self.metrics['near_collisions_per_hour'] = (self.metrics['near_collisions'] / self.time) * 3600
            
        # Update congestion time
        self._update_congestion_metrics()
        
        # Calculate per-zone balance (queue CV)
        self._update_zone_balance_metrics()
            
    def compute_shortest_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        # A* pathfinding
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
                
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                    continue
                    
                if self.grid[neighbor[1], neighbor[0]] in [CellType.WALL.value, CellType.SHELF.value]:
                    continue
                    
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return []  # No path found
    
    def _detect_congestion(self) -> bool:
        """Detect if there is congestion (multiple pickers in same small area)"""
        congestion_threshold = 3  # More than 3 pickers in same zone
        zone_radius = 5  # 5x5 area
        
        # Count pickers in each zone
        zone_counts = defaultdict(int)
        for picker in self.pickers:
            zone_x = picker.x // zone_radius
            zone_y = picker.y // zone_radius
            zone_counts[(zone_x, zone_y)] += 1
        
        # Check if any zone exceeds threshold
        for count in zone_counts.values():
            if count >= congestion_threshold:
                return True
        return False
    
    def _update_congestion_metrics(self):
        """Update congestion-related metrics"""
        # Track congestion events for analysis
        if self._detect_congestion():
            self.metrics['congestion_events'].append(self.time)
    
    def _get_picker_zone(self, picker: Picker) -> int:
        """Get zone ID for a picker's position"""
        zone_x = min(picker.x // self.zone_width, 1)
        zone_y = min(picker.y // self.zone_height, 1)
        return zone_y * 2 + zone_x  # Returns 0, 1, 2, or 3
    
    def _update_zone_balance_metrics(self):
        """Calculate per-zone balance using coefficient of variation (CV)"""
        # Clear zone queues for current timestep
        zone_orders = defaultdict(list)
        
        # Assign unassigned orders to zones based on item locations
        for order in self.orders:
            if order.assigned_picker is None and order.completion_time is None:
                # Determine zone based on first item location
                if order.items:
                    item_x, item_y = order.items[0]
                    zone_x = min(item_x // self.zone_width, 1)
                    zone_y = min(item_y // self.zone_height, 1)
                    zone_id = zone_y * 2 + zone_x
                    zone_orders[zone_id].append(order.order_id)
        
        # Store zone queue lengths
        queue_lengths = []
        for zone_id in range(4):
            queue_length = len(zone_orders[zone_id])
            queue_lengths.append(queue_length)
            self.metrics['zone_queues'][zone_id] = zone_orders[zone_id]
        
        # Calculate coefficient of variation (CV)
        if len(queue_lengths) > 0 and np.mean(queue_lengths) > 0:
            cv = np.std(queue_lengths) / np.mean(queue_lengths)
            self.metrics['zone_balance_cv'] = cv
        else:
            self.metrics['zone_balance_cv'] = 0.0