"""
订单生成模块 - 使用非齐次泊松过程模拟真实订单流
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Order:
    """订单定义"""
    order_id: int
    arrival_time: float
    priority: float  # 0-1, 1为最高优先级
    items: List[Dict]  # 物品列表 [{shelf_id, item_id, quantity}]
    order_type: str  # 'regular', 'urgent', 'bulk'
    weight_class: str  # 订单内物品的重量类型：'heavy', 'medium', 'light'（与订单类型解耦）
    customer_zone: int  # 客户区域
    deadline: Optional[float] = None  # 订单截止时间

class NonHomogeneousPoissonOrderGenerator:
    """
    非齐次泊松过程订单生成器
    模拟真实仓库的订单流，包括高峰期和低谷期
    """
    
    def __init__(self, config: Dict):
        """
        初始化订单生成器
        
        Args:
            config: 配置字典，包含：
                - base_rate: 基础订单到达率（订单/小时）
                - peak_hours: 高峰时段列表，如[(9,12), (14,17)]
                - peak_multiplier: 高峰期乘数（默认1.6）
                - off_peak_multiplier: 低谷期乘数（默认0.7）
                - urgent_order_prob: 紧急订单概率
                - bulk_order_prob: 大批量订单概率
                - simulation_hours: 模拟时长（小时）
        """
        self.base_rate = config.get('base_rate', 30)  # 基础30订单/小时
        self.peak_hours = config.get('peak_hours', [(9, 12), (14, 17), (19, 21)])
        self.peak_multiplier = config.get('peak_multiplier', 1.6)
        self.off_peak_multiplier = config.get('off_peak_multiplier', 0.7)
        self.urgent_order_prob = config.get('urgent_order_prob', 0.15)
        self.bulk_order_prob = config.get('bulk_order_prob', 0.10)
        self.simulation_hours = config.get('simulation_hours', 24)
        # 模式周期与缩放：
        # - pattern_period_hours：到达率模式的周期长度（默认按 24 小时一日）
        # - scale_pattern_to_simulation：若为 True，则将仿真时间 [0, simulation_hours]
        #   等比映射到一个完整的模式周期 [0, pattern_period_hours] 上，
        #   从而在任意仿真时长内都能“走完”一遍到达模式。
        self.pattern_period_hours = config.get('pattern_period_hours', 24)
        self.scale_pattern_to_simulation = bool(config.get('scale_pattern_to_simulation', False))
        # Deadline 配置（可由外部覆盖）。范围单位：小时。
        # 默认：urgent 0.5-1.0h，regular 1.0-2.0h，bulk 2.0-4.0h
        self.deadline_ranges = config.get('deadline_ranges', {
            'urgent': (0.5, 1.0),
            'regular': (1.0, 2.0),
            'bulk': (2.0, 4.0),
        })
        # 可选缩放因子：整体拉伸/压缩截止期
        self.deadline_scale = float(config.get('deadline_scale', 1.0))
        
        # SKU分布（帕累托分布：20%的SKU占80%的订单）
        self.n_skus = config.get('n_skus', 1000)
        self.setup_sku_distribution()
        
        # 订单计数器
        self.order_counter = 0
        self.current_time = 0.0
        
        # 预生成订单流（可选）
        self.pregenerated_orders = []
        self.use_pregeneration = config.get('use_pregeneration', True)
        
        if self.use_pregeneration:
            self.pregenerate_orders()
    
    def setup_sku_distribution(self):
        """设置SKU流行度分布（帕累托分布）"""
        # 生成帕累托分布的SKU流行度
        alpha = 1.5  # 帕累托指数
        sku_popularities = np.random.pareto(alpha, self.n_skus)
        sku_popularities = sku_popularities / sku_popularities.sum()
        
        # 排序SKU按流行度
        self.sku_popularities = np.sort(sku_popularities)[::-1]
        
        # 将SKU分配到不同货架区域
        # 热门SKU放在靠近站点的位置
        self.sku_locations = {}
        for i in range(self.n_skus):
            # 将热门SKU优先放在 Zone 2（左下），其余按原策略分布
            if i < self.n_skus * 0.2:  # 前20%热门SKU
                zone = 2  # 左下区
            elif i < self.n_skus * 0.5:  # 中等流行
                zone = 1
            else:  # 长尾SKU：更均匀地分布到四个区域
                zone = int(np.random.randint(0, 4))
            
            self.sku_locations[i] = {
                'zone': zone,
                'shelf_id': np.random.randint(zone * 5, (zone + 1) * 5),
                'popularity': self.sku_popularities[i]
            }
    
    def _base_arrival_rate(self, hour_in_period: float) -> float:
        """返回在模式周期内的基础到达率（不考虑仿真时长缩放）。

        参数 hour_in_period 应位于 [0, pattern_period_hours)。
        """
        hour_of_period = hour_in_period % max(1e-6, float(self.pattern_period_hours))
        # 检查是否在高峰时段（peak_hours 应以同一周期单位定义，默认按 24 小时）
        for start, end in self.peak_hours:
            if start <= hour_of_period < end:
                return self.base_rate * self.peak_multiplier
        # 夜间（22:00 - 6:00）为低谷期（若周期非24h，仍按数值区间处理）
        if hour_of_period >= 22 or hour_of_period < 6:
            return self.base_rate * self.off_peak_multiplier * 0.5
        return self.base_rate

    def get_arrival_rate(self, time_hour: float) -> float:
        """
        获取特定时间的订单到达率（支持按仿真时长缩放模式周期）
        
        Args:
            time_hour: 时间（小时），0-24循环
            
        Returns:
            该时间点的订单到达率（订单/小时）
        """
        # 将仿真时间映射到模式周期坐标
        if self.scale_pattern_to_simulation and float(self.simulation_hours) > 0:
            # 先对仿真时长取模，再等比映射到一个完整周期
            t_sim = time_hour % float(self.simulation_hours)
            hour_in_period = (t_sim / float(self.simulation_hours)) * float(self.pattern_period_hours)
        else:
            hour_in_period = time_hour % float(self.pattern_period_hours)
        return float(self._base_arrival_rate(hour_in_period))
    
    def generate_order_items(self, order_type: str, weight_class: str) -> List[Dict]:
        """
        生成订单物品列表
        
        Args:
            order_type: 订单类型
            
        Returns:
            物品列表（每个物品带数值重量 weight，按 weight_class 的范围采样）
        """
        items = []
        # 按 weight_class 定义数值重量范围
        ranges = {
            'light': (1.0, 30.0),
            'medium': (30.0, 70.0),
            'heavy': (70.0, 100.0),
        }
        low, high = ranges.get(weight_class, (1.0, 100.0))
        
        if order_type == 'bulk':
            # 大批量订单：5-20个物品
            n_items = np.random.randint(5, 21)
        elif order_type == 'urgent':
            # 紧急订单：1-3个物品
            n_items = np.random.randint(1, 4)
        else:
            # 常规订单：1-5个物品
            n_items = np.random.poisson(2) + 1  # 泊松分布，平均3个
            n_items = min(n_items, 5)
        
        # 根据SKU流行度选择物品
        for _ in range(n_items):
            # 使用指数分布偏向选择热门SKU
            sku_idx = min(int(np.random.exponential(self.n_skus * 0.2)), self.n_skus - 1)
            sku_info = self.sku_locations[sku_idx]
            # 按 weight_class 设定 value 取值范围（整数）
            if weight_class == 'light':
                v_low, v_high = 1, 50
            elif weight_class == 'medium':
                v_low, v_high = 10, 100
            else:  # heavy
                v_low, v_high = 50, 100
            
            items.append({
                'sku_id': sku_idx,
                'shelf_id': sku_info['shelf_id'],
                'zone': sku_info['zone'],
                'quantity': np.random.randint(1, 4) if order_type == 'bulk' else 1,
                # 为每个物品在所属 weight_class 的范围内生成数值重量（两位小数）
                'weight': float(np.round(np.random.uniform(low, high), 2)),
                # 为每个物品生成价值（整数）
                'value': int(np.random.randint(v_low, v_high + 1)),
            })
        
        return items
    
    def generate_single_order(self, arrival_time: float) -> Order:
        """
        生成单个订单
        
        Args:
            arrival_time: 订单到达时间
            
        Returns:
            Order对象
        """
        # 确定订单类型
        rand = np.random.random()
        if rand < self.urgent_order_prob:
            order_type = 'urgent'
            priority = np.random.uniform(0.8, 1.0)
        elif rand < self.urgent_order_prob + self.bulk_order_prob:
            order_type = 'bulk'
            priority = np.random.uniform(0.3, 0.5)
        else:
            order_type = 'regular'
            priority = np.random.uniform(0.4, 0.7)

        # 依据配置计算截止期（支持自定义范围与整体缩放）
        d_low, d_high = self.deadline_ranges.get(order_type, (1.0, 2.0))
        d_low = float(d_low)
        d_high = float(d_high)
        if d_high < d_low:
            d_low, d_high = d_high, d_low
        duration = float(np.random.uniform(d_low, d_high)) * max(1e-6, self.deadline_scale)
        deadline = arrival_time + duration
        
        # 随机确定重量类型（不由订单类型决定）
        r2 = np.random.random()
        if r2 < self.urgent_order_prob:  # 保持与生成器可调参数一致（或自定义分布）
            # 也可直接使用固定比例，这里示例：heavy 15%
            pass
        # 采用固定比例：heavy 15%，medium 60%，light 25%
        if r2 < 0.15:
            weight_class = 'heavy'
        elif r2 < 0.75:
            weight_class = 'medium'
        else:
            weight_class = 'light'

        # 生成订单物品（数量由订单类型决定），并按 weight_class 采样数值重量
        items = self.generate_order_items(order_type, weight_class)
        
        # 确定客户区域（影响配送站点选择）
        customer_zone = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2])

        order = Order(
            order_id=self.order_counter,
            arrival_time=arrival_time,
            priority=priority,
            items=items,
            order_type=order_type,
            weight_class=weight_class,
            customer_zone=customer_zone,
            deadline=deadline
        )
        
        self.order_counter += 1
        return order
    
    def pregenerate_orders(self):
        """
        预生成整个模拟期间的订单
        使用非齐次泊松过程
        """
        self.pregenerated_orders = []
        current_time = 0.0
        
        while current_time < self.simulation_hours:
            # 获取当前时间的到达率
            rate = self.get_arrival_rate(current_time)
            
            # 生成下一个订单的间隔时间（指数分布）
            inter_arrival_time = np.random.exponential(1.0 / rate)
            current_time += inter_arrival_time
            
            if current_time < self.simulation_hours:
                order = self.generate_single_order(current_time)
                self.pregenerated_orders.append(order)
        
        print(f"预生成{len(self.pregenerated_orders)}个订单，"
              f"平均到达率: {len(self.pregenerated_orders)/self.simulation_hours:.1f}订单/小时")
    
    def get_orders_in_window(self, start_time: float, end_time: float) -> List[Order]:
        """
        获取时间窗口内的订单
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            该时间窗口内的订单列表
        """
        if self.use_pregeneration:
            return [o for o in self.pregenerated_orders 
                   if start_time <= o.arrival_time < end_time]
        else:
            # 实时生成
            orders = []
            current = start_time
            while current < end_time:
                rate = self.get_arrival_rate(current)
                inter_arrival = np.random.exponential(1.0 / rate)
                current += inter_arrival
                if current < end_time:
                    orders.append(self.generate_single_order(current))
            return orders
    
    def get_next_order(self) -> Optional[Order]:
        """
        获取下一个订单（流式接口）
        
        Returns:
            下一个订单，如果没有则返回None
        """
        if self.use_pregeneration:
            # 找到下一个未处理的订单
            for order in self.pregenerated_orders:
                if order.arrival_time >= self.current_time:
                    self.current_time = order.arrival_time
                    return order
            return None
        else:
            # 实时生成下一个订单
            rate = self.get_arrival_rate(self.current_time)
            inter_arrival = np.random.exponential(1.0 / rate)
            self.current_time += inter_arrival
            
            if self.current_time < self.simulation_hours:
                return self.generate_single_order(self.current_time)
            return None
    
    def visualize_arrival_pattern(self, save_dir: str = 'results', filename: str = 'order_arrival_pattern.png'):
        """可视化订单到达模式，并新增“重量分布”展示。

        - 若启用 scale_pattern_to_simulation 且 simulation_hours>0：
          横坐标使用 [0, simulation_hours]，曲线采用 get_arrival_rate（与仿真一致）。
        - 否则：横坐标使用 [0, pattern_period_hours]，曲线采用基础模式 _base_arrival_rate。
        - 右侧新增第3个子图：基于已预生成订单的物品数值重量直方图（单位：相对计数）。
        """
        use_scaled = bool(self.scale_pattern_to_simulation) and float(self.simulation_hours) > 0
        if use_scaled:
            x_max = float(self.simulation_hours)
            hours = np.linspace(0, x_max, max(24, int(x_max * 10)))
            rates = [self.get_arrival_rate(h) for h in hours]
            x_label = 'Simulation Hours'
            title = 'Order Arrival Rate (scaled to simulation)'
        else:
            period = float(self.pattern_period_hours)
            hours = np.linspace(0, period, max(24, int(period * 10)))
            rates = [self._base_arrival_rate(h) for h in hours]
            x_label = 'Hour of Day'
            title = 'Order Arrival Rate Pattern'
        
        plt.figure(figsize=(18, 6))

        # 子图1：到达率曲线
        plt.subplot(1, 3, 1)
        plt.plot(hours, rates, 'b-', linewidth=2)
        plt.fill_between(hours, 0, rates, alpha=0.3)
        plt.xlabel(x_label)
        plt.ylabel('Arrival Rate (orders/hour)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        # 锁定横轴范围，避免标注或其它元素扩张坐标
        try:
            if use_scaled:
                plt.xlim(0, float(self.simulation_hours))
            else:
                plt.xlim(0, float(self.pattern_period_hours))
        except Exception:
            pass
        
        # 标记高峰时段：
        # - 未缩放：直接使用 peak_hours 区间
        # - 缩放：将 [start,end] 线性映射至 [0, simulation_hours]
        if not use_scaled:
            for start, end in self.peak_hours:
                plt.axvspan(start, end, alpha=0.2, color='red', label='Peak Hours')
        else:
            period = float(self.pattern_period_hours)
            simH = float(self.simulation_hours)
            if period > 0 and simH > 0:
                for start, end in self.peak_hours:
                    s = (float(start) / period) * simH
                    e = (float(end) / period) * simH
                    plt.axvspan(s, e, alpha=0.2, color='red', label='Peak (scaled)')
        
        # 子图2：订单类型分布
        plt.subplot(1, 3, 2)
        if self.pregenerated_orders:
            order_types = [o.order_type for o in self.pregenerated_orders]
            type_counts = {
                'regular': order_types.count('regular'),
                'urgent': order_types.count('urgent'),
                'bulk': order_types.count('bulk')
            }
            
            plt.pie(type_counts.values(), labels=type_counts.keys(), 
                   autopct='%1.1f%%', colors=['blue', 'red', 'green'])
            plt.title('Order Type Distribution')

        # 子图3：物品重量分布（直方图）
        plt.subplot(1, 3, 3)
        weights = []
        try:
            for o in self.pregenerated_orders:
                for it in getattr(o, 'items', []):
                    w = it.get('weight', None)
                    if w is not None:
                        weights.append(float(w))
        except Exception:
            pass
        if weights:
            import numpy as _np
            arr = _np.array(weights, dtype=_np.float32)
            # 使用固定范围 [0, 100] 与 20 个 bins；直方图显示相对频率
            bins = _np.linspace(0, 100, 21)
            plt.hist(arr, bins=bins, color='#8888ff', alpha=0.8, density=True, edgecolor='white')
            plt.xlim(0, 100)
            plt.xlabel('Item Weight')
            plt.ylabel('Density')
            plt.title('Item Weight Distribution')
            plt.grid(alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No weights available', ha='center', va='center')
            plt.axis('off')

        plt.tight_layout()
        # 仅保存到 results 目录，不在根目录放置副本
        import os as _os
        _os.makedirs(save_dir, exist_ok=True)
        out_path = _os.path.join(save_dir, filename)
        plt.savefig(out_path)
        print(f"订单到达模式图已保存到 {out_path}")

        return plt.gcf()
    
    def get_statistics(self) -> Dict:
        """获取订单生成统计信息"""
        if not self.pregenerated_orders:
            return {}
        
        stats = {
            'total_orders': len(self.pregenerated_orders),
            'avg_arrival_rate': len(self.pregenerated_orders) / self.simulation_hours,
            'order_types': {},
            'items_per_order': [],
            'zone_distribution': {}
        }
        
        for order in self.pregenerated_orders:
            # 订单类型统计
            stats['order_types'][order.order_type] = \
                stats['order_types'].get(order.order_type, 0) + 1
            
            # 物品数统计
            stats['items_per_order'].append(len(order.items))
            
            # 区域分布
            stats['zone_distribution'][order.customer_zone] = \
                stats['zone_distribution'].get(order.customer_zone, 0) + 1
        
        stats['avg_items_per_order'] = np.mean(stats['items_per_order'])
        stats['max_items_per_order'] = max(stats['items_per_order'])
        
        return stats


# 测试代码
if __name__ == "__main__":
    # 创建订单生成器
    config = {
        'base_rate': 40,  # 基础40订单/小时
        'peak_hours': [(9, 12), (14, 17), (19, 21)],  # 早高峰、午高峰、晚高峰
        'peak_multiplier': 1.6,
        'off_peak_multiplier': 0.7,
        'urgent_order_prob': 0.15,
        'bulk_order_prob': 0.10,
        'simulation_hours': 24,
        'n_skus': 1000
    }
    
    generator = NonHomogeneousPoissonOrderGenerator(config)
    
    # 获取统计信息
    stats = generator.get_statistics()
    print("\n订单生成统计:")
    print(f"  总订单数: {stats['total_orders']}")
    print(f"  平均到达率: {stats['avg_arrival_rate']:.1f} 订单/小时")
    print(f"  平均物品数/订单: {stats['avg_items_per_order']:.1f}")
    print(f"  订单类型分布: {stats['order_types']}")
    
    # 可视化
    generator.visualize_arrival_pattern()
    
    # 测试流式接口
    print("\n前10个订单:")
    generator.current_time = 0
    for i in range(10):
        order = generator.get_next_order()
        if order:
            print(f"  订单{order.order_id}: 时间={order.arrival_time:.2f}h, "
                  f"类型={order.order_type}, 物品数={len(order.items)}, "
                  f"优先级={order.priority:.2f}")
