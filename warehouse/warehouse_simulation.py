"""
仓库拣选仿真与数据合成工具链
基于 instruction.md 的完整实现
包含货位网格、SKU主数据、泊松订单、Nested-Logit分配、SimPy拣选轨迹和ROP补货
"""

import pandas as pd
import numpy as np
import simpy
import networkx as nx
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
from datetime import datetime
from nested_logit_model import NestedLogitModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WarehouseSimulation:
    def __init__(self, config=None):
        """初始化仓库仿真参数"""
        # 默认参数
        self.config = {
            # 仓库网格尺寸
            'GRID_X': 32,
            'GRID_Y': 20,
            'GRID_Z': 3,
            
            # SKU和库存参数
            'N_SKU': 1000,
            'ABC_SPLIT': (0.2, 0.3, 0.5),  # A/B/C SKU比例
            'ABC_QTY': (400, 120, 40),      # A/B/C 初始件数
            
            # 仿真参数
            'SIM_TIME': 8*60*60,            # 8小时
            'ORDER_LAMBDA': 2/60,           # 2单/分钟 (泊松到达率)
            'ZIPF_A': 1.2,                  # Zipf分布参数
            
            # 设备和人员
            'N_PICKERS': 6,
            'FORKLIFT_RATIO': 0.15,         # 需要叉车的slot比例
            
            # 操作时间参数
            'WALK_SPEED': 1,                # 1格/秒
            'PICK_TIME': 5,                 # 5秒/件
            'LOAD_TIME': 8,                 # 装车时间
            
            # 补货参数
            'ROP_SAFETY_DAYS': 0.5,         # 安全库存天数
            'LEAD_TIME_MEAN': 2*60*60,      # 平均补货提前期2小时
            'LEAD_TIME_STD': 30*60,         # 补货提前期标准差30分钟
            'REPLENISH_QTY_FACTOR': 1.5,    # 补货数量系数
            
            # 聚类参数
            'N_CLUSTERS': 8,
            
            # 随机种子
            'RANDOM_SEED': 42
        }
        
        # 更新配置
        if config:
            self.config.update(config)
            
        # 初始化随机数生成器
        self.rng = np.random.default_rng(self.config['RANDOM_SEED'])
        
        # 计算派生参数
        self.n_slots = self.config['GRID_X'] * self.config['GRID_Y'] * self.config['GRID_Z']
        
        # 数据存储
        self.slots = None
        self.sku_df = None
        self.orders = None
        self.assign = None
        self.picks = None
        self.replenishment_log = []
        
    def generate_slots(self):
        """生成货位拓扑"""
        logger.info("生成货位拓扑...")
        x, y, z = np.meshgrid(
            np.arange(self.config['GRID_X']), 
            np.arange(self.config['GRID_Y']), 
            np.arange(self.config['GRID_Z'])
        )
        
        self.slots = pd.DataFrame({
            'slot_id': np.arange(self.n_slots),
            'x': x.flatten(),
            'y': y.flatten(),
            'z': z.flatten(),
            'max_capacity': 999,
            'level': z.flatten()  # 层级信息
        })
        
        self.slots.to_csv("slots.csv", index=False)
        logger.info(f"生成 {self.n_slots} 个货位")
        
    def generate_sku_master(self):
        """生成SKU主数据"""
        logger.info("生成SKU主数据...")
        self.sku_df = pd.DataFrame({'sku_id': np.arange(self.config['N_SKU'])})
        
        # ABC分类
        abc_bins = [-1, 
                    int(self.config['N_SKU'] * self.config['ABC_SPLIT'][0]),
                    int(self.config['N_SKU'] * (self.config['ABC_SPLIT'][0] + self.config['ABC_SPLIT'][1])),
                    self.config['N_SKU']]
        
        self.sku_df['class'] = pd.cut(
            self.sku_df['sku_id'],
            bins=abc_bins,
            labels=['A', 'B', 'C']
        )
        
        # 初始数量
        self.sku_df['init_qty'] = self.sku_df['class'].map({
            'A': self.config['ABC_QTY'][0],
            'B': self.config['ABC_QTY'][1],
            'C': self.config['ABC_QTY'][2]
        })
        
        # 添加更真实的随机变化
        qty_variation = self.rng.normal(1.0, 0.1, size=len(self.sku_df))
        base_qty = self.sku_df['init_qty'].astype(float)
        self.sku_df['init_qty'] = (base_qty * np.clip(qty_variation, 0.8, 1.2)).astype(int)
        
        # 尺寸参数 (使用三角分布模拟真实商品尺寸)
        self.sku_df['height'] = np.clip(
            self.rng.triangular(left=5, mode=20, right=120, size=self.config['N_SKU']),
            5, 120
        ).astype(int)
        
        self.sku_df['length'] = np.clip(
            self.rng.triangular(left=5, mode=20, right=120, size=self.config['N_SKU']),
            5, 120
        ).astype(int)
        
        self.sku_df['width'] = np.clip(
            self.rng.triangular(left=5, mode=20, right=120, size=self.config['N_SKU']),
            5, 120
        ).astype(int)
        
        # 计算体积和重量
        self.sku_df['volume'] = self.sku_df['height'] * self.sku_df['length'] * self.sku_df['width']
        self.sku_df['weight'] = (self.sku_df['volume'] * 0.001 * self.rng.uniform(0.5, 2.0, size=self.config['N_SKU'])).astype(int)
        
        # 添加SKU需求频率估计（用于ROP计算）
        self.sku_df['daily_demand'] = self.sku_df['class'].map({
            'A': self.rng.uniform(50, 100),
            'B': self.rng.uniform(20, 50),
            'C': self.rng.uniform(5, 20)
        })
        
        self.sku_df.to_csv("sku_master.csv", index=False)
        logger.info(f"生成 {self.config['N_SKU']} 个SKU")
        
    def initial_inventory_allocation(self):
        """初始库存分配和货位优化(Slotting)"""
        logger.info("执行初始库存分配和货位优化...")
        
        # 随机分配SKU到货位
        self.slots['sku'] = self.rng.choice(self.sku_df['sku_id'], size=self.n_slots)
        self.slots['quantity'] = self.slots['sku'].map(
            self.sku_df.set_index('sku_id')['init_qty']
        )
        
        # Velocity-based slotting: 将A类SKU放到前1/3行
        high_demand_ids = self.sku_df.query("`class`=='A'")['sku_id'].values
        front_rows = self.slots.query("y < @self.config['GRID_Y']/3").index
        
        if len(front_rows) > 0 and len(high_demand_ids) > 0:
            self.slots.loc[front_rows, 'sku'] = self.rng.choice(
                high_demand_ids, 
                size=len(front_rows)
            )
            # 更新前排货位的数量
            self.slots['quantity'] = self.slots['sku'].map(
                self.sku_df.set_index('sku_id')['init_qty']
            )
        
        # 保存初始库存
        inventory_day0 = self.slots[['slot_id', 'sku', 'quantity']].copy()
        inventory_day0.to_csv("inventory_day0.csv", index=False)
        
        # 保存分配的货位信息
        assigned_slots = self.slots[['slot_id', 'x', 'y', 'z', 'sku']].copy()
        assigned_slots.to_csv("assigned_slots.csv", index=False)
        
        logger.info("库存分配完成")
        
    def generate_orders(self):
        """生成订单数据"""
        logger.info("生成订单数据...")
        orders_list = []
        t = 0.0
        order_idx = 0
        
        while t < self.config['SIM_TIME']:
            # 泊松到达
            t += self.rng.exponential(1/self.config['ORDER_LAMBDA'])
            if t >= self.config['SIM_TIME']:
                break
                
            # 订单行数
            n_lines = self.rng.integers(1, 6)
            
            # SKU按Zipf分布抽样
            skus = self.rng.zipf(a=self.config['ZIPF_A'], size=n_lines) % self.config['N_SKU']
            
            # SLA时间
            sla_sec = self.rng.choice([30*60, 60*60, 2*60*60], p=[0.3, 0.5, 0.2])
            
            # 设备需求
            eq_flag = self.rng.random() < self.config['FORKLIFT_RATIO']
            
            orders_list.append({
                'order_id': f"O{order_idx:05d}",
                'create_ts': t,
                'sku_list': list(map(int, skus)),
                'eq_flag': int(eq_flag),
                'sla_sec': int(sla_sec),
                'priority': self.rng.choice(['high', 'normal', 'low'], p=[0.2, 0.6, 0.2])
            })
            order_idx += 1
            
        self.orders = pd.DataFrame(orders_list)
        logger.info(f"生成 {len(self.orders)} 个订单")
        
    def nested_logit_assignment(self):
        """使用正确的Nested-Logit模型进行任务分配"""
        logger.info("执行Nested-Logit任务分配...")
        
        # K-means聚类创建空间巢（用于确定订单的初始位置）
        coords = self.slots[['x', 'y']].values
        clusters = KMeans(
            n_clusters=self.config['N_CLUSTERS'], 
            random_state=self.config['RANDOM_SEED']
        ).fit(coords)
        self.slots['cluster'] = clusters.labels_
        
        # 为订单添加更多属性
        # 初始化新列
        self.orders['pickup_location'] = None
        self.orders['cluster_id'] = 0
        
        for idx, order in self.orders.iterrows():
            # 根据订单中的SKU确定主要拣选位置
            main_sku = order['sku_list'][0] if order['sku_list'] else 0
            sku_slots = self.slots[self.slots['sku'] == main_sku]
            
            if not sku_slots.empty:
                # 选择最近的货位
                slot = sku_slots.iloc[0]
                self.orders.at[idx, 'pickup_location'] = (int(slot['x']), int(slot['y']))
                self.orders.at[idx, 'cluster_id'] = int(slot['cluster'])
            else:
                # 随机分配
                self.orders.at[idx, 'pickup_location'] = (
                    int(self.rng.integers(0, self.config['GRID_X'])),
                    int(self.rng.integers(0, self.config['GRID_Y']))
                )
                self.orders.at[idx, 'cluster_id'] = int(self.rng.integers(0, self.config['N_CLUSTERS']))
        
        # 初始化拣选员信息
        pickers = []
        picker_states = {}
        
        for i in range(self.config['N_PICKERS']):
            picker_id = f"P{i}"
            
            # 拣选员属性
            picker = {
                'id': picker_id,
                'has_forklift': self.rng.random() < 0.3,  # 30%的拣选员有叉车
                'max_capacity': self.rng.integers(10, 20),
                'skill_level': self.rng.uniform(0.5, 1.5)  # 技能水平
            }
            pickers.append(picker)
            
            # 拣选员状态
            picker_states[picker_id] = {
                'position': (0, 0),  # 初始位置在原点
                'current_workload': 0,
                'max_workload': 10,
                'assigned_orders': []
            }
        
        # 创建Nested Logit模型
        nl_model = NestedLogitModel(config={
            'params': {
                'lambda': 0.6,  # 巢内相关系数
                'w_distance': -0.4,
                'w_workload': -0.5,
                'w_priority': 1.2,
                'w_equipment': 3.0,  # 设备匹配很重要
                'w_time_pressure': 1.5
            }
        })
        
        # 使用Nested Logit模型分配订单
        self.assign = nl_model.assign_orders(
            self.orders,
            pickers,
            picker_states
        )
        
        # 添加额外字段以兼容原系统
        if 'picked_flag' not in self.assign.columns:
            self.assign['picked_flag'] = 0
            
        logger.info(f"使用Nested Logit模型分配了 {len(self.assign)} 个任务")
        
        # 输出分配统计
        if len(self.assign) > 0:
            nest_dist = self.assign['nest_id'].value_counts()
            logger.info("按巢分配的任务数：")
            for nest, count in nest_dist.items():
                logger.info(f"  {nest}: {count} 任务")
                
            # 计算平均概率
            avg_nest_prob = self.assign['nest_prob'].mean()
            avg_picker_prob = self.assign['picker_prob'].mean()
            logger.info(f"平均巢选择概率: {avg_nest_prob:.3f}")
            logger.info(f"平均拣选员条件概率: {avg_picker_prob:.3f}")
        
    def calculate_rop(self, sku_id):
        """计算再订货点(ROP)"""
        sku_info = self.sku_df[self.sku_df['sku_id'] == sku_id].iloc[0]
        daily_demand = sku_info['daily_demand']
        
        # ROP = 需求 × 提前期 + 安全库存
        lead_time_days = self.config['LEAD_TIME_MEAN'] / (24 * 60 * 60)
        safety_stock = daily_demand * self.config['ROP_SAFETY_DAYS']
        rop = daily_demand * lead_time_days + safety_stock
        
        return int(rop)
    
    def run_simulation(self):
        """运行SimPy仿真"""
        logger.info("开始运行仿真...")
        
        # 创建网格图
        G = nx.grid_2d_graph(self.config['GRID_X'], self.config['GRID_Y'])
        
        # 初始化仿真环境
        env = simpy.Environment()
        
        # 库存字典
        inv = self.slots.set_index('slot_id')['quantity'].to_dict()
        
        # 拣选日志
        pick_log = []
        
        # A*启发式函数
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def replenishment_process(env, inv):
            """补货进程"""
            while True:
                # 每小时检查一次库存
                yield env.timeout(60 * 60)
                
                # 检查每个货位的库存
                for slot_id, quantity in inv.items():
                    slot_info = self.slots[self.slots['slot_id'] == slot_id].iloc[0]
                    sku_id = slot_info['sku']
                    
                    # 计算ROP
                    rop = self.calculate_rop(sku_id)
                    
                    # 如果库存低于ROP，触发补货
                    if quantity < rop:
                        # 计算补货数量
                        replenish_qty = int(rop * self.config['REPLENISH_QTY_FACTOR'] - quantity)
                        
                        # 模拟补货提前期
                        lead_time = max(0, self.rng.normal(
                            self.config['LEAD_TIME_MEAN'],
                            self.config['LEAD_TIME_STD']
                        ))
                        
                        yield env.timeout(lead_time)
                        
                        # 执行补货
                        inv[slot_id] += replenish_qty
                        
                        self.replenishment_log.append({
                            'slot_id': slot_id,
                            'sku_id': sku_id,
                            'trigger_ts': env.now - lead_time,
                            'complete_ts': env.now,
                            'quantity': replenish_qty,
                            'rop': rop,
                            'stock_before': quantity,
                            'stock_after': inv[slot_id]
                        })
                        
                        logger.debug(f"补货完成: Slot {slot_id}, SKU {sku_id}, 数量 {replenish_qty}")
        
        def picker_process(env, pid, job_queue):
            """拣选员进程"""
            while True:
                if job_queue:
                    oid = job_queue.pop(0)
                    order = self.orders[self.orders.order_id == oid].iloc[0]
                    
                    # 构建拣选路径
                    route = []
                    for sku in order.sku_list:
                        # 找到包含该SKU且有库存的货位
                        possible_slots = self.slots[
                            (self.slots.sku == sku) & 
                            (self.slots.slot_id.map(lambda x: inv.get(x, 0) > 0))
                        ]
                        
                        if not possible_slots.empty:
                            # 选择最近的货位
                            slot_id = possible_slots.sample(1, random_state=self.rng).iloc[0]['slot_id']
                            route.append(slot_id)
                        else:
                            logger.warning(f"SKU {sku} 库存不足")
                    
                    # 执行拣选路径
                    current_pos = (0, 0)
                    
                    for slot_id in route:
                        slot = self.slots[self.slots.slot_id == slot_id].iloc[0]
                        target_pos = (int(slot['x']), int(slot['y']))
                        
                        # 计算路径
                        try:
                            path = nx.astar_path(G, current_pos, target_pos, heuristic=heuristic)
                            
                            # 行走
                            for node in path[1:]:
                                yield env.timeout(self.config['WALK_SPEED'])
                                pick_log.append({
                                    'picker_id': pid,
                                    'step_ts': env.now,
                                    'x': node[0],
                                    'y': node[1],
                                    'load': 0,
                                    'action': 'walk'
                                })
                            
                            # 拣选
                            yield env.timeout(self.config['PICK_TIME'])
                            if inv[slot_id] > 0:
                                inv[slot_id] -= 1
                                pick_log.append({
                                    'picker_id': pid,
                                    'step_ts': env.now,
                                    'x': target_pos[0],
                                    'y': target_pos[1],
                                    'load': 1,
                                    'action': 'pick'
                                })
                            
                            current_pos = target_pos
                            
                        except nx.NetworkXNoPath:
                            logger.warning(f"无法找到从 {current_pos} 到 {target_pos} 的路径")
                    
                    # 装车
                    yield env.timeout(self.config['LOAD_TIME'])
                    pick_log.append({
                        'picker_id': pid,
                        'step_ts': env.now,
                        'x': current_pos[0],
                        'y': current_pos[1],
                        'load': 0,
                        'action': 'load'
                    })
                    
                else:
                    yield env.timeout(1)
        
        # 创建任务队列
        job_queues = {f"P{i}": [] for i in range(self.config['N_PICKERS'])}
        for _, assignment in self.assign.iterrows():
            job_queues[assignment.picker_id].append(assignment.order_id)
        
        # 启动拣选员进程
        for pid in job_queues:
            env.process(picker_process(env, pid, job_queues[pid]))
        
        # 启动补货进程
        env.process(replenishment_process(env, inv))
        
        # 运行仿真
        env.run(until=self.config['SIM_TIME'])
        
        # 保存拣选轨迹
        self.picks = pd.DataFrame(pick_log)
        
        # 更新最终库存
        self.slots['quantity'] = self.slots['slot_id'].map(inv)
        
        logger.info("仿真完成")
        
    def save_results(self):
        """保存所有结果"""
        logger.info("保存仿真结果...")
        
        # 订单数据
        self.orders.to_csv("orders.csv", index=False)
        
        # 分配数据
        self.assign.to_csv("assign.csv", index=False)
        
        # 拣选轨迹
        if not self.picks.empty:
            # 只保留必要的列
            picks_output = self.picks[['picker_id', 'step_ts', 'x', 'y', 'load']].copy()
            picks_output.to_csv("picks.csv", index=False)
        
        # 最终库存
        inventory_end = self.slots[['slot_id', 'sku', 'quantity']].copy()
        inventory_end.to_csv("inventory_end.csv", index=False)
        
        # 补货日志
        if self.replenishment_log:
            replenishment_df = pd.DataFrame(self.replenishment_log)
            replenishment_df.to_csv("replenishment_log.csv", index=False)
            logger.info(f"记录了 {len(self.replenishment_log)} 次补货")
        
        logger.info("所有数据已保存")
        
    def run(self):
        """运行完整的仿真流程"""
        start_time = datetime.now()
        logger.info(f"开始仓库仿真 - {start_time}")
        
        # 生成基础数据
        self.generate_slots()
        self.generate_sku_master()
        self.initial_inventory_allocation()
        
        # 生成订单和分配
        self.generate_orders()
        self.nested_logit_assignment()
        
        # 运行仿真
        self.run_simulation()
        
        # 保存结果
        self.save_results()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"仿真完成 - 用时: {duration:.2f}秒")
        
        # 输出统计信息
        self.print_statistics()
        
    def print_statistics(self):
        """打印仿真统计信息"""
        print("\n=== 仿真统计 ===")
        print(f"仓库规模: {self.config['GRID_X']}×{self.config['GRID_Y']}×{self.config['GRID_Z']}")
        print(f"货位总数: {self.n_slots}")
        print(f"SKU总数: {self.config['N_SKU']}")
        print(f"订单总数: {len(self.orders)}")
        print(f"已分配任务: {len(self.assign)}")
        print(f"拣选动作数: {len(self.picks) if self.picks is not None else 0}")
        print(f"补货次数: {len(self.replenishment_log)}")
        
        if self.assign is not None and len(self.assign) > 0:
            assignment_rate = len(self.assign) / len(self.orders) * 100
            print(f"任务分配率: {assignment_rate:.1f}%")
        
        print("\n文件输出:")
        print("- slots.csv: 货位拓扑")
        print("- sku_master.csv: SKU主数据")
        print("- inventory_day0.csv: 初始库存")
        print("- assigned_slots.csv: 货位SKU分配")
        print("- orders.csv: 订单数据")
        print("- assign.csv: 任务分配")
        print("- picks.csv: 拣选轨迹")
        print("- inventory_end.csv: 最终库存")
        print("- replenishment_log.csv: 补货日志")


def main():
    """主函数"""
    # 可以自定义配置
    custom_config = {
        # 'GRID_X': 40,
        # 'GRID_Y': 25,
        # 'N_PICKERS': 8,
        # 'SIM_TIME': 4*60*60  # 4小时仿真
    }
    
    # 创建并运行仿真
    sim = WarehouseSimulation(config=custom_config)
    sim.run()


if __name__ == "__main__":
    main()