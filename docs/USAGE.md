# 仓库拣选仿真系统使用指南

## 概述

本项目是一个基于Python的仓库拣选仿真与数据合成工具链，用于生成多智能体强化学习（HRL）或Nested-Logit调度算法的训练数据。系统覆盖了从库存初始化到订单拣选、补货的完整仓库运营周期。

## 安装环境

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：
- pandas: 数据处理
- numpy: 数值计算
- simpy: 离散事件仿真
- networkx: 图算法和A*路径搜索
- scikit-learn: K-means聚类
- tqdm: 进度条
- pettingzoo: 多智能体强化学习环境
- matplotlib: 可视化

### 2. 验证安装

```bash
python3 -c "import simpy, networkx, sklearn; print('环境准备就绪')"
```

## 快速开始

### 运行默认仿真

```bash
python3 scripts/warehouse_system.py run-sim
```

默认配置将：
- 创建32×20×3的仓库网格（1920个货位）
- 生成1000个SKU
- 运行8小时仿真
- 使用6个拣选员
- 输出8-9个CSV文件

### 自定义运行示例

通过命令行参数覆盖默认配置：

```bash
python3 scripts/warehouse_system.py run-sim \
  --sim-time 14400 \
  --n-pickers 8 \
  --grid-x 40 --grid-y 25 --grid-z 3 \
  --order-lambda 0.05 \
  --seed 123
```

## 配置参数

可通过命令行参数传入配置（无需修改代码）：

- 仿真时长（秒）：`--sim-time`
- 拣选员数量：`--n-pickers`
- 仓库尺寸：`--grid-x --grid-y --grid-z`
- 订单到达率（单/秒）：`--order-lambda`
- 随机种子：`--seed`

示例见上节“自定义运行示例”。

## 输出文件说明

仿真运行后会生成以下CSV文件：

| 文件名 | 说明 | 主要字段 |
|--------|------|----------|
| `slots.csv` | 货位拓扑信息 | slot_id, x, y, z, max_capacity, level |
| `sku_master.csv` | SKU主数据 | sku_id, class, init_qty, dimensions, weight, daily_demand |
| `inventory_day0.csv` | 初始库存快照 | slot_id, sku, quantity |
| `assigned_slots.csv` | 货位-SKU分配 | slot_id, x, y, z, sku |
| `orders.csv` | 订单数据 | order_id, create_ts, sku_list, eq_flag, sla_sec, priority |
| `assign.csv` | 任务分配记录 | order_id, picker_id, assign_ts |
| `picks.csv` | 拣选轨迹 | picker_id, step_ts, x, y, load |
| `inventory_end.csv` | 期末库存 | slot_id, sku, quantity |
| `replenishment_log.csv` | 补货日志（可选） | slot_id, sku_id, quantity, timestamps |

## 数据分析示例

### 1. 查看订单分配率

```python
import pandas as pd

orders = pd.read_csv('orders.csv')
assign = pd.read_csv('assign.csv')

assignment_rate = len(assign) / len(orders) * 100
print(f"订单分配率: {assignment_rate:.1f}%")
```

### 2. 分析拣选员工作负荷

```python
picks = pd.read_csv('picks.csv')
workload = picks.groupby('picker_id').size()
print("各拣选员动作数:")
print(workload)
```

### 3. 库存变化分析

```python
inv_start = pd.read_csv('inventory_day0.csv')
inv_end = pd.read_csv('inventory_end.csv')

# 合并数据
inv_compare = inv_start.merge(
    inv_end, 
    on=['slot_id', 'sku'], 
    suffixes=('_start', '_end')
)

# 计算消耗
inv_compare['consumed'] = inv_compare['quantity_start'] - inv_compare['quantity_end']
total_consumed = inv_compare['consumed'].sum()
print(f"总消耗量: {total_consumed}")
```

### 4. 可视化拣选热力图

```python
import matplotlib.pyplot as plt
import numpy as np

picks = pd.read_csv('picks.csv')

# 创建热力图
grid_x, grid_y = 32, 20  # 根据配置调整
heatmap = np.zeros((grid_y, grid_x))

for _, pick in picks.iterrows():
    x, y = int(pick['x']), int(pick['y'])
    if 0 <= x < grid_x and 0 <= y < grid_y:
        heatmap[y, x] += 1

plt.figure(figsize=(12, 8))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.colorbar(label='访问次数')
plt.title('仓库拣选热力图')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.show()
```

## 与强化学习集成

### 1. 环境封装

```python
from warehouse.warehouse_simulation import WarehouseSimulation

class WarehouseEnv:
    def __init__(self, config):
        self.sim = WarehouseSimulation(config)
        self.reset()
        
    def reset(self):
        # 重置环境
        self.sim.generate_slots()
        self.sim.generate_sku_master()
        self.sim.initial_inventory_allocation()
        return self.get_state()
        
    def step(self, action):
        # 执行动作，返回新状态、奖励等
        pass
        
    def get_state(self):
        # 返回当前状态
        pass
```

### 2. 多智能体设置

使用PettingZoo框架：

```python
from pettingzoo import ParallelEnv

class WarehouseMAEnv(ParallelEnv):
    def __init__(self, n_pickers=6):
        self.n_pickers = n_pickers
        self.agents = [f"picker_{i}" for i in range(n_pickers)]
        # 初始化仿真环境
        
    def step(self, actions):
        # 并行执行所有智能体动作
        pass
```

## 常见问题

### Q1: 任务分配率很低怎么办？

A: 可以调整以下参数：
- 增加`N_PICKERS`（拣选员数量）
- 调整Nested-Logit的巢概率计算
- 检查订单生成率是否过高

### Q2: 补货没有触发？

A: 检查：
- 仿真时长是否足够（补货检查每小时一次）
- ROP参数设置是否合理
- 初始库存是否过高

### Q3: 运行速度慢？

A: 优化建议：
- 减少仿真时长
- 减少网格大小
- 使用更高效的路径算法

## 扩展功能

### 1. 添加AGV支持

修改`picker_process`函数，添加AGV类型的处理逻辑。

### 2. 多楼层电梯

扩展NetworkX图为3D，添加电梯节点和等待时间。

### 3. 批次拣选

在任务分配前添加订单聚类步骤。

### 4. 实时监控

集成Dash或Streamlit创建实时仪表板。

## 参考资料

- [SimPy文档](https://simpy.readthedocs.io/)
- [NetworkX教程](https://networkx.org/documentation/stable/tutorial.html)
- [PettingZoo文档](https://pettingzoo.farama.org/)

## 联系方式

如有问题，请查看项目的instruction.md文件或提交Issue。