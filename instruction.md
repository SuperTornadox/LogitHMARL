本项目是一套 **Python-驱动的仓库拣选仿真与数据合成工具链**：它在 180 行左右的脚本内完成货位网格、SKU 主数据、泊松到达订单、Nested-Logit 经理分配、SimPy 离散事件驱动的拣选轨迹以及 ROP 补货机制的端到端生成，输出六张结构化 CSV 供后续 HRL / 机器学习实验直接使用。脚本依赖纯 Python 开源库（SimPy、NetworkX、NumPy、scikit-learn、pandas），无外部服务即可在几秒内产出一整天的中型仓库历史日志。下面按功能模块详细说明，方便 Cursor 等 IDE 在阅读代码时快速定位逻辑、扩展功能并理解背后的运营与建模假设。  

## 1 项目目标与定位  
- **研究级合成数据**：为多智能体强化学习（HRL）或 Nested-Logit 调度算法提供真实感强、可控的训练 / 验证数据集，而无需昂贵的传感器采集或商业 WMS 日志许可。  
- **端到端链路**：覆盖库存初始化→订单到达→经理分配→拣选执行→补货回补的完整仓库生命周期，且各模块可独立开关。  
- **零外部依赖**：仅用标准科学计算栈即可运行；SimPy 负责离散事件调度，NetworkX 负责栅格 A* 路径，scikit-learn 负责空间簇划分。 ([Overview — SimPy 4.1.2.dev8+g81c7218 documentation](https://simpy.readthedocs.io/?utm_source=chatgpt.com), [Tutorial — NetworkX 3.4.2 documentation](https://networkx.org/documentation/stable/tutorial.html?utm_source=chatgpt.com), [KMeans — scikit-learn 1.6.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?utm_source=chatgpt.com))  

## 2 核心技术与设计依据  

### 2.1 订单与需求  
- **泊松到达**：订单到达间隔采样自指数分布 λ=120 单/时，符合经典顾客到达/呼叫中心负荷建模假设。 ([Basic Concepts of the Poisson Process - Probability Course](https://www.probabilitycourse.com/chapter11/11_1_2_basic_concepts_of_the_poisson_process.php?utm_source=chatgpt.com))  
- **Zipf-长尾 SKU 频率**：订单行内 SKU 根据 Zipf(a=1.2) 抽样，模拟电子商务常见“20/80” 需求分布。 ([numpy.random.zipf — NumPy v2.2 Manual](https://numpy.org/doc/2.2/reference/random/generated/numpy.random.zipf.html?utm_source=chatgpt.com))  

### 2.2 库存初始化与 Slotting  
- **ABC 分类** 定义初始件数：A/B/C SKU 以 20 % / 30 % / 50 % 占比划分价值，并分配 400/120/40 件起量，符合经典库存分析。 ([ABC Inventory Analysis & Management | NetSuite](https://www.netsuite.com/portal/resource/articles/inventory-management/abc-inventory-analysis.shtml?utm_source=chatgpt.com))  
- **高周转前置**：将 A 类 SKU 优先 Slot 到网格前 1/3 行，参考 Velocity-Based 仓位策略以缩短行走距离。 ([SKU velocity and efficient warehouse slotting - Interlake Mecalux](https://www.interlakemecalux.com/blog/sku-velocity-slotting?utm_source=chatgpt.com), [Optimal SKU Slotting Methods for Warehouse Efficiency](https://opsdesign.com/optimal-sku-slotting/?utm_source=chatgpt.com))  

### 2.3 经理分配（Nested-Logit 简化采样）  
- 脚本为演示使用经验巢概率；替换为 Statsmodels 或 PyLogit 估计的 θ、φ 即可得到符合离散选择理论的分层概率。 ([Examples - Statsmodels](https://www.statsmodels.org/v0.12.2/examples/index.html?utm_source=chatgpt.com), [Deep Reinforcement Learning for Dynamic Order Picking in ... - arXiv](https://arxiv.org/abs/2408.01656v1/?utm_source=chatgpt.com))  
- 真实 Nested-Logit 能克服 MNL 的 IIA 假设，在“同一区域、同设备”任务相关场景更贴合实务调度。 ([Examples - Statsmodels](https://www.statsmodels.org/v0.12.2/examples/index.html?utm_source=chatgpt.com))  

### 2.4 拣选执行  
- **NetworkX A\***：在 32 × 20 栅格上预计算最短路，节点表示可行走地面，成本按曼哈顿距离启发式。 ([Tutorial — NetworkX 3.4.2 documentation](https://networkx.org/documentation/stable/tutorial.html?utm_source=chatgpt.com))  
- **SimPy 进程**：每位拣选员建为独立 Process；行走、拣选、装车均 `yield env.timeout()`，可插入堵塞或设备等待。 ([Overview — SimPy 4.1.2.dev8+g81c7218 documentation](https://simpy.readthedocs.io/?utm_source=chatgpt.com))  

### 2.5 补货机制  
- 采用 **ROP = 需求×提前期 + 安全库存** 公式；安全天数默认 0.5 天，提前期服从均值 2 h 正态分布。 ([How to Calculate Reorder Points with the ROP Formula - ShipBob](https://www.shipbob.com/blog/reorder-point-formula/?utm_source=chatgpt.com), [Safety Stock Formula & Calculation: The 6 best methods](https://abcsupplychain.com/safety-stock-formula-calculation/?utm_source=chatgpt.com))  

## 3 输出文件说明  

| 文件 | 主要字段 | 用途 |
|------|---------|------|
| `slots.csv` | slot_id,x,y,max_capacity,level | 货位拓扑 |
| `sku_master.csv` | sku_id,class,init_qty | SKU 基础表 |
| `inventory_day0.csv` | slot_id,sku,quantity | T0 库存快照 |
| `orders.csv` | order_id,create_ts,sku_list,eq_flag,sla_sec,nest_id | 需求侧输入 |
| `assign.csv` | order_id,picker_id,assign_ts | 经理决策输出 |
| `picks.csv` | picker_id,step_ts,x,y,load | 拣选员轨迹 |
| `inventory_end.csv` | slot_id,sku,quantity | 仿真结束库存 |

所有表均为 UTF-8 无索引 CSV，可直接用 pandas/SQL 导入分析。  

## 4 运行与复现步骤  
1. `pip install pandas numpy simpy networkx scikit-learn tqdm` 安装依赖。  
2. 运行脚本生成完整数据；默认 8 小时仿真在笔记本 <5 秒完成。  
3. 若需接入 RLlib：  
   * 替换 **assign** 逻辑为 Manager-Policy 调用；  
   * 将 `picker_proc` 的离散动作暴露为 PettingZoo API。  

## 5 可扩展点  

| 方向 | 简介 |
|------|------|
| **AGV/混合搬运** | 给一部分订单设置“重型设备”需求，添加 AGV Agent 并在冲突调度中体现优先权。 |
| **多楼层 / 电梯** | 将 NetworkX 图拓展为多层立体节点，边权加入等待时间。 |
| **批次波次拣选** | 在 Manager 侧加入订单聚类（如 k-means 货位距离），参考 2024 年 DRL 动态拣选研究。 ([Deep Reinforcement Learning for Dynamic Order Picking in ... - arXiv](https://arxiv.org/abs/2408.01656v1/?utm_source=chatgpt.com)) |
| **可视化 Dash** | 用 Plotly 显示拣选热力图、库存周转 KPI。 |

## 6 理论与实践参考  

1. SimPy 官方文档 – 离散事件框架概览 ([Overview — SimPy 4.1.2.dev8+g81c7218 documentation](https://simpy.readthedocs.io/?utm_source=chatgpt.com))  
2. NetworkX Tutorial – 图与 A* 搜索示例 ([Tutorial — NetworkX 3.4.2 documentation](https://networkx.org/documentation/stable/tutorial.html?utm_source=chatgpt.com))  
3. NumPy `random.zipf` – Zipf 抽样 API ([numpy.random.zipf — NumPy v2.2 Manual](https://numpy.org/doc/2.2/reference/random/generated/numpy.random.zipf.html?utm_source=chatgpt.com))  
4. ABC 分析原理与业务价值 ([ABC Inventory Analysis & Management | NetSuite](https://www.netsuite.com/portal/resource/articles/inventory-management/abc-inventory-analysis.shtml?utm_source=chatgpt.com))  
5. Poisson 过程基础与到达模型 ([Basic Concepts of the Poisson Process - Probability Course](https://www.probabilitycourse.com/chapter11/11_1_2_basic_concepts_of_the_poisson_process.php?utm_source=chatgpt.com))  
6. Statsmodels Example Index – 离散选择 / Nested-Logit 实例 ([Examples - Statsmodels](https://www.statsmodels.org/v0.12.2/examples/index.html?utm_source=chatgpt.com))  
7. scikit-learn KMeans 文档 ([KMeans — scikit-learn 1.6.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?utm_source=chatgpt.com))  
8. Reorder Point 计算公式与解释 ([How to Calculate Reorder Points with the ROP Formula - ShipBob](https://www.shipbob.com/blog/reorder-point-formula/?utm_source=chatgpt.com))  
9. SKU Velocity-Based Slotting 文章 ([SKU velocity and efficient warehouse slotting - Interlake Mecalux](https://www.interlakemecalux.com/blog/sku-velocity-slotting?utm_source=chatgpt.com))  
10. 2024 年 DRL 动态拣选论文 ([Deep Reinforcement Learning for Dynamic Order Picking in ... - arXiv](https://arxiv.org/abs/2408.01656v1/?utm_source=chatgpt.com))  
11. Safety Stock / ROP 深度分析 ([Safety Stock Formula & Calculation: The 6 best methods](https://abcsupplychain.com/safety-stock-formula-calculation/?utm_source=chatgpt.com))  
12. 现代仓位优化综述（OpsDesign） ([Optimal SKU Slotting Methods for Warehouse Efficiency](https://opsdesign.com/optimal-sku-slotting/?utm_source=chatgpt.com))  

---

依托以上设计与参考，**本项目既能作为教学级示例，快速让研究者复现“库存-订单-任务-执行-补货”全链路，又可作为工业级算法验证沙盒，在保证可解释性的同时为多智能体学习或优化模型提供高质量、可扩展的数据基础。**