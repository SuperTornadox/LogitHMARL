"""
Synthetic warehouse day— order, inventory & pick trace generator
Author: ChatGPT (o3) — 2025-04-28
"""

import pandas as pd, numpy as np, simpy, networkx as nx
from sklearn.cluster import KMeans
from tqdm import tqdm

# ---------- 0. 参数 ----------
GRID_X, GRID_Y, GRID_Z       = 32, 20, 3            # 仓库网格尺寸
N_SLOTS              = GRID_X * GRID_Y * GRID_Z   # one slot per walkable cell
N_SKU                = 1000
SIM_TIME             = 8*60*60           # 8 h = 28 800 s
ORDER_LAMBDA         = 2/60              # 2 单 / 分钟  ➜ 120 单/时  (Poisson) :contentReference[oaicite:10]{index=10}
ZIPF_A               = 1.2               # Zipf 分布参数 (长尾) :contentReference[oaicite:11]{index=11}
ABC_SPLIT            = (.2, .3, .5)      # A/B/C SKU比例 :contentReference[oaicite:12]{index=12}
ABC_QTY              = (np.random.randint(200,400), np.random.randint(100,200), np.random.randint(50,100))    # 初始件数 A/B/C
FORKLIFT_RATIO       = 0.15              # 需要叉车的 slot 比例
N_PICKERS            = 6
WALK_SPEED           = 1                 # 1 格 / s
PICK_TIME            = 5                 # 5 s / 件
LOAD_TIME            = 8                 # 装车
ROP_DAYS             = 0.5               # 安全库存天数 (简化)
LEAD_TIME_MEAN       = 2*60*60           # 2 h
RNG                  = np.random.default_rng(42) #创建受控随机流，保证结果可复现；42 只是惯用示例种子。

# ---------- 1. 生成货位 ----------
x, y, z = np.meshgrid(np.arange(GRID_X), np.arange(GRID_Y), np.arange(GRID_Z))
slots = (pd.DataFrame({'slot_id': np.arange(N_SLOTS),
                       'x': x.flatten(),
                       'y': y.flatten(),
                       'z': z.flatten(),
                       'max_capacity': 999}))
slots.to_csv("slots.csv", index=False)


# ---------- 2. 生成 SKU 主数据 ----------
sku_df = pd.DataFrame({'sku_id': np.arange(N_SKU)})
sku_df['class'] = pd.cut(sku_df['sku_id'],
                         bins=[-1, int(N_SKU*ABC_SPLIT[0]),
                               int(N_SKU*(ABC_SPLIT[0]+ABC_SPLIT[1])), N_SKU],
                         labels=['A','B','C'])
sku_df['init_qty'] = sku_df['class'].map(
        {'A': ABC_QTY[0], 'B': ABC_QTY[1], 'C': ABC_QTY[2]})
sku_df['height']=np.clip(RNG.triangular(left=5,mode=20,right=120,size=N_SKU),5,120).astype(int)
sku_df['length']=np.clip(RNG.triangular(left=5,mode=20,right=120,size=N_SKU),5,120).astype(int)
sku_df['width']=np.clip(RNG.triangular(left=5,mode=20,right=120,size=N_SKU),5,120).astype(int) #长宽高在 5～120 cm 之间
sku_df.to_csv("sku_master.csv", index=False)

# ---------- 3. 初始库存分配 & slotting ----------
slots['sku']      = RNG.choice(sku_df['sku_id'], size=N_SLOTS)
slots['quantity'] = slots['sku'].map(sku_df.set_index('sku_id')['init_qty'])
# slotting: 把 A 类移到前 1/3 排 (y 轴)&#8203;:contentReference[oaicite:13]{index=13}
high_demand_ids = sku_df.query("`class`=='A'")['sku_id']
front_rows      = slots.query("y < @GRID_H/3").index
slots.loc[front_rows, 'sku'] = RNG.choice(high_demand_ids, size=len(front_rows))
slots.to_csv("inventory_day0.csv", index=False)

# ---------- 4. 订单生成 ----------
def gen_order_id(i): return f"O{i:05d}"
orders = []
t = 0.0
order_idx = 0
while t < SIM_TIME:
    # Poisson inter-arrival
    t += RNG.exponential(1/ORDER_LAMBDA)
    # 抽取行数（订单行） 1–5
    n_lines = RNG.integers(1, 6)
    # SKU 按 Zipf 抽样
    skus = RNG.zipf(a=ZIPF_A, size=n_lines) % N_SKU
    # SLA tag
    sla_sec = RNG.choice([30*60, 60*60, 2*60*60], p=[0.3,0.5,0.2])
    # 装备需求 flag
    eq_flag = RNG.random() < FORKLIFT_RATIO
    orders.append({'order_id': gen_order_id(order_idx),
                   'create_ts': t,
                   'sku_list': list(map(int, skus)),
                   'eq_flag': int(eq_flag),
                   'sla_sec': int(sla_sec)})
    order_idx += 1
orders = pd.DataFrame(orders)

# ---------- 5. K-Means 空间巢 + Nested-Logit 经理分配 ----------
coords = slots[['x','y']].values
k = 8
clusters = KMeans(n_clusters=k, random_state=0).fit(coords)          #&#8203;:contentReference[oaicite:14]{index=14}
slots['cluster'] = clusters.labels_
orders['nest_id'] = (orders['eq_flag'].astype(str) + "_" +
                     RNG.choice(k, size=len(orders)).astype(str) + "_" +
                     pd.cut(orders['sla_sec'],
                            bins=[0,1800,3600,SIM_TIME],
                            labels=['F','N','L']).astype(str))

# 简化 NL：巢概率 ~ 经验频率；巢内随机
nest_probs = orders['nest_id'].value_counts(normalize=True).to_dict()
assign_rows = []
for _, row in orders.iterrows():
    if RNG.random() < nest_probs[row['nest_id']]:          # 经理接受该巢
        picker = RNG.integers(0, N_PICKERS)                # 轮询可改HRL
        assign_rows.append({'order_id': row['order_id'],
                            'picker_id': f"P{picker}",
                            'assign_ts': row['create_ts'],
                            'picked_flag': 0})
assign = pd.DataFrame(assign_rows)

# ---------- 6. 拣选仿真 ----------
G = nx.grid_2d_graph(GRID_X, GRID_Y)                       #&#8203;:contentReference[oaicite:15]{index=15}
env = simpy.Environment()
inv      = slots.set_index('slot_id')['quantity'].to_dict()
pick_log = []
def heuristic(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def picker_proc(env, pid, job_q):
    while True:
        if job_q:
            oid = job_q.pop(0)
            order = orders.loc[orders.order_id==oid].iloc[0]
            # 找到各 SKU 的 slot
            route = []
            for sku in order.sku_list:
                # 筛选出包含当前 sku 的货位
                possible_slots = slots.loc[slots.sku == sku]
                # 检查是否有找到对应的货位
                if not possible_slots.empty:
                    # 如果找到了，从中随机抽取一个
                    slot_id = possible_slots.sample(1, random_state=RNG).iloc[0]['slot_id']
                    route.append(slot_id)
                else:
                    # 如果没找到，需要决定如何处理
                    # 选项 1: 跳过这个 SKU
                    print(f"警告：SKU '{sku}' 在 'slots' 中没有找到对应的货位，已跳过。")
                    # 选项 2: 抛出更明确的错误
                    # raise ValueError(f"错误：SKU '{sku}' 在 'slots' 中没有找到对应的货位。请检查数据一致性。")
                    # 选项 3: 执行其他逻辑...
                    pass # 当前示例选择跳过
            # 简单最近邻排序（可替换TSP）
            cur = (0,0)
            for sid in route:
                target = (int(slots.loc[sid,'x']), int(slots.loc[sid,'y']))
                path = nx.astar_path(G, cur, target, heuristic=heuristic)
                for node in path[1:]:
                    yield env.timeout(WALK_SPEED)
                    pick_log.append((pid, env.now, node[0], node[1], 0))
                # pick action
                yield env.timeout(PICK_TIME)
                inv[sid] -= 1
                cur = target
            # 装车
            yield env.timeout(LOAD_TIME)
        else:
            yield env.timeout(1)

job_queues = {f"P{i}":[] for i in range(N_PICKERS)}
for _, a in assign.iterrows():
    job_queues[a.picker_id].append(a.order_id)
for pid in job_queues:
    env.process(picker_proc(env, pid, job_queues[pid]))
env.run(until=SIM_TIME)

picks = pd.DataFrame(pick_log, columns=['picker_id','step_ts','x','y','load'])
slots['quantity'] = slots.index.map(inv.get)
slots.to_csv("inventory_end.csv", index=False)
orders.to_csv("orders.csv",   index=False)
assign.to_csv("assign.csv",   index=False)
picks.to_csv("picks.csv",     index=False)
print("Synthetic data generated ✓")