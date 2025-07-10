# 项目结构设置

```python
LogitMARL
├── warehouse_env
    └── env
        └── warehouse_env.py #是存储环境以及任何辅助函数的地方。
    └── warehouse_env_v0.py #是一个导入环境的文件 - 我们使用文件名进行环境版本控制
├── README.md
└── requirements.txt #是一个用于跟踪您环境依赖的文件。至少应该包含 pettingzoo 。请通过 == 对所有依赖进行版本控制。
```
## 假设条件-状态空间
我们先假设一个 10* 10* 3 的仓库环境。

一个集装箱入库之后，会拆包出体积固定，质量有上限的物体。

为了模拟，我们生成了一些模拟的 SKU 和任务

每个 SKU 有下列参数：
1. SKU 编号 `sku_id`
2. 出库频率 `f_out`
3. 入库频率 `f_in`
4. SKU 大小 `sku_volumn` 以 mm 为单位
5. SKU 重量 `sku_weight` 以 g 为单位
6. SKU 种类 `sku genre` 例如：普通、易碎

每个 item 有下列参数：
1. 继承所有 SKU 的参数
2. item 位置 `item_position:[x,y,z]`
3. item 可达度 `item_accessibility`

每个库位有下列参数：
1. 库位位置 `storage_position:[x,y,z]`
2. 库位剩余重量 `storage_remain_weight`
3. 库位剩余体积 `storgae_remain_volumn`
4. 库位可达度 `storage_accessibility`

每个任务有下列参数
1. 任务内容：item 编号、数量的数组 `task_content:[item_id,item_num]`
2. 任务要求时间 `task_req_time`
3. 任务已进行时间 `task_time`
4. 任务奖励（完成可以获得奖励）`task_reward`
5. 任务惩罚（未完成将获得惩罚）`task_penalty`
6. 任务类型（出库或入库）`task_type`

每个 picker 有下列参数：
1. picker 位置 `picker_position:[x,y,z]`
2. picker 状态（是否空闲） `picker_status`
3. picker 剩余体积 `picker_remain_volumn`
4. picker 剩余重量 `picker_remain_weight`


## 管理者
当有智能体空闲且有任务可用时，将任务分配给picker

观察空间：
T 任务（任务 ID，任务类型，位置信息，截止时间，基础时长（由理想资源完成 所需的时间））
P 资源列表（拣选员类型，拣选员位置，拣选员状态，当前工作负荷）

策略：
对每一个可能的分配对计算效用值 U(T,P)
U(T, P) = [基础效用项，如 w_p * 优先级(T) - w_w * 负荷(P) - w_d * 距离(P, T) + ...] + 惩罚项(任务类型(T), 资源类型(P))
权重 (w_p, w_w, w_d, ...) 由 RL 算法学习得到

##这里使用了这个惩罚项来体现嵌套 Logit 的思想