"""
快速测试订单生成优化效果
测试 Zone 概率查找表 + 二分查找优化
"""

import time
import numpy as np
from src.env.order_generation import NonHomogeneousPoissonOrderGenerator

# 设置随机种子
np.random.seed(42)

# 创建测试配置（模拟 VARIANT-1 的订单配置）
test_config = {
    'base_rate': 2000,  # 2000订单/小时
    'peak_hours': [(0, 6), (12, 18)],
    'peak_multiplier': 1.8,
    'off_peak_multiplier': 0.6,
    'urgent_order_prob': 0.25,
    'bulk_order_prob': 0.15,
    'simulation_hours': 10.0,  # 测试10小时（相当于60k步）
    'n_skus': 1000,
    'use_pregeneration': True,
    'pattern_period_hours': 24,
    'scale_pattern_to_simulation': True,

    # VARIANT-1 配置
    'deadline_ranges': {
        'urgent': (0.3, 0.6),
        'regular': (0.8, 1.5),
        'bulk': (1.5, 3.0),
    },
    'deadline_scale': 1.0,
    'bursty_prob': 0.15,
    'burst_size': (10, 20),
    'burst_zone_correlation': 0.75,

    # Zone peak schedule（关键优化目标）
    'zone_peak_schedule': [
        (0, 6, [0.5, 0.3, 0.1, 0.1]),   # 早期：Zone 0 主导
        (6, 12, [0.2, 0.5, 0.2, 0.1]),  # 上午：Zone 1 主导
        (12, 18, [0.1, 0.2, 0.5, 0.2]), # 下午：Zone 2 主导
        (18, 24, [0.1, 0.1, 0.3, 0.5]), # 晚间：Zone 3 主导
    ],
}

print("=" * 70)
print("订单生成优化测试")
print("=" * 70)

# 测试1: 订单预生成速度
print("\n[测试1] 订单预生成速度")
start_time = time.time()
generator = NonHomogeneousPoissonOrderGenerator(test_config)
gen_time = time.time() - start_time

n_orders = len(generator.pregenerated_orders)
print(f"✓ 预生成 {n_orders:,} 个订单")
print(f"✓ 用时: {gen_time:.2f} 秒")
print(f"✓ 速度: {n_orders/gen_time:.0f} 订单/秒")

# 测试2: Zone概率查找表
print("\n[测试2] Zone概率查找表")
if generator.zone_lookup_table is not None:
    print(f"✓ 查找表已创建: shape={generator.zone_lookup_table.shape}")
    print(f"✓ 分辨率: {generator.zone_lookup_bins} bins")

    # 测试几个时间点
    test_times = [0.5, 3.0, 9.0, 15.0, 21.0]
    print("\n  时间点测试:")
    for t in test_times:
        probs = generator.get_zone_probs(t)
        print(f"    t={t:5.1f}h -> Zone概率: {probs}")
else:
    print("✗ 查找表未创建")

# 测试3: 订单窗口查询速度（二分查找 vs 线性扫描）
print("\n[测试3] 订单窗口查询速度")

# 模拟多次查询（类似训练过程）
n_queries = 1000
window_size = 0.5  # 0.5小时窗口

# 测试二分查找（当前实现）
start_time = time.time()
for i in range(n_queries):
    start_t = np.random.uniform(0, test_config['simulation_hours'] - window_size)
    end_t = start_t + window_size
    orders = generator.get_orders_in_window(start_t, end_t)
binary_time = time.time() - start_time

print(f"✓ 二分查找: {n_queries} 次查询用时 {binary_time:.3f} 秒")
print(f"✓ 平均查询时间: {binary_time/n_queries*1000:.2f} ms")

# 测试4: 缓存的 arrival_times
print("\n[测试4] Arrival Times 缓存")
if hasattr(generator, '_cached_arrival_times'):
    cached_len = len(generator._cached_arrival_times)
    orders_len = len(generator.pregenerated_orders)
    print(f"✓ 缓存已创建: {cached_len:,} 个时间戳")
    print(f"✓ 与订单数量一致: {cached_len == orders_len}")

    # 验证排序
    is_sorted = all(generator._cached_arrival_times[i] <= generator._cached_arrival_times[i+1]
                    for i in range(len(generator._cached_arrival_times)-1))
    print(f"✓ 已正确排序: {is_sorted}")
else:
    print("✗ 缓存未创建")

# 测试5: 订单统计
print("\n[测试5] 订单统计")
stats = generator.get_statistics()
print(f"✓ 总订单数: {stats['total_orders']:,}")
print(f"✓ 平均到达率: {stats['avg_arrival_rate']:.1f} 订单/小时")
print(f"✓ 订单类型分布: {stats['order_types']}")

# 测试6: 窗口查询正确性验证
print("\n[测试6] 窗口查询正确性验证")
test_windows = [(0, 1), (5, 6), (9, 10)]
for start_t, end_t in test_windows:
    orders = generator.get_orders_in_window(start_t, end_t)
    # 验证所有订单都在窗口内
    valid = all(start_t <= o.arrival_time < end_t for o in orders)
    print(f"  窗口 [{start_t}, {end_t}): {len(orders)} 个订单, 正确性: {valid}")

print("\n" + "=" * 70)
print("✅ 所有测试完成!")
print("=" * 70)

# 估算全规模性能
print("\n[性能估算]")
full_sim_hours = 166.67  # 200k步
n_envs = 16
scale_factor = full_sim_hours / test_config['simulation_hours']
est_orders_per_env = int(n_orders * scale_factor)
est_total_orders = est_orders_per_env * n_envs
est_gen_time = gen_time * scale_factor * n_envs

print(f"全规模配置 (200k步, 16环境):")
print(f"  预计每环境订单数: {est_orders_per_env:,}")
print(f"  预计总订单数: {est_total_orders:,}")
print(f"  预计初始化时间: {est_gen_time:.1f} 秒 ({est_gen_time/60:.1f} 分钟)")
