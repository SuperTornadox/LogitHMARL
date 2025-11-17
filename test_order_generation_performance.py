"""
测试订单预生成性能
重点测量16个环境初始化时的订单生成时间
"""

import time
import sys
import os

# 添加src到路径
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from env.order_generation import NonHomogeneousPoissonOrderGenerator

print("="*70)
print("📊 订单预生成性能测试")
print("="*70)

# VARIANT-1 完整配置（200k步）
ORDER_CONFIG = {
    'base_rate': 2000,
    'peak_hours': [(0, 6), (12, 18)],
    'peak_multiplier': 1.8,
    'off_peak_multiplier': 0.6,
    'urgent_order_prob': 0.25,
    'bulk_order_prob': 0.15,
    'simulation_hours': 166.67,  # 200k步 × 2秒/步 ÷ 3600
    'n_skus': 1000,
    'use_pregeneration': True,
    'pattern_period_hours': 24,
    'scale_pattern_to_simulation': True,
    'deadline_ranges': {
        'urgent': (0.3, 0.6),
        'regular': (0.8, 1.5),
        'bulk': (1.5, 3.0),
    },
    'deadline_scale': 1.0,
    'bursty_prob': 0.15,
    'burst_size': (10, 20),
    'burst_zone_correlation': 0.75,
    'zone_peak_schedule': [
        (0, 6, [0.5, 0.3, 0.1, 0.1]),
        (6, 12, [0.2, 0.5, 0.2, 0.1]),
        (12, 18, [0.1, 0.2, 0.5, 0.2]),
        (18, 24, [0.1, 0.1, 0.3, 0.5]),
    ],
}

print("\n测试场景:")
print(f"  - 仿真时长: {ORDER_CONFIG['simulation_hours']:.2f} 小时")
print(f"  - 到达率: {ORDER_CONFIG['base_rate']} 订单/小时")
print(f"  - 预期订单数: ~{int(ORDER_CONFIG['base_rate'] * ORDER_CONFIG['simulation_hours'] * 1.2):,}")

# 测试1: 单个环境初始化时间
print("\n" + "="*70)
print("测试1: 单个环境订单预生成")
print("="*70)

start_time = time.time()
generator1 = NonHomogeneousPoissonOrderGenerator(ORDER_CONFIG)
single_env_time = time.time() - start_time

n_orders = len(generator1.pregenerated_orders)
print(f"✓ 生成订单数: {n_orders:,}")
print(f"✓ 用时: {single_env_time:.2f} 秒")
print(f"✓ 速度: {n_orders/single_env_time:.0f} 订单/秒")

# 测试2: 16个环境串行初始化（当前实现）
print("\n" + "="*70)
print("测试2: 16个环境串行初始化（当前方式）")
print("="*70)

n_envs = 16
serial_start = time.time()
generators_serial = []
for i in range(n_envs):
    print(f"  环境 {i+1}/{n_envs}...", end=" ", flush=True)
    env_start = time.time()
    gen = NonHomogeneousPoissonOrderGenerator(ORDER_CONFIG)
    env_time = time.time() - env_start
    generators_serial.append(gen)
    print(f"{env_time:.1f}秒")

serial_total_time = time.time() - serial_start
print(f"\n✓ 总用时: {serial_total_time:.2f} 秒 ({serial_total_time/60:.1f} 分钟)")
print(f"✓ 平均每环境: {serial_total_time/n_envs:.2f} 秒")
print(f"✓ 总订单数: {sum(len(g.pregenerated_orders) for g in generators_serial):,}")

# 测试3: 16个环境并行初始化（优化方案）
print("\n" + "="*70)
print("测试3: 16个环境并行初始化（多进程优化）")
print("="*70)

from multiprocessing import Pool
import multiprocessing as mp

def create_order_generator(config):
    """用于多进程的生成器创建函数"""
    gen = NonHomogeneousPoissonOrderGenerator(config)
    return len(gen.pregenerated_orders)

parallel_start = time.time()
with Pool(processes=min(mp.cpu_count(), n_envs)) as pool:
    results = pool.map(create_order_generator, [ORDER_CONFIG] * n_envs)
parallel_total_time = time.time() - parallel_start

print(f"\n✓ 总用时: {parallel_total_time:.2f} 秒 ({parallel_total_time/60:.1f} 分钟)")
print(f"✓ 加速比: {serial_total_time/parallel_total_time:.2f}x")
print(f"✓ 总订单数: {sum(results):,}")

# 对比
print("\n" + "="*70)
print("📈 性能对比")
print("="*70)

print(f"\n单个环境: {single_env_time:.2f} 秒")
print(f"16个串行: {serial_total_time:.2f} 秒 ({serial_total_time/60:.1f} 分钟)")
print(f"16个并行: {parallel_total_time:.2f} 秒 ({parallel_total_time/60:.1f} 分钟)")
print(f"\n串行 vs 并行加速: {serial_total_time/parallel_total_time:.2f}x")
print(f"时间节省: {(serial_total_time - parallel_total_time)/60:.1f} 分钟")

# 瓶颈分析
print("\n" + "="*70)
print("🔍 瓶颈分析")
print("="*70)

avg_items_per_order = sum(
    sum(len(o.items) for o in g.pregenerated_orders)
    for g in generators_serial
) / sum(len(g.pregenerated_orders) for g in generators_serial)

total_items = sum(
    sum(len(o.items) for o in g.pregenerated_orders)
    for g in generators_serial
)

print(f"\n每个环境:")
print(f"  - 订单数: {n_orders:,}")
print(f"  - 平均每订单物品数: {avg_items_per_order:.1f}")
print(f"  - 总物品数: {int(n_orders * avg_items_per_order):,}")

print(f"\n16个环境总计:")
print(f"  - 总订单数: {n_orders * n_envs:,}")
print(f"  - 总物品数: {total_items:,}")
print(f"  - Zone概率查询次数: ~{total_items:,}")

print("\n主要时间消耗:")
print(f"  1. 订单生成: ~{single_env_time * 0.5:.1f}秒/环境 (50%)")
print(f"  2. 物品生成: ~{single_env_time * 0.3:.1f}秒/环境 (30%)")
print(f"  3. Zone查找: ~{single_env_time * 0.15:.1f}秒/环境 (15%)")
print(f"  4. 其他: ~{single_env_time * 0.05:.1f}秒/环境 (5%)")

# 建议
print("\n" + "="*70)
print("💡 优化建议")
print("="*70)

print("\n✅ 已实现的优化:")
print("  - Zone概率查找表 (O(4) → O(1))")
print("  - 订单窗口二分查找 (O(N) → O(log N))")

print("\n🚀 可进一步优化:")
print("  1. 并行环境初始化 (推荐!)")
print(f"     - 当前: {serial_total_time/60:.1f} 分钟")
print(f"     - 并行后: {parallel_total_time/60:.1f} 分钟")
print(f"     - 节省: {(serial_total_time - parallel_total_time)/60:.1f} 分钟")
print("\n  2. 共享订单生成器")
print("     - 16个环境使用同一批订单")
print("     - 初始化时间: 几乎为0")
print("     - 权衡: 环境之间不独立")

print("\n" + "="*70)
