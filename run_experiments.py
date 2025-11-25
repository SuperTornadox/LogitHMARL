#!/usr/bin/env python3
"""精简版实验入口：只保留核心配置与编排（彻底解耦）。"""

import os
import shutil
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

# 统一检测可用设备（优先 CUDA，其次 MPS，然后 CPU）
HAS_MPS = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
PRIMARY_DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if HAS_MPS else 'cpu')

# ==================== Reproducibility: Random Seed ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
elif HAS_MPS and hasattr(torch, 'mps'):
    try:
        torch.mps.manual_seed(SEED)  # type: ignore[attr-defined]
    except Exception:
        pass
print(f"[SEED] Set random seed to {SEED} for reproducibility")

# ==================== Mode Control: Test vs Full vs Large vs UltraLarge ====================
# 环境变量 MODE 控制训练规模：
# - MODE=test: 测试模式（小规模，本地MacBook快速验证）
# - MODE=large: 大规模环境评估（100×100，只评估不训练）
# - MODE=ultralarge: 超大规模训练（100×100，500 pickers，完整训练）
# - MODE=full 或未设置: 完整训练模式（大规模，Google Colab A100）
MODE = os.environ.get('MODE', 'full').lower()

if MODE == 'ultralarge':
    print("\n" + "="*60)
    print("🚀🚀 ULTRA-LARGE SCALE MODE: 100×100环境，500 pickers训练")
    print("="*60)
    TRAINING_STEPS = 20000      # 更多训练步数，大环境需要更长时间收敛
    BATCH_SIZE = 4096           # 超大批量
    BUFFER_SIZE = 200000        # 超大buffer
    N_ENVS = 8                  # 8个并行环境
    LEARNING_RATE = 5e-5        # 更小的学习率，更稳定
    EVAL_MAX_STEPS = 300        # 完整评估
    print(f"  - 环境规模: 100×100")
    print(f"  - Pickers: 500")
    print(f"  - 训练步数: {TRAINING_STEPS}")
    print(f"  - 批量大小: {BATCH_SIZE}")
    print(f"  - 并行环境: {N_ENVS}")
    print(f"  - 评估步数: {EVAL_MAX_STEPS}")
    print("="*60 + "\n")
elif MODE == 'large':
    print("\n" + "="*60)
    print("🏭 LARGE SCALE MODE: 100×100环境评估")
    print("="*60)
    TRAINING_STEPS = 0          # 不训练，只评估
    BATCH_SIZE = 64
    BUFFER_SIZE = 5000
    N_ENVS = 1                  # 单环境评估
    LEARNING_RATE = 3e-4
    EVAL_MAX_STEPS = 200        # 200步评估
    print(f"  - 环境规模: 100×100")
    print(f"  - Pickers: 200")
    print(f"  - 训练步数: {TRAINING_STEPS} (只评估)")
    print(f"  - 评估步数: {EVAL_MAX_STEPS}")
    print("="*60 + "\n")
elif MODE == 'test':
    print("\n" + "="*60)
    print("🧪 TEST MODE: 小规模快速验证（适合本地MacBook）")
    print("="*60)
    TRAINING_STEPS = 1000       # 快速测试规模
    BATCH_SIZE = 64             # 减轻内存压力
    BUFFER_SIZE = 5000          # 小buffer（后续ASYNC再调小）
    N_ENVS = 2                  # 适度并行，降低CPU负担
    LEARNING_RATE = 3e-4        # 标准学习率
    EVAL_MAX_STEPS = 200        # 测试评估步数略短，降低压力
    print(f"  - 训练步数: {TRAINING_STEPS}")
    print(f"  - 批量大小: {BATCH_SIZE}")
    print(f"  - 并行环境: {N_ENVS}")
    print(f"  - 评估步数: {EVAL_MAX_STEPS}")
    print("="*60 + "\n")
elif MODE == 'medium':
    print("\n" + "="*60)
    print("📏 MEDIUM MODE: 中等规模环境（验证计算复杂度）")
    print("="*60)
    TRAINING_STEPS = 1000       # test模式的训练步数
    BATCH_SIZE = 64             # 适中批量
    BUFFER_SIZE = 5000          # 小buffer
    N_ENVS = 2                  # 并行环境
    LEARNING_RATE = 3e-4        # 标准学习率
    EVAL_MAX_STEPS = 200        # 评估步数
    print(f"  - 环境规模: 24×24 (4倍于12×12)")
    print(f"  - 货架数: ~180 (4倍于45)")
    print(f"  - Pickers: 30")
    print(f"  - 训练步数: {TRAINING_STEPS}")
    print(f"  - 目的: 验证Optimal的O(n²)计算复杂度")
    print("="*60 + "\n")
elif MODE == 'test_transfer' or MODE == 'test_large':
    print("\n" + "="*60)
    print("🧪➡️🏭 TEST TRANSFER MODE: 小环境快速训练 + 大环境迁移评估")
    print("="*60)
    TRAINING_STEPS = 1000       # test模式的训练步数
    BATCH_SIZE = 64             # test模式的批量
    BUFFER_SIZE = 5000          # test模式的buffer
    N_ENVS = 2                  # test模式的并行环境
    LEARNING_RATE = 3e-4        # 标准学习率
    EVAL_MAX_STEPS = 200        # 评估步数
    print(f"  - 训练: 12×12环境, 15 pickers, {TRAINING_STEPS} steps")
    print(f"  - 评估: 100×100环境, 200 pickers, {EVAL_MAX_STEPS} steps")
    print(f"  - 批量大小: {BATCH_SIZE}")
    print(f"  - 并行环境: {N_ENVS}")
    print(f"  - 策略: 小环境快速训练后迁移到大环境评估")
    print("="*60 + "\n")
else:
    print("\n" + "="*60)
    print("🚀 FULL MODE: 12×12完整训练 + 100×100迁移评估")
    print("="*60)
    TRAINING_STEPS = 10000      # 10k步完整训练实验 (modified for comprehensive tests)
    BATCH_SIZE = 2048           # 大批量
    BUFFER_SIZE = 100000        # 大buffer
    N_ENVS = 8                  # 8个并行环境（平衡效率和GIL竞争）
    LEARNING_RATE = 1e-4        # 更稳定的学习率
    EVAL_MAX_STEPS = 300        # 完整评估
    print(f"  - 环境: 12×12, 15 pickers")
    print(f"  - 训练步数: {TRAINING_STEPS}")
    print(f"  - 批量大小: {BATCH_SIZE}")
    print(f"  - 并行环境: {N_ENVS}")
    print(f"  - 评估步数: {EVAL_MAX_STEPS}")
    print(f"  - 策略: 训练后迁移到100×100评估")
    print("="*60 + "\n")

# 评估参数（根据MODE动态调整）
eval_cfg = dict(
    n_episodes=1,
    # 设为0表示不按订单数提前结束，让拣货员在同一集内连续执行多个任务
    target_orders=0,
    max_time_limit=EVAL_MAX_STEPS,  # 动态：test=50, full=300
    #一个step为2秒
)

# 订单生成与截止期配置（集中在 run_experiments.py）
# - 到达节奏：base_rate、peak_hours、倍率；可选将到达模式按仿真时长缩放
# - 截止期：按类型的范围 + 全局缩放
# 注意：环境会把 simulation_hours 与 episode_duration 对齐
ORDER_CONFIG = {
    # 到达节奏 - CURRICULUM LEARNING: 降低订单率
    'base_rate': 1200,                 # 基础到达率（订单/小时）← 降低50% (原2400)
    'peak_hours': [(9, 12), (14, 17)], # 高峰时段（与 period 同单位，默认24h制）
    'peak_multiplier': 3.0,            # 高峰倍率 ← 降低 (原4.0)
    'off_peak_multiplier': 0.7,        # 非高峰倍率（夜间还会额外*0.5）← 提高 (原0.6)
    'use_pregeneration': True,         # 是否预生成整段订单流
    'n_skus': 100,                    # SKU 数（影响分布/区域）
    # 模式周期与缩放：保持按“整天(24h)”的模式，再缩放到模拟时长
    'pattern_period_hours': 24,        # 到达模式周期（小时）
    'scale_pattern_to_simulation': True,  # 将仿真时长映射为完整的一个周期
    # 截止期（按订单类型）- CURRICULUM LEARNING: 放宽时间限制
    'deadline_ranges': {
        'urgent': (0.06, 0.12),  # 3.6-7.2 分钟 ← 翻倍 (原 0.03-0.06)
        'regular': (0.16, 0.30), # 9.6-18 分钟 ← 翻倍 (原 0.08-0.15)
        'bulk': (0.20, 0.36),    # 12-21.6 分钟 ← 翻倍 (原 0.10-0.18)
    },
    'deadline_scale': 1.2,           # 全局缩放 ← 放宽20% (原 0.6)
    'urgent_order_prob': 0.40,       # 提升紧急订单比例
    # VARIANT-1: 突发式订单（burst arrivals）配置
    'bursty_prob': 0.35,             # 每个时间窗口35%概率触发突发
    'burst_size': (12, 28),          # 每次突发生成12-28个订单
    'burst_zone_correlation': 0.90,  # 突发订单中90%集中在同一区域
    # 更极端的分区峰值时段：与分区拥堵相呼应，突出"组内成簇、组间差异"
    # 概率向量长度=4，和为1；会被订单生成器在每件物品选择SKU区域时使用
        'zone_peak_schedule': [
            # 前1/3：Zone 0 爆单 (40%)，Zone 1 次之 (30%)，Zone 2/3 冷清
            (0.0, 8.0,  [0.40, 0.30, 0.15, 0.15]),
            # 中1/3：Zone 2 爆单 (40%)，Zone 3 次之 (30%)
            (8.0, 16.0, [0.15, 0.15, 0.40, 0.30]),
            # 后1/3：Zone 1/3 交叉爆单
            (16.0, 24.0,[0.20, 0.30, 0.20, 0.30]),
        ],
    }

# 统一环境参数（在 MODE 判断外定义）
env_cfg = dict(
        width=12,
        height=12,
        n_pickers=20,
        n_stations=5,
        # 与订单生成配置保持一致
        order_rate=ORDER_CONFIG.get('base_rate', 2000),
        max_items=1,  # 单个订单最多拣货数量
        min_forklifts=4,  # 更紧俏的重载资源
        forklift_ratio=0.25,  # 叉车占比25%
        # 垂直过道所在列（用于布局留空与 Zone 左/右划分）；范围[1, width-2]
        col_aisle=6,  # 垂直过道位置
        # 速度/拥堵/载重效率配置
        speed_config={
            'base_speed': {'regular': 1.0, 'forklift': 1.5},
            'carry_alpha': {'regular': 1.2, 'forklift': 0.6}, #载重减速强度（按 weight/100 缩放）
            'congestion_mult': 0.7,
        },
        # VARIANT-1: 重量阈值（降低阈值，增加重型任务比例）
        weight_thresholds={
            'medium': 18.0,  # medium阈值进一步降低
            'heavy': 45.0,   # heavy 分类阈值降低（更多heavy任务）
            'forklift_only': 65.0,  # 降低叉车专属阈值 (更多叉车专属任务)
        },
        # VARIANT-1: 区域容量约束
        zone_capacity={
            0: 3,  # 稍微放宽 (from 2) -> 给RL试错空间
            1: 3,
            2: 4,
            3: 3,
        },
        zone_overcapacity_penalty=-2.0,      # 大幅降低致死打击 (from -5) -> 鼓励尝试
        zone_overcapacity_speed_mult=0.5,    # 稍微能动一点 (from 0.4) -> 别直接卡死
        stuck_release_steps=80,
        stuck_release_penalty=-1.0,
    )

if MODE == 'ultralarge':
    # 100×100超大规模环境配置 - 500 pickers
    env_cfg.update({
        'width': 100,
        'height': 100,
        'n_pickers': 500,        # 500个拣货员
        'n_stations': 12,        # 更多工作站
        'n_shelves': 3000,       # 30% of grid
        'col_aisle': 50,         # 中间过道
        'min_forklifts': 100,    # 更多叉车
        'forklift_ratio': 0.20,  # 500个pickers中20%是叉车
    })
    ORDER_CONFIG.update({
        'base_rate': 100000,     # 100k订单/小时 (匹配500 pickers规模)
        'n_skus': 300,           # 更多SKUs
        'peak_multiplier': 2.5,  # 降低峰值倍率，避免过载
        'bursty_prob': 0.25,     # 降低突发概率
        'burst_size': (15, 35),  # 适中的突发大小
    })
elif MODE == 'large':
    # 100×100大规模环境配置
    env_cfg.update({
        'width': 100,
        'height': 100,
        'n_pickers': 200,
        'n_stations': 8,
        'n_shelves': 3000,  # 30% of grid
        'col_aisle': 50,    # 中间过道
        'min_forklifts': 40,
        'forklift_ratio': 0.25,
    })
    ORDER_CONFIG.update({
        'base_rate': 40000,  # 40k订单/小时
        'n_skus': 200,
    })
elif MODE == 'test':
    ORDER_CONFIG.update({
        'deadline_scale': 0.65,
        'base_rate': 3000,
        'bursty_prob': 0.20,
        'burst_size': (10, 20),
        'burst_zone_correlation': 0.50,
        'zone_peak_schedule': [
            (0.0, 24.0, [0.25, 0.25, 0.25, 0.25]),
        ],
    })
    env_cfg['n_pickers'] = 15  # 减少测试时的agent数量，缓解CPU压力
    env_cfg['zone_overcapacity_penalty'] = -2.0
    env_cfg['zone_overcapacity_speed_mult'] = 0.5
elif MODE == 'medium':
    # 24×24中等环境配置（4倍于12×12）
    env_cfg.update({
        'width': 24,
        'height': 24,
        'n_pickers': 30,      # 2倍于test模式的15
        'n_stations': 6,      # 增加工作站
        'n_shelves': 180,     # 4倍于45个货架（保持30%覆盖率）
        'col_aisle': 12,      # 中间过道位置（24/2）
        'min_forklifts': 6,   # 增加叉车
        'forklift_ratio': 0.25,
    })
    ORDER_CONFIG.update({
        'base_rate': 6000,    # 2倍订单率（匹配30 pickers）
        'n_skus': 150,        # 更多SKUs
        'deadline_scale': 0.65,
        'bursty_prob': 0.20,
        'burst_size': (15, 30),  # 稍大的突发
        'burst_zone_correlation': 0.50,
    })
elif MODE == 'test_transfer' or MODE == 'test_large':
    # test_transfer: 100×100大环境配置（快速训练1000步+评估）
    env_cfg.update({
        'width': 100,
        'height': 100,
        'n_pickers': 200,
        'n_stations': 8,
        'n_shelves': 3000,
        'col_aisle': 50,
        'min_forklifts': 40,
        'forklift_ratio': 0.25,
    })
    ORDER_CONFIG.update({
        'base_rate': 40000,  # 40k订单/小时
        'n_skus': 200,
        'deadline_scale': 0.65,
    })


# 训练参数（根据MODE动态调整）
train_cfg = dict(
    training_steps=TRAINING_STEPS,  # test=1k, full=200k
    batch_size=BATCH_SIZE,          # test=64, full=2048
    learning_rate=LEARNING_RATE,    # test=3e-4, full=1e-4
    buffer_size=BUFFER_SIZE,        # test=5k, full=100k
    update_freq=4,
    target_update_freq=50,
    hidden_dim=256,
)

# 调试/可视化选项（根据MODE动态调整）
debug_cfg = dict(
    enable=True,                  # 启用逐步调试/帧保存（所有方法生效）
    log_every=100,                  # 每步打印一次速度与动作（需要打印时设>=1）
    save_plots=True,  # 强制开启保存图片以便生成动图
    frames_dir='results',         # 保存到 results 下（每集一个子目录）
    frame_size=(20, 20),            # 帧图尺寸 (width, height) in inches
    plot_every=10,                 # 每步保存一帧
    debug_first_episode_only=True, # 仅保存/打印首个 episode，避免过多输出
    make_animation=True,              # 所有模式都生成动画
    animation_fps=10,              # 动图/视频帧率
    clean_before_run=False,       # 【重要修改】默认禁用清理，防止误删训练好的模型！
)


# VARIANT-1: NL-HMARL hyperparameters (根据MODE动态调整)
nl_cfg = dict(
    # Model
    hidden_dim=train_cfg.get('hidden_dim', 256),
    n_nests=8,              # VARIANT-1: 8个巢 (zone×urgency: 4×2)
    learn_eta=True,         # 学习每个巢的 eta
    eta_init=0.5,           # IMPROVED: 降低eta初值以增加巢内竞争 (from 0.7)
    device=PRIMARY_DEVICE,
    # Training
    manager_lr=train_cfg.get('learning_rate', 1e-3) * 1.5,  # IMPROVED: 更快的manager学习 (1.5x)
    worker_lr=train_cfg.get('learning_rate', 1e-3) * 1.2,   # IMPROVED: 稍快的worker学习 (1.2x)
    max_tasks=20,           # number of tasks the manager considers per step
    gamma=0.99,             # IMPROVED: 恢复标准gamma以更好地考虑长期奖励
    update_every=8,         # reserved; per-step update in current impl
    entropy_coef_manager=0.08,  # IMPROVED: 进一步增加探索 (from 0.05)
    entropy_coef_workers=0.05,  # IMPROVED: 大幅增加worker探索 (from 0.01)
    train_log_every=max(1, train_cfg.get('training_steps', 1) // 200),
    # Evaluation
    deterministic_eval=False,
    n_envs=N_ENVS,  # 动态：test=1, full=16
)

# VARIANT-1: DQN hyperparameters (支持GPU加速)
dqn_cfg = dict(
    device=PRIMARY_DEVICE,
    n_envs=1,
)


# 统一速度函数示例：
# - 叉车(FORKLIFT)：移动速度恒为其 base_speed
# - 普通工人：未携货不降速；携货按移动效率 eff = (heavy-weight+10)/(heavy+10) * (1-拥堵减速)
def speed_fn(env):
    speeds = {}
    for p in env.pickers:
        t = getattr(p, 'current_task', None)
        base = float(getattr(p, 'speed', 1.0))
        if getattr(p.type, 'name', '') == 'FORKLIFT':
            # 叉车恒定为 base_speed（不受重量/拥堵影响）
            speeds[p.id] = max(0.3, base)
        else:
            # 普通工人：未携货不降速；携货按效率降速
            if p.carrying_items and t is not None:
                eff = float(env._compute_movement_efficiency(p, t))
            else:
                eff = 1.0
            # 不强制下限：让速度真实反映效率（可能小于1格/步）
            # 为避免极端拥堵/重量导致速度为0，设置正下限
            speeds[p.id] = max(0.3, base * max(0.3, eff))
    return speeds


def combine_gifs_grid(results_dir='results', output_path="results/all_methods_grid.gif"):
    import os
    import math
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[warn] PIL not installed, skipping GIF combination.")
        return

    # Find all method subdirectories with animation.gif
    method_dirs = []
    if not os.path.exists(results_dir):
        return

    for d in os.listdir(results_dir):
        path = os.path.join(results_dir, d)
        if os.path.isdir(path) and "_ep00" in d:
            gif_path = os.path.join(path, "animation.gif")
            if os.path.exists(gif_path):
                method_dirs.append((d, gif_path))
    
    method_dirs.sort()
    if not method_dirs:
        print("[info] No animations found to combine.")
        return

    gifs = []
    labels = []
    max_frames = 0
    base_duration = 200
    
    print(f"[info] Found {len(method_dirs)} animations to combine...")
    
    # Load font - INCREASED SIZE to 120 for better visibility
    font = None
    try:
        # Try system fonts for Mac
        font_candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf"
        ]
        for fp in font_candidates:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, 120)  # Increased from 60 to 120
                print(f"[info] Using font: {fp} with size 120")
                break
        if font is None:
             # Try load_default with size (Pillow 10+)
             try:
                font = ImageFont.load_default(size=120)
             except TypeError:
                font = ImageFont.load_default()
    except Exception as e:
        print(f"[warn] Font loading failed: {e}")
        font = None

    for name, path in method_dirs:
        try:
            im = Image.open(path)
            # Extract frames
            frames = []
            try:
                while True:
                    frames.append(im.copy())
                    im.seek(im.tell() + 1)
            except EOFError:
                pass
            
            if frames:
                gifs.append(frames)
                # Clean label: remove _ep00
                label = name.replace('_ep00', '')
                labels.append(label)
                max_frames = max(max_frames, len(frames))
                if 'duration' in frames[0].info:
                    base_duration = frames[0].info['duration']
        except Exception as e:
            print(f"[warn] Error reading {path}: {e}")
    
    if not gifs:
        return

    # Grid dimensions
    n = len(gifs)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    # Frame size (assume all same size, use first as ref)
    w, h = gifs[0][0].size
    
    combined_frames = []
    
    # Create frames
    for i in range(max_frames):
        # White background
        grid_w = cols * w
        grid_h = rows * h
        canvas = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        
        for idx, frames in enumerate(gifs):
            r = idx // cols
            c = idx % cols
            
            # Loop frames if this gif is shorter
            frame = frames[i % len(frames)]
            
            x = c * w
            y = r * h
            canvas.paste(frame, (x, y))
            
            # Draw label with enhanced visibility
            text = labels[idx]

            # Draw background for text
            # Get text size
            tw, th = 200, 60  # Larger fallback size
            if font:
                if hasattr(draw, 'textbbox'):
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                elif hasattr(draw, 'textsize'):
                    tw, th = draw.textsize(text, font=font)

            # Position: Top-Center with larger padding for better visibility
            tx = x + (w - tw) // 2  # Center horizontally within this cell
            ty = y + 20  # More top padding

            # Draw a semi-transparent background with thick black border for high contrast
            padding = 20  # Increased padding
            draw.rectangle(
                (tx - padding, ty - padding, tx + tw + padding, ty + th + padding),
                fill=(255, 255, 255),  # White background
                outline=(0, 0, 0),     # Black border
                width=5                # Thicker border from 3 to 5
            )

            # Draw text in bold black for maximum contrast
            draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
            
        combined_frames.append(canvas)
        
    # Save
    if combined_frames:
        print(f"[info] Saving combined GIF to {output_path}...")
        combined_frames[0].save(
            output_path,
            save_all=True,
            append_images=combined_frames[1:],
            duration=base_duration,
            loop=0
        )
        print(f"✅ Combined GIF saved: {output_path}")



def main():
    # 将 src 加入 sys.path，使用标准 import
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from exp.evaluate import evaluate_method
    from env.dynamic_warehouse_env import DynamicWarehouseEnv

    methods = [
        'NL-HMARL-AC',    # NL 管理层 + 工人层 A-C 学习
        'NL-HMARL',       # NL 管理层 + 传统工人
        'Softmax-AC',     # Softmax 管理层 + 工人层 A-C 学习
        'Softmax',        # Softmax 管理层 + 传统工人
        'DQN-Guided',     # DQN 引导
        'DQN-Pure',       # DQN 纯学习
        'S-Shape',        # S-Shape 分配
        'Return',         # Return 分配
        'Optimal',        # Optimal 分配
    ]
    # 允许用命令行参数快速筛选/续跑：
    #   --only m1,m2     只跑指定方法
    #   --start-from m   从指定方法开始（包括该方法）
    try:
        args = sys.argv[1:]
        if '--only' in args:
            i = args.index('--only')
            sel = args[i+1].split(',') if i+1 < len(args) else []
            sel = [s.strip() for s in sel if s.strip()]
            if sel:
                methods = sel
        elif '--start-from' in args:
            i = args.index('--start-from')
            name = args[i+1] if i+1 < len(args) else None
            if name and name in methods:
                methods = methods[methods.index(name):]
    except Exception:
        pass
    print("methods: ", methods)

    # Devices (自动检测GPU)
    nl_dev = PRIMARY_DEVICE
    dqn_dev = PRIMARY_DEVICE
    print(f"Devices -> NL: {nl_dev}, DQN: {dqn_dev}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    elif HAS_MPS:
        print("MPS: Apple Silicon GPU (Metal Performance Shaders)")

    # 先生成订单生成的 pattern（保存到 results/order_arrival_pattern.png）
    try:
        from env.order_generation import NonHomogeneousPoissonOrderGenerator
        og_cfg = dict(ORDER_CONFIG)
        # 与本次评估时长对齐：根据 step 时长与最大步数估算模拟总时长（小时）
        # 注意：evaluate_method 内部的环境使用 time_step=2.0 秒
        step_seconds = 2.0
        sim_hours_for_plot = float(eval_cfg['max_time_limit']) * step_seconds / 3600.0
        og_cfg['simulation_hours'] = max(1e-6, sim_hours_for_plot)
        gen = NonHomogeneousPoissonOrderGenerator(og_cfg)
        os.makedirs('results', exist_ok=True)
        # 先删除旧图，避免查看器缓存或覆盖不生效的困扰
        try:
            old_paths = [os.path.join('results', 'order_arrival_pattern.png'), 'order_arrival_pattern.png']
            for _p in old_paths:
                if os.path.exists(_p):
                    os.remove(_p)
        except Exception:
            pass
        fig = gen.visualize_arrival_pattern()
        # 追加保存一份到 results 下（会覆盖同名文件）
        try:
            import matplotlib.pyplot as _plt
            _plt.close(fig)
        except Exception:
            pass
        print('已保存: results/order_arrival_pattern.png')
    except Exception as e:
        print(f"[warn] 订单到达模式图生成失败: {e}")

    # 清理上一轮产物（帧目录与训练指标）
    if debug_cfg.get('clean_before_run', False):
        frames_root = debug_cfg.get('frames_dir', 'results')
        metrics_root = os.path.join('results', 'train_metrics')
        try:
            os.makedirs(frames_root, exist_ok=True)
            for m in methods:
                # 清理帧：results/<method>_ep*/
                prefix = f"{m}_ep"
                for name in os.listdir(frames_root):
                    p = os.path.join(frames_root, name)
                    if os.path.isdir(p) and name.startswith(prefix):
                        shutil.rmtree(p, ignore_errors=True)
                # 清理该方法训练指标目录
                mp = os.path.join(metrics_root, m)
                if os.path.isdir(mp):
                    shutil.rmtree(mp, ignore_errors=True)
        except Exception as e:
            print(f"[warn] cleanup skipped: {e}")

    results = []
    import time
    start_time = time.time()
    method_times = []  # 记录每个方法的用时

    for method_idx, m in enumerate(methods, 1):
        method_start = time.time()

        print(f"\n{'='*70}")
        print(f"📊 方法进度: [{method_idx}/{len(methods)}] {m}")
        if method_times:
            avg_time = sum(method_times) / len(method_times)
            remaining = (len(methods) - method_idx + 1) * avg_time
            print(f"⏱️  预估剩余时间: {remaining/60:.1f} 分钟 ({remaining/3600:.1f} 小时)")
        print(f"{'='*70}")

        # 单次配置，所有方法一致使用
        debug_this = debug_cfg['enable']
        # 动态设置订单生成的模拟时长：避免训练中订单耗尽
        # 训练步长对应的总小时数（每步2秒）
        _step_seconds = 2.0
        _sim_hours_train = float(train_cfg['training_steps']) * _step_seconds / 3600.0
        _oc = dict(ORDER_CONFIG)
        # 若用户未显式提供，则用训练时长的1.5倍保障有量
        _oc['simulation_hours'] = max(float(_oc.get('simulation_hours', 0.0)), max(2.0, 1.5 * _sim_hours_train))

        r = evaluate_method(
            m,
            env_cfg['width'], env_cfg['height'], env_cfg['n_pickers'], env_cfg.get('n_shelves', 0), env_cfg['n_stations'],
            env_cfg['order_rate'], env_cfg['max_items'],
            eval_cfg['n_episodes'], eval_cfg['target_orders'], eval_cfg['max_time_limit'],
            train_cfg['training_steps'], train_cfg['batch_size'], train_cfg['learning_rate'],
            train_cfg['buffer_size'], train_cfg['update_freq'], train_cfg['target_update_freq'],
            train_cfg['hidden_dim'],
            # debug/viz
            verbose=debug_this,
            log_every=debug_cfg['log_every'],
            save_plots=(debug_cfg['save_plots'] and debug_this),
            plot_dir=debug_cfg['frames_dir'],
            plot_figsize=debug_cfg.get('frame_size'),
            plot_every=debug_cfg['plot_every'],
            debug_first_episode_only=debug_cfg['debug_first_episode_only'],
            make_animation=debug_cfg['make_animation'],
            animation_fps=debug_cfg['animation_fps'],
            # 直接使用 DynamicWarehouseEnv 创建环境
            env_ctor=lambda cfg: DynamicWarehouseEnv(cfg),
            # 传递额外的环境配置，确保叉车生成
            env_extra={
                'min_forklifts': env_cfg.get('min_forklifts', 0),
                'forklift_ratio': env_cfg.get('forklift_ratio', 0.2),
                'speed_config': env_cfg.get('speed_config'),
                'weight_thresholds': env_cfg.get('weight_thresholds'),
                'reward_config': env_cfg.get('reward_config'),
                # 将订单生成/截止期配置合入生成器（深合并）
                'order_config': _oc,
                # 分区拥堵时段加成：在订单高发区制造“额外拥堵”，逼迫策略在不同阶段动态选巢
                # 数值为对拥堵减速的额外加成（0..1），例如 0.3 表示在拥堵时额外+0.3 的减速
                # 更极端的“分区拥堵时段”设置：
                # - 凌晨/早高峰重压 0/1 区（左侧/右侧上半区）
                # - 午后/傍晚重压 2/3 区（左侧/右侧下半区）
                # - 夜间也保留一定均匀拥堵，避免完全无差异
                # 数值为额外拥堵减速（0..1），会与局部拥堵叠加并裁剪到 [0,1]
                'zone_congestion_schedule': [
                    # 移除强制拥堵，改为均匀微小扰动，让RL自己去应对真实的动态拥堵
                    (0.0, 24.0, [0.10, 0.10, 0.10, 0.10]),
                ],
            },
            # 统一速度函数（可按需替换）：速度完全由该函数决定
            speed_function=speed_fn,
            # 分配时的 value 权重（越大越偏向高价值）
            assign_value_weight=0.05,
            # NL-HMARL training/eval config
            nl_cfg=nl_cfg,
            # DQN training config
            dqn_cfg=dqn_cfg,
        )
        results.append(r)

        # 记录方法完成时间
        method_elapsed = time.time() - method_start
        method_times.append(method_elapsed)
        print(f"\n✅ {m} 完成! 用时: {method_elapsed/60:.1f} 分钟")
        print(f"📈 总进度: {method_idx}/{len(methods)} ({method_idx/len(methods)*100:.1f}%)")

        total_elapsed = time.time() - start_time
        if method_idx < len(methods):
            avg_time = sum(method_times) / len(method_times)
            remaining_methods = len(methods) - method_idx
            est_remaining = remaining_methods * avg_time
            est_total = total_elapsed + est_remaining
            print(f"⏱️  已用时间: {total_elapsed/60:.1f} 分钟, 预估剩余: {est_remaining/60:.1f} 分钟")
            print(f"🎯 预计完成时间: {est_total/60:.1f} 分钟 ({est_total/3600:.2f} 小时)")

    # 所有方法完成
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"🎉 所有实验完成!")
    print(f"⏱️  总用时: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")
    print(f"📊 完成方法数: {len(methods)}")
    print(f"{'='*70}\n")

    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    # 若存在历史结果，合并：用本次结果覆盖同名 method 的旧行，保留其它方法的旧行
    try:
        old_path = os.path.join('results', 'results.csv')
        if os.path.exists(old_path):
            old = pd.read_csv(old_path)
            if 'method' in old.columns and 'method' in df.columns:
                new_methods = set(df['method'].astype(str).tolist())
                old = old[~old['method'].astype(str).isin(new_methods)]
                df = pd.concat([old, df], ignore_index=True)
    except Exception as e:
        print(f"[warn] merging previous results failed: {e}")
    df.to_csv('results/results.csv', index=False)

    print("\n📋 各方法任务收益/罚没明细：")
    for _, row in df.iterrows():
        try:
            method_name = str(row.get('method', 'unknown'))
            raw_v = float(row.get('raw_value', 0.0))
            dec_v = float(row.get('decayed_value', 0.0))
            pen = float(row.get('penalty', 0.0))
            done = int(row.get('tasks_completed', 0))
            print(f"  - {method_name:<12} raw={raw_v:.1f}, decayed={dec_v:.1f}, penalty={pen:.1f}, tasks={done}")
        except Exception as _e:
            print(f"  - {row}")

    print("\nResults summary:\n", df)
    print("\nResults summary:\n", df)
    print("saved -> results/results.csv (merged if previous existed)")

    # Combine GIFs
    combine_gifs_grid()



if __name__ == '__main__':
    main()
