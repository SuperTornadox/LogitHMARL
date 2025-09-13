#!/usr/bin/env python3
"""精简版实验入口：只保留核心配置与编排（彻底解耦）。"""

import os
import shutil
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 评估参数（固定配置于 main，可以按需修改）
eval_cfg = dict(
    n_episodes=3,
    # 设为0表示不按订单数提前结束，让拣货员在同一集内连续执行多个任务
    target_orders=0,
    max_time_limit=30*60,#以step为单位
    #一个step为2秒
)

# 订单生成与截止期配置（集中在 run_experiments.py）
# - 到达节奏：base_rate、peak_hours、倍率；可选将到达模式按仿真时长缩放
# - 截止期：按类型的范围 + 全局缩放
# 注意：环境会把 simulation_hours 与 episode_duration 对齐
ORDER_CONFIG = {
    # 到达节奏
    'base_rate':2000,                 # 基础到达率（订单/小时）
    'peak_hours': [(9, 12), (14, 17)], # 高峰时段（与 period 同单位，默认24h制）
    'peak_multiplier': 3.0,            # 高峰倍率
    'off_peak_multiplier': 0.7,        # 非高峰倍率（夜间还会额外*0.5）
    'use_pregeneration': True,         # 是否预生成整段订单流
    'n_skus': 100,                    # SKU 数（影响分布/区域）
    # 模式周期与缩放：保持按“整天(24h)”的模式，再缩放到模拟时长
    'pattern_period_hours': 24,        # 到达模式周期（小时）
    'scale_pattern_to_simulation': True,  # 将仿真时长映射为完整的一个周期
    # 截止期（按订单类型）
    'deadline_ranges': {
        'urgent': (0.05,0.1),  # 6-12 分钟
        'regular': (0.1, 0.2), # 12-24 分钟
        'bulk': (0.1, 0.2),    # 12-24 分钟
    },
    'deadline_scale': 1.0,           # 全局缩放，<1 更严格，>1 更宽松
}

# 统一环境参数
env_cfg = dict(
    width=100,
    height=100,
    n_pickers=20,
    n_stations=5,
    # 与订单生成配置保持一致
    order_rate=ORDER_CONFIG.get('base_rate', 2000),
    max_items=1,  # 单个订单最多拣货数量
    min_forklifts=1,  # 至少1辆叉车
    forklift_ratio=0.2,  # 叉车占比（与 min_forklifts 取 max）
    # 垂直过道所在列（用于布局留空与 Zone 左/右划分）；范围[1, width-2]
    col_aisle=5,  # 垂直过道位置
    # 速度/拥堵/载重效率配置
    speed_config={
        'base_speed': {'regular': 1.0, 'forklift': 1.5},
        'carry_alpha': {'regular': 1.2, 'forklift': 0.6}, #载重减速强度（按 weight/100 缩放）
        'congestion_mult': 0.7,
    },
    # 重量阈值（用于效率公式与分类参考）
    weight_thresholds={
        'medium': 30.0,  # medium 的参考权重（用于 (medium - weight + 10)/(medium + 10)）
        'heavy': 50.0,   # heavy 分类阈值（不再强制叉车）
        'forklift_only': 90.0,  # 新增：仅叉车可搬阈值
    },
)


# 训练参数（Flat-DQN）
train_cfg = dict(
    training_steps=100000,
    batch_size=256,
    learning_rate=1e-3,
    buffer_size=10000,
    update_freq=4,
    target_update_freq=50,
    hidden_dim=256,
)

# 调试/可视化选项（一次配置，适用于所有方法）
debug_cfg = dict(
    enable=False,                  # 启用逐步调试/帧保存（所有方法生效）
    log_every=10,                  # 每步打印一次速度与动作（需要打印时设>=1）
    save_plots=True,              # 保存每步帧图
    frames_dir='results',         # 保存到 results 下（每集一个子目录）
    frame_size=(20, 20),            # 帧图尺寸 (width, height) in inches
    plot_every=1,                 # 每步保存一帧
    debug_first_episode_only=True, # 仅保存/打印首个 episode，避免过多输出
    make_animation=True,          # 训练后自动合成 GIF/MP4
    animation_fps=10,              # 动图/视频帧率
    clean_before_run=True,        # 运行前清理上次的帧目录与训练指标
)


# NL-HMARL hyperparameters (exposed to evaluate)
nl_cfg = dict(
    # Model
    hidden_dim=train_cfg.get('hidden_dim', 256),
    n_nests=2,              # number of nests: 0=非叉车，1=叉车
    learn_eta=False,        # whether to learn eta per nest
    eta_init=1.0,           # initial eta value per nest
    device='auto',  # 'auto' | 'cpu' | 'cuda' | 'cuda:0' | 'mps'
    # Training
    manager_lr=train_cfg.get('learning_rate', 1e-3),
    worker_lr=train_cfg.get('learning_rate', 1e-3),
    max_tasks=20,           # number of tasks the manager considers per step
    gamma=0.99,
    update_every=8,         # reserved; per-step update in current impl
    entropy_coef_manager=0.01,
    entropy_coef_workers=0.01,
    train_log_every=max(1, train_cfg.get('training_steps', 1) // 200),
    # Evaluation
    deterministic_eval=False,
)

# DQN hyperparameters (device only for now)
dqn_cfg = dict(
    device='auto',  # 'auto' | 'cpu' | 'cuda' | 'cuda:0' | 'mps'
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
            speeds[p.id] = base
        else:
            # 普通工人：未携货不降速；携货按效率降速
            if p.carrying_items and t is not None:
                eff = float(env._compute_movement_efficiency(p, t))
            else:
                eff = 1.0
            # 不强制下限：让速度真实反映效率（可能小于1格/步）
            speeds[p.id] = base * eff
    return speeds


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
        'NL-HMARL',       # NL 管理层 + 工人启发式移动
        'Softmax',        # Softmax 管理层 + 工人启发式移动
        'Softmax-AC',     # Softmax 管理层 + 工人层 A-C 学习
    ]
    print("methods: ", methods)

    # Show current devices for NL and DQN before running
    def _resolve_device(dev):
        try:
            import torch as _torch  # type: ignore
            if str(dev).lower() == 'auto':
                if _torch.cuda.is_available():
                    return 'cuda'
                if hasattr(_torch.backends, 'mps') and getattr(_torch.backends.mps, 'is_available', lambda: False)():
                    return 'mps'
                return 'cpu'
            return str(dev)
        except Exception:
            return 'cpu' if str(dev).lower() == 'auto' else str(dev)

    nl_dev = _resolve_device(nl_cfg.get('device', 'cpu'))
    dqn_dev = _resolve_device(dqn_cfg.get('device', 'auto'))
    print(f"Devices -> NL: {nl_dev}, DQN: {dqn_dev}")

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
    for m in methods:
        print(f"\n>>> Evaluating {m} ...")
        # 单次配置，所有方法一致使用
        debug_this = debug_cfg['enable']
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
                'col_aisle': env_cfg.get('col_aisle'),
                # 将订单生成/截止期配置合入生成器（深合并）
                'order_config': ORDER_CONFIG,
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

    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/results.csv', index=False)
    print("\nResults summary:\n", df)
    print("saved -> results/results.csv")


if __name__ == '__main__':
    main()
