"""
NL-HMARL-AC 异步训练函数
替代原有的串行训练，使用异步并行框架
"""

import torch
from baselines.nl_hmarl import NLHMARL
from exp.async_trainer import AsyncTrainer
from exp.async_nl_hmarl import NLHMARLAsyncAdapter


def train_nl_hmarl_ac_async(
    *,
    env_ctor,
    env_config: dict,
    training_steps: int = 5000,
    hidden_dim: int = 256,
    lr_manager: float = 1e-3,
    lr_workers: float = 1e-3,
    max_tasks: int = 20,
    gamma: float = 0.99,
    entropy_coef_manager: float = 0.01,
    entropy_coef_workers: float = 0.01,
    # NL manager structure
    n_nests: int = 4,
    learn_eta: bool = False,
    eta_init: float = 1.0,
    device: str = 'cuda',
    speed_function=None,
    log_metrics: bool = True,
    log_every: int = 100,
    metrics_dir: str = 'results/train_metrics',
    metrics_tag: str = 'NL-HMARL-AC-Async',
    # 异步训练参数
    n_envs: int = 16,
    batch_size: int = 256,
    buffer_size: int = 50000,
    update_interval: int = 10,
    num_collect_threads: int = 4,
):
    """
    NL-HMARL-AC 异步训练（CPU收集 + GPU训练并行）

    Args:
        env_ctor: 环境构造函数
        env_config: 环境配置
        training_steps: 训练步数
        ... (其他参数同train_nl_hmarl_ac)
        n_envs: 并行环境数量
        batch_size: 训练批量大小
        buffer_size: 经验队列容量
        update_interval: 模型更新间隔
        num_collect_threads: CPU收集线程数

    Returns:
        model: 训练好的模型
    """
    import numpy as np
    from env.dynamic_warehouse_env import PickerType

    print("="*70)
    print("🚀 启动NL-HMARL-AC异步训练")
    print(f"  - 训练步数: {training_steps}")
    print(f"  - 并行环境: {n_envs}")
    print(f"  - 收集线程: {num_collect_threads}")
    print(f"  - 批量大小: {batch_size}")
    print(f"  - 设备: {device}")
    print("="*70)

    # 创建环境列表
    if speed_function is None:
        def speed_function(e):
            return {p.id: float(getattr(p, 'speed', 1.0)) for p in e.pickers}

    envs = []
    for i in range(n_envs):
        env = env_ctor(dict(env_config))
        env.set_speed_function(speed_function)
        env.reset()
        envs.append(env)

    # 获取维度信息
    n_agents = len(envs[0].pickers)
    env0 = envs[0]
    from exp.obs import get_global_state, get_task_features
    state_dim = len(get_global_state(env0))
    task_feat_dim = get_task_features(env0, max_tasks=max_tasks, pending_only=True).shape[1]

    # Worker观测维度（局部+全局）
    from exp.obs import get_agent_observation
    sample_obs = get_agent_observation(env0, env0.pickers[0], include_global=True)
    worker_obs_dim = len(sample_obs)

    print(f"\n📐 模型维度:")
    print(f"  - 全局状态维度: {state_dim}")
    print(f"  - 任务特征维度: {task_feat_dim}")
    print(f"  - Worker观测维度: {worker_obs_dim}")
    print(f"  - Agent数量: {n_agents}")
    print(f"  - Nest数量: {n_nests}")

    # 创建模型
    model = NLHMARL(
        state_dim=state_dim,
        n_tasks=max_tasks,  # NLH MARL使用n_tasks参数
        n_nests=n_nests,
        worker_obs_dim=worker_obs_dim,
        worker_action_dim=7,  # UP/DOWN/LEFT/RIGHT/IDLE/PICK/DROP
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        device=device,
        learn_eta=learn_eta,
        eta_init=eta_init,
    )

    # 创建优化器
    opt_manager = torch.optim.Adam(
        list(model.manager.parameters()) + list(model.value_net.parameters()),
        lr=lr_manager
    )
    opt_workers = torch.optim.Adam(model.workers.parameters(), lr=lr_workers)

    # 创建异步适配器
    adapter = NLHMARLAsyncAdapter(
        model=model,
        n_agents=n_agents,
        max_tasks=max_tasks,
        gamma=gamma,
        entropy_coef_manager=entropy_coef_manager,
        entropy_coef_worker=entropy_coef_workers,
        device=device,
    )
    adapter.total_training_steps = training_steps

    # 创建collect_fn和train_fn
    collect_fn = adapter.create_collect_fn()
    train_fn = adapter.create_train_fn(opt_manager, opt_workers)

    # 创建异步训练器
    async_trainer = AsyncTrainer(
        model=model,
        envs=envs,
        collect_fn=collect_fn,
        train_fn=train_fn,
        batch_size=batch_size,
        buffer_size=buffer_size,
        update_interval=update_interval,
        num_collect_threads=num_collect_threads,
    )

    # 运行异步训练
    print("\n🏃 开始异步训练...\n")
    try:
        trained_model = async_trainer.train(
            total_steps=training_steps,
            log_interval=log_every
        )
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        async_trainer.stop()
        trained_model = model

    # 保存指标（简化版，异步训练的指标收集需要额外处理）
    if log_metrics:
        import os
        import pandas as pd

        out_dir = os.path.join(metrics_dir, metrics_tag)
        os.makedirs(out_dir, exist_ok=True)

        stats = async_trainer.get_stats()
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(os.path.join(out_dir, 'final_stats.csv'), index=False)
        print(f"\n📊 训练统计已保存到 {out_dir}/final_stats.csv")

    print("\n✅ 异步训练完成！")
    return trained_model
