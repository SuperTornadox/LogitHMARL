"""
测试异步训练功能
小规模快速验证异步训练框架是否正常工作
"""

import os
import sys

# 添加src目录到路径
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

os.environ['MODE'] = 'test'  # 使用测试模式

import torch
from exp.train_nl_hmarl_ac_async import train_nl_hmarl_ac_async
from exp.env_factory import create_test_env

print("="*70)
print("🧪 异步训练功能测试")
print("="*70)

# 测试配置（极小规模）
test_config = {
    'width': 20,
    'height': 20,
    'n_pickers': 4,  # 减少agent数量
    'n_shelves': 20,
    'n_stations': 4,
    'n_charging_pads': 1,
    'levels_per_shelf': 3,
    'time_step': 2.0,
    'order_config': {
        'base_rate': 500,  # 降低订单率
        'peak_hours': [(0, 6)],
        'peak_multiplier': 1.5,
        'off_peak_multiplier': 0.8,
        'simulation_hours': 1.0,  # 短时间
        'use_pregeneration': True,
    },
}

def env_ctor(cfg):
    """环境构造函数"""
    return create_test_env(
        cfg['width'], cfg['height'],
        cfg['n_pickers'], cfg['n_shelves'], cfg['n_stations'],
        cfg['order_config']['base_rate'], 5
    )

# 异步训练参数
training_params = {
    'env_ctor': env_ctor,
    'env_config': test_config,
    'training_steps': 100,  # 极少步数，仅测试功能
    'hidden_dim': 64,  # 小模型
    'lr_manager': 3e-4,
    'lr_workers': 3e-4,
    'max_tasks': 10,  # 减少任务数
    'gamma': 0.99,
    'entropy_coef_manager': 0.01,
    'entropy_coef_workers': 0.01,
    'n_nests': 4,
    'learn_eta': False,
    'eta_init': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'log_metrics': True,
    'log_every': 20,
    'metrics_dir': 'results/test_async',
    'metrics_tag': 'AsyncTest',
    # 异步参数
    'n_envs': 4,  # 4个并行环境
    'batch_size': 32,  # 小批量
    'buffer_size': 1000,  # 小缓冲区
    'update_interval': 5,  # 频繁更新
    'num_collect_threads': 2,  # 2个收集线程
}

print(f"\n📋 测试配置:")
print(f"  - 训练步数: {training_params['training_steps']}")
print(f"  - 并行环境: {training_params['n_envs']}")
print(f"  - 收集线程: {training_params['num_collect_threads']}")
print(f"  - 批量大小: {training_params['batch_size']}")
print(f"  - 设备: {training_params['device']}")
print()

# 运行异步训练
try:
    print("🚀 启动异步训练测试...")
    model = train_nl_hmarl_ac_async(**training_params)
    print("\n✅ 异步训练测试成功！")
    print(f"   模型类型: {type(model).__name__}")
    print(f"   模型设备: {next(model.parameters()).device}")

except Exception as e:
    print(f"\n❌ 异步训练测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("测试完成")
print("="*70)
