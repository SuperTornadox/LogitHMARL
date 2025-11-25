# LogitHMARL - Nested-Logit Hierarchical Multi-Agent Reinforcement Learning

动态仓库环境中的分层多智能体强化学习研究项目。

## 项目概述

本项目实现并比较了多种基于强化学习的仓库任务分配方法，重点研究Nested-Logit HMARL在大规模动态环境中的性能和可扩展性。

## 主要方法

- **NL-HMARL**: Nested-Logit分层多智能体强化学习（主要贡献）
- **Softmax-HMARL**: 基于Softmax的分层MARL
- **Flat MARL DQN**: 扁平化DQN基线
- **Rule-based**: 规则基线方法（Optimal, Return, S-Shape等）

## 快速开始

### 环境要求
```bash
pip install -r requirements.txt
```

### 运行实验
```bash
# 运行所有方法的完整实验
python run_experiments.py

# 只运行特定方法
python run_experiments.py --only NL-HMARL,Softmax
```

### Colab运行
参见 `run_variant1_colab.ipynb` 或查看 `docs/COLAB_INSTRUCTIONS.md`

## 项目结构

```
LogitHMARL/
├── src/                    # 核心源代码
│   ├── baselines/         # 基线方法实现
│   ├── env/               # 动态仓库环境
│   ├── exp/               # 训练和评估框架
│   ├── models/            # 神经网络模型
│   └── utils/             # 工具函数
├── results/                # 实验结果
│   ├── results.csv        # 结果汇总
│   └── train_metrics/     # 训练指标和模型
├── docs/                   # 技术文档
├── paper/                  # 论文相关
└── archive_old_results/    # 历史实验归档
```

详细结构说明见 `PROJECT_STRUCTURE.md`

## 实验结果

最新实验结果（12×12环境，10k训练步数）：

| Method | Tasks | Total Value | Performance |
|--------|-------|-------------|-------------|
| Optimal (上界) | 157 | 25,764 | 100% |
| NL-HMARL | 131 | 10,632 | 41.3% |
| Softmax | 131 | 10,754 | 41.7% |

完整结果见 `results/results.csv`

## 文档

- `docs/COMPLEXITY_THEORY_ANALYSIS.md` - 复杂度理论分析
- `docs/OPTIMAL_COMPLEXITY_ANALYSIS.md` - 最优算法复杂度分析
- `docs/SCALABILITY_COMPARISON.md` - 可扩展性对比
- `docs/GPU_SUPPORT.md` - GPU训练支持说明
- `docs/COLAB_INSTRUCTIONS.md` - Colab使用指南

## 当前研究状态

正在进行3轮独立训练以获得统计显著的结果：
- ✅ Round 1完成
- 🔄 Round 2进行中
- ⏳ Round 3待开始
- ⏳ 24×24迁移学习测试
- ⏳ 统计分析

## 引用

如果本项目对您的研究有帮助，请引用：
```
[待补充]
```

## 许可

[待补充]

## 联系方式

[待补充]
