# NL-HMARL: Nested-Logit Hierarchical Multi-Agent Reinforcement Learning

A framework for real-time task allocation in robotic warehouses using Nested-Logit based hierarchical multi-agent reinforcement learning.

## Overview

This project implements and compares multiple reinforcement learning methods for warehouse task allocation, with a focus on the Nested-Logit HMARL (NL-HMARL) approach that relaxes the Independence of Irrelevant Alternatives (IIA) assumption inherent in standard softmax policies.

## Methods

- **NL-HMARL**: Nested-Logit Hierarchical MARL (main contribution)
- **Softmax-HMARL**: Standard softmax-based Hierarchical MARL
- **Rule-based**: S-Shape, Return, and Greedy-Nearest (Optimal) baselines

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### Run Experiments
```bash
python run_experiments.py
```

## Project Structure

```
LogitHMARL/
├── src/
│   ├── baselines/         # Algorithm implementations
│   │   ├── nl_hmarl.py    # NL-HMARL
│   │   ├── softmax_hmarl.py
│   │   └── rule_based.py
│   ├── env/               # Warehouse simulation
│   ├── exp/               # Training and evaluation
│   └── models/            # Neural network models
├── results/               # Experiment results
├── paper/                 # Paper (English)
├── paper_cn/              # Paper (Chinese)
├── run_experiments.py     # Main entry point
└── requirements.txt
```

## Key Results

Across 6 configurations (3 difficulty levels × 2 scales), NL-HMARL outperforms Softmax-HMARL in 5/6 cases (83.3% win rate):

| Config | Scale | NL-HMARL Advantage |
|--------|-------|-------------------|
| Config1-Easy | 24×24 | +8.5% |
| Config2-Medium | 12×12 | +8.7% |
| Config2-Medium | 24×24 | +34.2% |
| Config3-Hard | 12×12 | +6.3% |
| Config3-Hard | 24×24 | +52.6% |

NL-HMARL shows stronger advantages in complex environments with uniform task utility distributions (where IIA problems are more severe).

## Citation

If you find this work useful, please cite:

```bibtex
@article{he2024nlhmarl,
  title={Nested-Logit Hierarchical Multi-Agent Reinforcement Learning for Real-Time Task Allocation in Robotic Warehouses},
  author={He, Xuchen and Madisetti, Vijay K.},
  year={2024}
}
```

## License

MIT License
