# GPU 加速支持说明

## ✅ 确认：代码完全支持GPU加速

你的代码**已经正确配置了GPU支持**！当在有CUDA GPU的环境运行时，会自动使用GPU加速。

## GPU配置位置

### 1. run_experiments.py (主配置文件)

**第175行 - NL-HMARL GPU配置:**
```python
nl_cfg = dict(
    device='cuda' if torch.cuda.is_available() else 'cpu',  # 自动检测GPU
    ...
)
```

**第192行 - DQN GPU配置:**
```python
dqn_cfg = dict(
    device='cuda' if torch.cuda.is_available() else 'cpu',  # 自动检测GPU
    ...
)
```

### 2. src/exp/evaluate.py (训练执行)

**第88行 - DQN使用GPU:**
```python
_dev = str(dqn_cfg.get('device', 'cpu'))
model = train_flat_dqn(..., device=_dev)
```

**第112行 - NL-HMARL使用GPU:**
```python
_nl_dev = str(nl_cfg.get('device', 'cpu'))
model = train_nl_hmarl(..., device=_nl_dev)
```

**第140行 - NL-HMARL-AC使用GPU:**
```python
_nl_dev = str(nl_cfg.get('device', 'cpu'))
model = train_nl_hmarl_ac(..., device=_nl_dev)
```

## 本地MacBook vs Google Colab A100

### MacBook (当前环境)
- **GPU状态**: ❌ 无CUDA GPU
- **运行模式**: CPU-only (非常慢！)
- **预计时间**: 10-20+ 小时 (200k步，16环境)
- **已停止**: 已终止本地慢速运行

### Google Colab A100
- **GPU状态**: ✅ CUDA GPU可用
- **运行模式**: GPU加速 (快10-20倍！)
- **预计时间**: 4-6 小时 (200k步，16环境)
- **推荐**: **强烈推荐使用Colab运行完整实验**

## GPU自动检测流程

代码启动时会自动：

1. **检测CUDA可用性**
   ```python
   if torch.cuda.is_available():
       device = 'cuda'
   else:
       device = 'cpu'
   ```

2. **打印GPU信息** (已修正打印输出)
   ```python
   print(f"Devices -> NL: {nl_dev}, DQN: {dqn_dev}")
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")
       print(f"CUDA Version: {torch.version.cuda}")
   ```

3. **自动传递到所有训练方法**
   - NL-HMARL
   - NL-HMARL-AC
   - DQN-Guided
   - DQN-Pure
   - Softmax
   - Softmax-AC

## Stable-Baselines3 GPU支持

用于部分baseline方法的SB3库也会自动使用GPU：

```python
from stable_baselines3 import PPO

# SB3会自动检测并使用CUDA
# 如果torch.cuda.is_available() == True，则使用GPU
model = PPO("MlpPolicy", env, device="auto")  # auto = 自动检测
```

## 验证GPU正在使用

在Colab运行时，可以通过以下方式验证：

### 方法1: 检查nvidia-smi
```bash
!nvidia-smi
```
应该看到Python进程在使用GPU内存。

### 方法2: 检查PyTorch
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### 方法3: 监控GPU使用率
在Colab中运行：
```bash
!watch -n 1 nvidia-smi
```

## 性能对比估算

| 环境 | 设备 | 单次方法训练 | 9个方法总计 |
|------|------|--------------|-------------|
| MacBook | CPU | ~80-120分钟 | ~12-18小时 |
| Colab A100 | GPU | ~30-40分钟 | ~4-6小时 |

**加速倍数**: 约 **10-15倍**

## 内存需求

### GPU内存 (A100: 40GB)
- NL-HMARL: ~2-4GB
- DQN: ~1-2GB
- 16并行环境: ~4-6GB
- **总计**: ~8-12GB (A100完全足够)

### CPU内存
- 16并行环境: ~2-4GB
- 模型参数: ~500MB-1GB
- **总计**: ~3-5GB

## 使用建议

### ✅ 推荐做法
1. **在Colab A100上运行完整实验** (MODE=full)
   - 使用GPU加速
   - 4-6小时完成所有9个方法
   - 上传 `run_variant1_colab.ipynb` 到Colab

2. **本地只运行小规模测试** (MODE=test)
   - 验证代码正确性
   - 1000步，1个环境
   - 5-10分钟完成

### ❌ 不推荐
- 在MacBook上运行MODE=full (太慢，10-20+小时)

## 下一步操作

1. **上传项目到Google Drive**
   ```
   上传整个 LogitHMARL 文件夹
   ```

2. **打开 run_variant1_colab.ipynb**
   ```
   在Colab中打开笔记本
   Runtime → Change runtime type → A100 GPU
   ```

3. **运行实验**
   ```python
   # 笔记本会自动：
   # 1. 检测GPU（应显示A100）
   # 2. 安装依赖
   # 3. 设置MODE=full
   # 4. 运行所有9个方法
   # 5. 保存结果
   ```

4. **监控进度**
   ```bash
   # 在Colab cell中：
   !tail -f variant1_colab.log

   # 或者查看GPU使用：
   !nvidia-smi
   ```

## 常见问题

**Q: 如何确认代码在使用GPU？**
A: 查看输出日志，应该显示：
```
Devices -> NL: cuda, DQN: cuda
GPU: NVIDIA A100-SXM4-40GB
CUDA Version: 11.8
```

**Q: 如果Colab显示CPU而不是GPU？**
A:
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. GPU type → A100 (如果有Colab Pro)

**Q: 训练过程中GPU使用率低？**
A: 正常现象，因为：
- 环境仿真在CPU运行
- GPU主要用于神经网络前向/反向传播
- 平均GPU使用率可能只有30-50%

**Q: 可以在MacBook上使用MPS加速吗？**
A: 理论可行但需要额外配置。建议直接使用Colab A100。

## 总结

✅ **代码已完全支持GPU**
✅ **自动检测，无需手动配置**
✅ **在Colab A100上会自动使用GPU加速**
✅ **预期加速10-15倍**
✅ **强烈推荐使用Colab运行完整实验**
