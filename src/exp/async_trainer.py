"""
异步并行训练框架
实现CPU环境仿真和GPU神经网络训练的并行执行，提升GPU利用率
"""

import threading
import queue
import time
import copy
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Callable
from collections import deque


class ExperienceBuffer:
    """线程安全的经验队列"""

    def __init__(self, maxsize: int = 50000):
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        self._total_added = 0
        self._total_sampled = 0

    def add(self, experience: Dict[str, Any]):
        """添加经验（阻塞，直到队列有空间）"""
        self.queue.put(experience, block=True)
        with self.lock:
            self._total_added += 1

    def add_batch(self, experiences: List[Dict[str, Any]]):
        """批量添加经验"""
        for exp in experiences:
            self.add(exp)

    def sample(self, batch_size: int, timeout: float = 5.0) -> Optional[List[Dict[str, Any]]]:
        """采样一批经验（带超时）"""
        batch = []
        deadline = time.time() + timeout

        while len(batch) < batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                # 超时：返回已收集的经验（可能小于batch_size）
                break

            try:
                exp = self.queue.get(timeout=min(0.1, remaining))
                batch.append(exp)
                with self.lock:
                    self._total_sampled += 1
            except queue.Empty:
                # 队列暂时为空，继续等待
                continue

        return batch if len(batch) > 0 else None

    def size(self) -> int:
        """当前队列大小"""
        return self.queue.qsize()

    def stats(self) -> Dict[str, int]:
        """统计信息"""
        with self.lock:
            return {
                'queue_size': self.queue.qsize(),
                'total_added': self._total_added,
                'total_sampled': self._total_sampled,
            }


class SharedModel:
    """线程安全的模型共享器"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.lock = threading.Lock()
        self.version = 0  # 模型版本号

    def get_model_copy(self) -> torch.nn.Module:
        """获取模型的副本（用于CPU线程推理）"""
        with self.lock:
            # 深拷贝模型参数（避免梯度计算）
            model_copy = copy.deepcopy(self.model)
            model_copy.eval()  # 设置为评估模式
            return model_copy

    def update_model(self, new_state_dict: Dict):
        """更新共享模型（GPU训练后）"""
        with self.lock:
            self.model.load_state_dict(new_state_dict)
            self.version += 1

    def get_version(self) -> int:
        """获取模型版本号"""
        with self.lock:
            return self.version


class AsyncTrainer:
    """异步训练控制器"""

    def __init__(
        self,
        model: torch.nn.Module,
        envs: List[Any],
        collect_fn: Callable,
        train_fn: Callable,
        batch_size: int = 256,
        buffer_size: int = 50000,
        update_interval: int = 10,  # 每训练N步更新一次共享模型
        num_collect_threads: int = 1,
    ):
        """
        Args:
            model: 神经网络模型
            envs: 环境列表
            collect_fn: 收集函数 collect_fn(env, model) -> List[experience]
            train_fn: 训练函数 train_fn(model, batch) -> (loss, metrics)
            batch_size: 训练批量大小
            buffer_size: 经验队列最大容量
            update_interval: 模型更新间隔（步数）
            num_collect_threads: CPU收集线程数量
        """
        self.shared_model = SharedModel(model)
        self.experience_buffer = ExperienceBuffer(maxsize=buffer_size)
        self.envs = envs
        self.collect_fn = collect_fn
        self.train_fn = train_fn
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.num_collect_threads = num_collect_threads

        # 控制标志
        self.stop_flag = threading.Event()
        self.threads = []

        # 统计信息
        self.stats_lock = threading.Lock()
        self.stats = {
            'collect_steps': 0,
            'train_steps': 0,
            'total_experiences': 0,
            'avg_loss': 0.0,
            'gpu_utilization': 0.0,
        }

    def _collect_worker(self, thread_id: int, env_indices: List[int]):
        """CPU收集线程工作函数"""
        print(f"[Collect-{thread_id}] 启动，负责环境 {env_indices}")
        local_model = self.shared_model.get_model_copy()
        model_version = 0
        steps = 0

        while not self.stop_flag.is_set():
            # 定期更新本地模型
            if steps % self.update_interval == 0:
                current_version = self.shared_model.get_version()
                if current_version > model_version:
                    local_model = self.shared_model.get_model_copy()
                    model_version = current_version

            # 收集经验
            for env_idx in env_indices:
                if self.stop_flag.is_set():
                    break

                env = self.envs[env_idx]
                try:
                    # 调用用户提供的收集函数
                    experiences = self.collect_fn(env, local_model)

                    # 添加到队列
                    if experiences:
                        self.experience_buffer.add_batch(experiences)

                        with self.stats_lock:
                            self.stats['collect_steps'] += 1
                            self.stats['total_experiences'] += len(experiences)

                except Exception as e:
                    print(f"[Collect-{thread_id}] 错误: {e}")
                    continue

            steps += 1

        print(f"[Collect-{thread_id}] 停止")

    def _train_worker(self):
        """GPU训练线程工作函数"""
        print("[Train] GPU训练线程启动")
        model = self.shared_model.model
        model.train()
        steps = 0
        loss_ema = None

        while not self.stop_flag.is_set():
            # 从队列采样经验
            batch = self.experience_buffer.sample(self.batch_size, timeout=2.0)

            if batch is None or len(batch) < self.batch_size // 2:
                # 经验不足，等待收集
                time.sleep(0.1)
                continue

            try:
                # 调用用户提供的训练函数
                loss, metrics = self.train_fn(model, batch)

                # 更新统计
                if loss_ema is None:
                    loss_ema = loss
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss

                with self.stats_lock:
                    self.stats['train_steps'] += 1
                    self.stats['avg_loss'] = loss_ema

                # 定期更新共享模型
                if steps % self.update_interval == 0:
                    self.shared_model.update_model(model.state_dict())

                steps += 1

            except Exception as e:
                print(f"[Train] 错误: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 最后一次更新
        self.shared_model.update_model(model.state_dict())
        print("[Train] GPU训练线程停止")

    def start(self):
        """启动异步训练"""
        print("="*70)
        print("🚀 启动异步并行训练")
        print(f"  - 环境数量: {len(self.envs)}")
        print(f"  - 收集线程: {self.num_collect_threads}")
        print(f"  - 批量大小: {self.batch_size}")
        print(f"  - 缓冲区大小: {self.experience_buffer.queue.maxsize}")
        print("="*70)

        self.stop_flag.clear()

        # 启动GPU训练线程
        train_thread = threading.Thread(target=self._train_worker, daemon=True)
        train_thread.start()
        self.threads.append(train_thread)

        # 启动CPU收集线程
        envs_per_thread = len(self.envs) // self.num_collect_threads
        for i in range(self.num_collect_threads):
            start_idx = i * envs_per_thread
            if i == self.num_collect_threads - 1:
                # 最后一个线程处理剩余所有环境
                env_indices = list(range(start_idx, len(self.envs)))
            else:
                env_indices = list(range(start_idx, start_idx + envs_per_thread))

            thread = threading.Thread(
                target=self._collect_worker,
                args=(i, env_indices),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)

        print(f"✅ 已启动 {len(self.threads)} 个线程（1个GPU训练 + {self.num_collect_threads}个CPU收集）")

    def stop(self):
        """停止异步训练"""
        print("\n⏸️  停止异步训练...")
        self.stop_flag.set()

        # 等待所有线程结束
        for thread in self.threads:
            thread.join(timeout=5.0)

        self.threads.clear()
        print("✅ 异步训练已停止")

    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        with self.stats_lock:
            stats = self.stats.copy()

        # 添加队列统计
        stats.update(self.experience_buffer.stats())
        return stats

    def train(self, total_steps: int, log_interval: int = 100):
        """运行异步训练"""
        self.start()

        try:
            start_time = time.time()
            last_log_time = start_time

            while True:
                time.sleep(1.0)  # 每秒检查一次

                stats = self.get_stats()
                train_steps = stats['train_steps']

                # 打印进度
                if train_steps % log_interval == 0 and train_steps > 0:
                    elapsed = time.time() - last_log_time
                    steps_per_sec = log_interval / elapsed if elapsed > 0 else 0

                    print(f"步数: {train_steps}/{total_steps} | "
                          f"损失: {stats['avg_loss']:.4f} | "
                          f"队列: {stats['queue_size']} | "
                          f"速度: {steps_per_sec:.1f} steps/s")

                    last_log_time = time.time()

                # 检查是否完成
                if train_steps >= total_steps:
                    break

        finally:
            self.stop()

        total_time = time.time() - start_time
        print(f"\n✅ 训练完成！总用时: {total_time:.1f}秒")
        print(f"   平均速度: {total_steps/total_time:.1f} steps/s")

        return self.shared_model.model
