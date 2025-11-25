"""
多进程并行环境包装器
使用Python multiprocessing实现环境并行化，提升CPU利用率和GPU利用率
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import numpy as np
import torch
from typing import Dict, List, Any, Callable
import time
import traceback


def env_worker(env_id: int, env_fn: Callable, cmd_queue: Queue, result_queue: Queue, stop_event: Event):
    """环境工作进程"""
    try:
        # 创建环境
        env = env_fn()
        result_queue.put(('ready', env_id, None))

        while not stop_event.is_set():
            try:
                # 等待命令（带超时）
                if not cmd_queue.empty():
                    cmd, data = cmd_queue.get(timeout=0.1)

                    if cmd == 'step':
                        # 执行环境step
                        actions = data
                        obs, reward, done, info = env.step(actions)
                        result_queue.put(('step_result', env_id, (obs, reward, done, info)))

                    elif cmd == 'reset':
                        # 重置环境
                        obs = env.reset()
                        result_queue.put(('reset_result', env_id, obs))

                    elif cmd == 'get_state':
                        # 获取环境状态
                        result_queue.put(('state_result', env_id, env))

                    elif cmd == 'stop':
                        break

                else:
                    time.sleep(0.001)  # 避免忙等待

            except Exception as e:
                result_queue.put(('error', env_id, str(e)))
                traceback.print_exc()

    except Exception as e:
        result_queue.put(('init_error', env_id, str(e)))
    finally:
        result_queue.put(('stopped', env_id, None))


class ParallelEnvPool:
    """并行环境池"""

    def __init__(self, env_fn_list: List[Callable], num_workers: int = None):
        """
        Args:
            env_fn_list: 环境构造函数列表
            num_workers: 工作进程数（默认=CPU核心数）
        """
        self.num_envs = len(env_fn_list)
        self.num_workers = num_workers or min(mp.cpu_count(), self.num_envs)

        # 创建队列和事件
        self.cmd_queues = [Queue() for _ in range(self.num_workers)]
        self.result_queue = Queue()
        self.stop_event = Event()

        # 启动工作进程
        self.processes = []
        for i in range(self.num_workers):
            # 计算此进程负责的环境
            env_indices = list(range(i, self.num_envs, self.num_workers))

            for env_idx in env_indices:
                p = Process(
                    target=env_worker,
                    args=(env_idx, env_fn_list[env_idx], self.cmd_queues[i % self.num_workers],
                          self.result_queue, self.stop_event),
                    daemon=True
                )
                p.start()
                self.processes.append(p)

        # 等待所有环境准备完毕
        ready_count = 0
        while ready_count < self.num_envs:
            msg_type, env_id, data = self.result_queue.get(timeout=10.0)
            if msg_type == 'ready':
                ready_count += 1
            elif msg_type == 'init_error':
                raise RuntimeError(f"环境 {env_id} 初始化失败: {data}")

        print(f"✅ 启动了 {self.num_envs} 个环境在 {self.num_workers} 个进程中")

    def step_all(self, actions_dict: Dict[int, Any]):
        """
        并行执行所有环境的step

        Args:
            actions_dict: {env_id: actions}

        Returns:
            results_dict: {env_id: (obs, reward, done, info)}
        """
        # 发送step命令到所有环境
        for env_id, actions in actions_dict.items():
            worker_id = env_id % self.num_workers
            self.cmd_queues[worker_id].put(('step', actions))

        # 收集结果
        results = {}
        for _ in range(len(actions_dict)):
            msg_type, env_id, data = self.result_queue.get(timeout=30.0)
            if msg_type == 'step_result':
                results[env_id] = data
            elif msg_type == 'error':
                print(f"⚠️ 环境 {env_id} 错误: {data}")

        return results

    def close(self):
        """关闭所有环境"""
        self.stop_event.set()

        # 等待所有进程结束
        for p in self.processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()

        # 清空队列
        for q in self.cmd_queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break

        print("✅ 所有并行环境已关闭")


class BatchedEnvCollector:
    """批量环境数据收集器（优化GPU利用率）"""

    def __init__(self, envs: List[Any], collect_steps_per_update: int = 16):
        """
        Args:
            envs: 环境列表（可以是普通列表或ParallelEnvPool）
            collect_steps_per_update: 每次GPU更新前收集多少步
        """
        self.envs = envs
        self.num_envs = len(envs)
        self.collect_steps_per_update = collect_steps_per_update

        # 经验缓冲区
        self.buffer = []

    def collect_batch(self, model: torch.nn.Module, step_fn: Callable) -> List[Dict]:
        """
        收集一批经验

        Args:
            model: 神经网络模型
            step_fn: 单步执行函数 step_fn(env, model) -> experience_dict

        Returns:
            experiences: 经验列表
        """
        experiences = []

        # 串行收集（但环境内部可能并行）
        for _ in range(self.collect_steps_per_update):
            for env in self.envs:
                exp = step_fn(env, model)
                if exp:
                    experiences.append(exp)

        return experiences

    def collect_batch_parallel(self, model: torch.nn.Module, batch_step_fn: Callable) -> List[Dict]:
        """
        并行收集一批经验（所有环境同时step）

        Args:
            model: 神经网络模型
            batch_step_fn: 批量执行函数 batch_step_fn(envs, model) -> List[experience_dict]

        Returns:
            experiences: 经验列表
        """
        experiences = []

        for _ in range(self.collect_steps_per_update):
            batch_exp = batch_step_fn(self.envs, model)
            experiences.extend(batch_exp)

        return experiences


def demo_usage():
    """使用示例"""
    from exp.env_factory import create_test_env

    # 创建环境构造函数列表
    n_envs = 16
    env_fns = []
    for i in range(n_envs):
        def make_env(seed=i):
            return lambda: create_test_env(20, 20, 8, 20, 4, 30, 5)
        env_fns.append(make_env())

    # 创建并行环境池
    env_pool = ParallelEnvPool(env_fns, num_workers=4)

    try:
        # 模拟训练循环
        for step in range(10):
            # 准备动作（示例）
            actions_dict = {i: {} for i in range(n_envs)}

            # 并行执行
            results = env_pool.step_all(actions_dict)

            print(f"步骤 {step}: 收集了 {len(results)} 个环境的结果")

    finally:
        env_pool.close()


if __name__ == '__main__':
    demo_usage()
