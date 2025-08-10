"""
参数学习模块：基于历史数据学习Nested Logit模型参数
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class NestedLogitLearning:
    """
    使用最大似然估计学习Nested Logit模型参数
    """
    
    def __init__(self):
        self.params = None
        self.history = []
        
    def collect_data(self, order_features: Dict, picker_features: Dict, 
                     chosen_picker: str, outcome: Dict):
        """
        收集实际分配数据用于学习
        
        参数:
        - order_features: 订单特征（优先级、SKU数量等）
        - picker_features: 各拣货员特征（位置、负载等）
        - chosen_picker: 实际选择的拣货员
        - outcome: 结果指标（完成时间、距离等）
        """
        self.history.append({
            'order': order_features,
            'pickers': picker_features,
            'choice': chosen_picker,
            'outcome': outcome
        })
        
    def learn_parameters(self, batch_size: int = 1000):
        """
        基于历史数据学习模型参数
        
        使用最大似然估计优化参数
        """
        if len(self.history) < batch_size:
            logger.warning(f"数据量不足: {len(self.history)} < {batch_size}")
            return
            
        # 准备训练数据
        X, y = self._prepare_training_data(self.history[-batch_size:])
        
        # 初始参数
        init_params = {
            'w_distance': -0.5,
            'w_workload': -0.3,
            'w_priority': 1.0,
            'w_equipment': 2.0,
            'lambda': 0.7
        }
        
        # 最大似然估计
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=list(init_params.values()),
            args=(X, y),
            method='L-BFGS-B',
            bounds=[(-5, 5)] * 4 + [(0.1, 1.0)]  # lambda的范围是(0,1]
        )
        
        if result.success:
            self.params = {
                'w_distance': result.x[0],
                'w_workload': result.x[1],
                'w_priority': result.x[2],
                'w_equipment': result.x[3],
                'lambda': result.x[4]
            }
            logger.info(f"参数学习完成: {self.params}")
        else:
            logger.error(f"参数学习失败: {result.message}")
            
    def _prepare_training_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 这里需要将历史数据转换为特征矩阵
        # 简化示例
        features = []
        choices = []
        
        for record in data:
            # 提取特征
            order_feat = record['order']
            picker_feats = record['pickers']
            choice = record['choice']
            
            # 构建特征向量（简化）
            for picker_id, picker_feat in picker_feats.items():
                feat = [
                    picker_feat['distance'],
                    picker_feat['workload'],
                    order_feat['priority_score'],
                    picker_feat['has_equipment']
                ]
                features.append(feat)
                choices.append(1 if picker_id == choice else 0)
                
        return np.array(features), np.array(choices)
        
    def _negative_log_likelihood(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """计算负对数似然"""
        # 计算效用
        utilities = X @ params[:-1]  # 不包括lambda
        lambda_param = params[-1]
        
        # 计算选择概率（简化的Nested Logit）
        exp_utilities = np.exp(utilities / lambda_param)
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        # 负对数似然
        epsilon = 1e-10  # 避免log(0)
        nll = -np.sum(y * np.log(probabilities + epsilon))
        
        return nll
        
    def predict(self, order_features: Dict, picker_features: Dict) -> Dict[str, float]:
        """
        使用学习到的参数预测分配概率
        """
        if self.params is None:
            raise ValueError("模型参数尚未学习")
            
        # 使用学习到的参数计算概率
        probabilities = {}
        
        for picker_id, features in picker_features.items():
            utility = (
                self.params['w_distance'] * features['distance'] +
                self.params['w_workload'] * features['workload'] +
                self.params['w_priority'] * order_features['priority_score'] +
                self.params['w_equipment'] * features['has_equipment']
            )
            probabilities[picker_id] = np.exp(utility / self.params['lambda'])
            
        # 归一化
        total = sum(probabilities.values())
        for picker_id in probabilities:
            probabilities[picker_id] /= total
            
        return probabilities


# 使用示例
if __name__ == "__main__":
    learner = NestedLogitLearning()
    
    # 模拟收集数据
    for i in range(1000):
        order_feat = {
            'priority_score': np.random.choice([1, 2, 3]),
            'num_items': np.random.randint(1, 10)
        }
        
        picker_feat = {}
        for j in range(6):
            picker_feat[f'P{j}'] = {
                'distance': np.random.uniform(10, 100),
                'workload': np.random.randint(0, 20),
                'has_equipment': np.random.choice([0, 1])
            }
            
        # 模拟选择
        chosen = np.random.choice(list(picker_feat.keys()))
        
        outcome = {
            'completion_time': np.random.uniform(60, 300),
            'travel_distance': np.random.uniform(50, 200)
        }
        
        learner.collect_data(order_feat, picker_feat, chosen, outcome)
    
    # 学习参数
    learner.learn_parameters()
    
    # 预测
    test_order = {'priority_score': 2}
    test_pickers = {
        'P0': {'distance': 50, 'workload': 5, 'has_equipment': 1},
        'P1': {'distance': 30, 'workload': 10, 'has_equipment': 0}
    }
    
    probs = learner.predict(test_order, test_pickers)
    print(f"预测概率: {probs}")