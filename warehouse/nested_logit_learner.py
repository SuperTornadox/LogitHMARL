"""
带学习能力的Nested Logit模型
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NestedLogitLearner:
    """
    可学习参数的Nested Logit模型
    """
    
    def __init__(self, num_pickers: int = 6):
        self.num_pickers = num_pickers
        self.picker_ids = [f"P{i}" for i in range(num_pickers)]
        
        # 初始参数
        self.params = {
            'w_distance': -0.5,
            'w_workload': -0.3,
            'w_priority': 1.0,
            'w_equipment': 2.0,
            'w_idle_time': 0.5,
            'lambda_urgent': 0.7,
            'lambda_regular': 0.5,
            'nest_constant_forklift': 0.5
        }
        
        # 特征标准化器
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """准备训练特征"""
        features = []
        choices = []
        nest_info = []
        
        for idx, row in df.iterrows():
            # 订单特征
            priority_score = row['priority_score']
            need_forklift = row['need_forklift']
            
            # 对每个拣货员构建特征
            for picker_id in self.picker_ids:
                prefix = f"picker_{picker_id}_"
                
                # 拣货员特征
                feat = [
                    row[prefix + 'distance'],
                    row[prefix + 'workload'],
                    priority_score,
                    row[prefix + 'equipment_match'],
                    row[prefix + 'idle_time'] / 60.0  # 转换为分钟
                ]
                
                features.append(feat)
                choices.append(row[prefix + 'chosen'])
                
                # 确定nest
                if priority_score >= 2:  # high or normal priority
                    if need_forklift and row[prefix + 'has_forklift']:
                        nest = 'urgent_forklift'
                    else:
                        nest = 'urgent_normal'
                else:
                    if need_forklift and row[prefix + 'has_forklift']:
                        nest = 'regular_forklift'
                    else:
                        nest = 'regular_normal'
                        
                nest_info.append({
                    'order_idx': idx,
                    'picker': picker_id,
                    'nest': nest
                })
                
        features = np.array(features)
        choices = np.array(choices)
        
        # 标准化特征
        if not self.is_fitted:
            features = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            features = self.scaler.transform(features)
            
        return features, choices, nest_info
        
    def compute_utilities(self, features: np.ndarray, params: np.ndarray) -> np.ndarray:
        """计算效用值"""
        weights = params[:5]  # 前5个是特征权重
        utilities = features @ weights
        return utilities
        
    def nested_logit_probabilities(self, utilities: np.ndarray, nest_info: List[Dict], 
                                  params: np.ndarray) -> np.ndarray:
        """计算Nested Logit概率"""
        lambda_urgent = params[5]
        lambda_regular = params[6]
        nest_const_forklift = params[7]
        
        probabilities = np.zeros_like(utilities)
        
        # 按订单分组计算
        order_indices = sorted(set([n['order_idx'] for n in nest_info]))
        
        for order_idx in order_indices:
            # 获取该订单的所有选择
            order_mask = [i for i, n in enumerate(nest_info) if n['order_idx'] == order_idx]
            order_utilities = utilities[order_mask]
            order_nests = [nest_info[i]['nest'] for i in order_mask]
            
            # 计算nest层概率
            nest_values = {}
            for nest in set(order_nests):
                nest_mask = [i for i, n in enumerate(order_nests) if n == nest]
                
                # 选择合适的lambda
                if 'urgent' in nest:
                    lambda_val = lambda_urgent
                else:
                    lambda_val = lambda_regular
                    
                # 添加nest常数
                if 'forklift' in nest:
                    nest_utilities = order_utilities[nest_mask] + nest_const_forklift
                else:
                    nest_utilities = order_utilities[nest_mask]
                    
                # 计算inclusive value
                iv = lambda_val * np.log(np.sum(np.exp(nest_utilities / lambda_val)))
                nest_values[nest] = iv
                
            # 计算nest选择概率
            nest_probs = {}
            total_nest_exp = sum(np.exp(v) for v in nest_values.values())
            for nest, value in nest_values.items():
                nest_probs[nest] = np.exp(value) / total_nest_exp
                
            # 计算最终概率
            for i, (util, nest) in enumerate(zip(order_utilities, order_nests)):
                lambda_val = lambda_urgent if 'urgent' in nest else lambda_regular
                nest_mask = [j for j, n in enumerate(order_nests) if n == nest]
                nest_utils = order_utilities[nest_mask]
                
                # 在nest内的条件概率
                if 'forklift' in nest:
                    nest_utils = nest_utils + nest_const_forklift
                    util = util + nest_const_forklift
                    
                exp_utils = np.exp(nest_utils / lambda_val)
                cond_prob = np.exp(util / lambda_val) / np.sum(exp_utils)
                
                # 总概率
                probabilities[order_mask[i]] = nest_probs[nest] * cond_prob
                
        return probabilities
        
    def negative_log_likelihood(self, params: np.ndarray, features: np.ndarray, 
                               choices: np.ndarray, nest_info: List[Dict]) -> float:
        """负对数似然函数"""
        # 计算效用
        utilities = self.compute_utilities(features, params)
        
        # 计算概率
        probabilities = self.nested_logit_probabilities(utilities, nest_info, params)
        
        # 避免log(0)
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        
        # 负对数似然
        nll = -np.sum(choices * np.log(probabilities))
        
        # 添加正则化
        reg_term = 0.01 * np.sum(params[:5]**2)  # L2正则化
        
        return nll + reg_term
        
    def fit(self, df: pd.DataFrame, max_iter: int = 1000):
        """训练模型"""
        logger.info("Preparing features...")
        features, choices, nest_info = self.prepare_features(df)
        
        # 初始参数
        x0 = list(self.params.values())
        
        # 参数边界
        bounds = [
            (-5, 0),    # w_distance (负值)
            (-5, 0),    # w_workload (负值)
            (0, 5),     # w_priority (正值)
            (0, 5),     # w_equipment (正值)
            (0, 5),     # w_idle_time (正值)
            (0.1, 1.0), # lambda_urgent
            (0.1, 1.0), # lambda_regular
            (-2, 2)     # nest_constant_forklift
        ]
        
        logger.info("Optimizing parameters...")
        result = minimize(
            fun=self.negative_log_likelihood,
            x0=x0,
            args=(features, choices, nest_info),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': True}
        )
        
        if result.success:
            # 更新参数
            self.params = {
                'w_distance': result.x[0],
                'w_workload': result.x[1],
                'w_priority': result.x[2],
                'w_equipment': result.x[3],
                'w_idle_time': result.x[4],
                'lambda_urgent': result.x[5],
                'lambda_regular': result.x[6],
                'nest_constant_forklift': result.x[7]
            }
            logger.info(f"Optimization successful! Final loss: {result.fun:.4f}")
            logger.info(f"Learned parameters: {self.params}")
        else:
            logger.error(f"Optimization failed: {result.message}")
            
    def predict_probabilities(self, order_features: Dict, picker_features: Dict[str, Dict]) -> Dict[str, float]:
        """预测分配概率"""
        features = []
        nest_info = []
        
        for picker_id, pf in picker_features.items():
            feat = [
                pf['distance'],
                pf['workload'],
                order_features['priority_score'],
                pf['equipment_match'],
                pf['idle_time'] / 60.0
            ]
            features.append(feat)
            
            # 确定nest
            if order_features['priority_score'] >= 2:
                if order_features['need_forklift'] and pf['has_forklift']:
                    nest = 'urgent_forklift'
                else:
                    nest = 'urgent_normal'
            else:
                if order_features['need_forklift'] and pf['has_forklift']:
                    nest = 'regular_forklift'
                else:
                    nest = 'regular_normal'
                    
            nest_info.append({
                'order_idx': 0,
                'picker': picker_id,
                'nest': nest
            })
            
        features = np.array(features)
        features = self.scaler.transform(features)
        
        # 计算效用和概率
        params = list(self.params.values())
        utilities = self.compute_utilities(features, np.array(params))
        probabilities = self.nested_logit_probabilities(utilities, nest_info, np.array(params))
        
        # 返回字典形式
        return {picker_id: prob for picker_id, prob in zip(picker_features.keys(), probabilities)}
        
    def save_model(self, filename: str):
        """保存模型"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }, f)
            
    def load_model(self, filename: str):
        """加载模型"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.params = data['params']
            self.scaler = data['scaler']
            self.is_fitted = data['is_fitted']


if __name__ == "__main__":
    # 测试模型
    print("Loading historical data...")
    df = pd.read_csv('historical_assignments.csv')
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # 训练模型
    model = NestedLogitLearner()
    model.fit(train_df)
    
    # 保存模型
    model.save_model('nested_logit_model.pkl')