"""
高级Nested Logit模型实现
包含参数估计、模型验证和性能优化
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class NestedLogitConfig:
    """Nested Logit模型配置"""
    # 模型结构参数
    n_nests: int = 6  # 巢的数量
    lambda_param: float = 0.7  # 巢内相关系数
    
    # 效用函数参数
    beta_distance: float = -0.5
    beta_workload: float = -0.3
    beta_priority: float = 1.0
    beta_equipment: float = 2.0
    beta_time: float = 1.5
    beta_skill_match: float = 0.8
    
    # 巢特定常数
    nest_constants: Dict[str, float] = None
    
    # 优化参数
    max_iter: int = 1000
    tolerance: float = 1e-6
    
    def __post_init__(self):
        if self.nest_constants is None:
            self.nest_constants = {
                'urgent_forklift': 0.5,
                'urgent_normal': 0.3,
                'normal_forklift': 0.2,
                'normal_normal': 0.0,
                'low_forklift': -0.2,
                'low_normal': -0.3
            }


class AdvancedNestedLogit:
    """
    高级Nested Logit模型
    
    特性：
    1. 完整的Nested Logit概率计算
    2. 最大似然参数估计
    3. 模型诊断和验证
    4. 效率优化（向量化计算）
    """
    
    def __init__(self, config: NestedLogitConfig = None):
        self.config = config or NestedLogitConfig()
        self.is_fitted = False
        self.estimation_results = None
        
    def compute_systematic_utility(self, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        计算系统效用 V = X'β
        
        参数:
        - X: 特征矩阵 (n_obs, n_features)
        - beta: 参数向量 (n_features,)
        
        返回:
        - V: 效用向量 (n_obs,)
        """
        return X @ beta
        
    def compute_inclusive_value(self, utilities: np.ndarray, 
                               lambda_param: float,
                               nest_membership: np.ndarray) -> np.ndarray:
        """
        计算每个巢的包容值
        
        IV_m = ln(Σ_j∈Bm exp(V_j/λ_m))
        """
        n_nests = nest_membership.shape[1]
        iv = np.zeros(n_nests)
        
        for m in range(n_nests):
            # 获取属于巢m的选择项
            in_nest = nest_membership[:, m] > 0
            if np.any(in_nest):
                nest_utils = utilities[in_nest] / lambda_param
                iv[m] = logsumexp(nest_utils)
            else:
                iv[m] = -np.inf
                
        return iv
        
    def compute_nest_probabilities(self, inclusive_values: np.ndarray,
                                  nest_constants: np.ndarray,
                                  lambda_param: float) -> np.ndarray:
        """
        计算选择每个巢的概率
        
        P(m) = exp(α_m + λ*IV_m) / Σ_k exp(α_k + λ*IV_k)
        """
        # 计算巢的效用
        nest_utilities = nest_constants + lambda_param * inclusive_values
        
        # 使用logsumexp避免数值溢出
        log_probs = nest_utilities - logsumexp(nest_utilities)
        probs = np.exp(log_probs)
        
        return probs
        
    def compute_conditional_probabilities(self, utilities: np.ndarray,
                                         lambda_param: float,
                                         nest_membership: np.ndarray) -> np.ndarray:
        """
        计算在每个巢内的条件选择概率
        
        P(j|m) = exp(V_j/λ_m) / Σ_k∈Bm exp(V_k/λ_m)
        """
        n_alternatives = len(utilities)
        n_nests = nest_membership.shape[1]
        cond_probs = np.zeros((n_alternatives, n_nests))
        
        for m in range(n_nests):
            in_nest = nest_membership[:, m] > 0
            if np.any(in_nest):
                nest_utils = utilities[in_nest] / lambda_param
                log_probs = nest_utils - logsumexp(nest_utils)
                cond_probs[in_nest, m] = np.exp(log_probs)
                
        return cond_probs
        
    def predict_probabilities(self, X: np.ndarray, 
                            nest_membership: np.ndarray) -> np.ndarray:
        """
        预测选择概率
        
        P(j) = Σ_m P(m) * P(j|m)
        """
        # 提取参数
        beta = self._pack_parameters()
        
        # 计算效用
        utilities = self.compute_systematic_utility(X, beta[:X.shape[1]])
        
        # 计算包容值
        iv = self.compute_inclusive_value(utilities, 
                                         self.config.lambda_param,
                                         nest_membership)
        
        # 计算巢概率
        nest_constants = beta[X.shape[1]:X.shape[1]+self.config.n_nests]
        nest_probs = self.compute_nest_probabilities(iv, 
                                                     nest_constants,
                                                     self.config.lambda_param)
        
        # 计算条件概率
        cond_probs = self.compute_conditional_probabilities(utilities,
                                                           self.config.lambda_param,
                                                           nest_membership)
        
        # 计算最终概率
        probs = cond_probs @ nest_probs
        
        return probs
        
    def log_likelihood(self, params: np.ndarray, X: np.ndarray,
                      y: np.ndarray, nest_membership: np.ndarray) -> float:
        """
        计算对数似然函数
        
        LL = Σ_n Σ_j y_nj * ln(P_nj)
        """
        # 更新参数
        self._unpack_parameters(params)
        
        # 计算概率
        probs = self.predict_probabilities(X, nest_membership)
        
        # 避免log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        
        # 计算对数似然
        ll = np.sum(y * np.log(probs))
        
        return -ll  # 返回负值用于最小化
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            nest_membership: np.ndarray,
            initial_params: Optional[np.ndarray] = None) -> Dict:
        """
        使用最大似然估计拟合模型
        
        参数:
        - X: 特征矩阵 (n_obs, n_features)
        - y: 选择指示矩阵 (n_obs, n_alternatives)
        - nest_membership: 巢成员矩阵 (n_alternatives, n_nests)
        - initial_params: 初始参数值
        
        返回:
        - 估计结果字典
        """
        logger.info("开始估计Nested Logit模型参数...")
        
        # 初始化参数
        if initial_params is None:
            n_params = X.shape[1] + self.config.n_nests + 1  # beta + nest_constants + lambda
            initial_params = np.zeros(n_params)
            initial_params[-1] = 0.5  # lambda初始值
            
        # 定义约束：lambda必须在(0,1]之间
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[-1]},  # lambda > 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[-1]}  # lambda <= 1
        ]
        
        # 优化
        result = minimize(
            fun=self.log_likelihood,
            x0=initial_params,
            args=(X, y, nest_membership),
            method='L-BFGS-B',
            constraints=constraints,
            options={
                'maxiter': self.config.max_iter,
                'ftol': self.config.tolerance
            }
        )
        
        # 保存结果
        self.estimation_results = {
            'success': result.success,
            'log_likelihood': -result.fun,
            'parameters': result.x,
            'n_iterations': result.nit,
            'message': result.message
        }
        
        # 更新模型参数
        self._unpack_parameters(result.x)
        self.is_fitted = True
        
        # 计算标准误差和统计量
        self._compute_statistics(X, y, nest_membership)
        
        logger.info(f"参数估计完成. LL = {self.estimation_results['log_likelihood']:.4f}")
        
        return self.estimation_results
        
    def _compute_statistics(self, X: np.ndarray, y: np.ndarray,
                          nest_membership: np.ndarray):
        """计算参数的标准误差、t统计量等"""
        # 计算Hessian矩阵的逆（信息矩阵）
        # 这里简化处理，实际应用需要数值计算Hessian
        
        n_params = len(self.estimation_results['parameters'])
        self.estimation_results['std_errors'] = np.ones(n_params) * 0.1  # 占位
        self.estimation_results['t_stats'] = (
            self.estimation_results['parameters'] / 
            self.estimation_results['std_errors']
        )
        
        # 计算伪R²
        # McFadden's pseudo R²
        ll_model = self.estimation_results['log_likelihood']
        ll_null = -len(y) * np.log(y.shape[1])  # 等概率模型
        self.estimation_results['pseudo_r2'] = 1 - ll_model / ll_null
        
    def _pack_parameters(self) -> np.ndarray:
        """将模型参数打包成向量"""
        params = []
        
        # Beta参数
        params.extend([
            self.config.beta_distance,
            self.config.beta_workload,
            self.config.beta_priority,
            self.config.beta_equipment,
            self.config.beta_time,
            self.config.beta_skill_match
        ])
        
        # 巢常数
        params.extend(list(self.config.nest_constants.values()))
        
        # Lambda参数
        params.append(self.config.lambda_param)
        
        return np.array(params)
        
    def _unpack_parameters(self, params: np.ndarray):
        """从向量解包参数到模型配置"""
        idx = 0
        
        # Beta参数
        self.config.beta_distance = params[idx]
        self.config.beta_workload = params[idx+1]
        self.config.beta_priority = params[idx+2]
        self.config.beta_equipment = params[idx+3]
        self.config.beta_time = params[idx+4]
        self.config.beta_skill_match = params[idx+5]
        idx += 6
        
        # 巢常数
        nest_names = list(self.config.nest_constants.keys())
        for i, name in enumerate(nest_names):
            self.config.nest_constants[name] = params[idx+i]
        idx += len(nest_names)
        
        # Lambda参数
        self.config.lambda_param = params[idx]
        
    def simulate_choices(self, X: np.ndarray, 
                        nest_membership: np.ndarray,
                        n_simulations: int = 1) -> np.ndarray:
        """
        基于模型模拟选择
        
        返回:
        - 选择矩阵 (n_simulations, n_alternatives)
        """
        probs = self.predict_probabilities(X, nest_membership)
        
        choices = np.zeros((n_simulations, len(probs)))
        for i in range(n_simulations):
            choice = np.random.choice(len(probs), p=probs)
            choices[i, choice] = 1
            
        return choices
        
    def save_model(self, filepath: str):
        """保存模型参数"""
        model_data = {
            'config': {
                'n_nests': self.config.n_nests,
                'lambda_param': self.config.lambda_param,
                'beta_distance': self.config.beta_distance,
                'beta_workload': self.config.beta_workload,
                'beta_priority': self.config.beta_priority,
                'beta_equipment': self.config.beta_equipment,
                'beta_time': self.config.beta_time,
                'beta_skill_match': self.config.beta_skill_match,
                'nest_constants': self.config.nest_constants
            },
            'is_fitted': self.is_fitted,
            'estimation_results': self.estimation_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
            
        logger.info(f"模型已保存到 {filepath}")
        
    def load_model(self, filepath: str):
        """加载模型参数"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
            
        # 恢复配置
        config_data = model_data['config']
        self.config = NestedLogitConfig(**config_data)
        
        # 恢复状态
        self.is_fitted = model_data['is_fitted']
        self.estimation_results = model_data['estimation_results']
        
        logger.info(f"模型已从 {filepath} 加载")