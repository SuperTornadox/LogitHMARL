"""
Nested Logit模型实现
基于离散选择理论的正确实现，用于仓库任务分配
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NestedLogitModel:
    """
    实现标准的Nested Logit模型用于任务分配
    
    模型结构:
    - 上层：选择巢（nest），如设备类型、紧急程度组合
    - 下层：在巢内选择具体的拣选员
    
    效用函数:
    U(i,j) = V(i,j) + ε(i,j)
    其中 V(i,j) 是可观测效用，ε(i,j) 是随机项
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化Nested Logit模型
        
        参数:
        - config: 模型配置参数
        """
        self.config = config or {}
        
        # 模型参数（可通过历史数据估计）
        self.params = {
            # 巢层参数
            'lambda': 0.7,  # 巢内相关系数 (0,1]，越接近1越类似MNL
            
            # 效用函数权重
            'w_distance': -0.5,      # 距离权重（负值表示距离越远效用越低）
            'w_workload': -0.3,      # 工作负荷权重
            'w_priority': 1.0,       # 优先级权重
            'w_equipment': 2.0,      # 设备匹配权重
            'w_time_pressure': 1.5,  # 时间压力权重
            
            # 巢特定参数
            'nest_constants': {
                'urgent_forklift': 0.5,
                'urgent_normal': 0.3,
                'normal_forklift': 0.2,
                'normal_normal': 0.0,
                'low_forklift': -0.2,
                'low_normal': -0.3
            }
        }
        
        # 更新参数
        if 'params' in self.config:
            self.params.update(self.config['params'])
            
    def create_nests(self, orders: pd.DataFrame) -> Dict[str, List[int]]:
        """
        创建巢结构
        基于订单属性（设备需求、紧急程度）创建巢
        """
        nests = {}
        
        for idx, order in orders.iterrows():
            # 确定紧急程度
            if order['sla_sec'] <= 30 * 60:
                urgency = 'urgent'
            elif order['sla_sec'] <= 60 * 60:
                urgency = 'normal'
            else:
                urgency = 'low'
                
            # 确定设备需求
            equipment = 'forklift' if order.get('eq_flag', 0) else 'normal'
            
            # 巢ID
            nest_id = f"{urgency}_{equipment}"
            
            if nest_id not in nests:
                nests[nest_id] = []
            nests[nest_id].append(idx)
            
        return nests
        
    def calculate_utility(self, order: pd.Series, picker: Dict, 
                         picker_state: Dict) -> float:
        """
        计算任务-拣选员配对的效用
        
        参数:
        - order: 订单信息
        - picker: 拣选员信息
        - picker_state: 拣选员当前状态（位置、工作负荷等）
        
        返回:
        - 效用值
        """
        utility = 0.0
        
        # 1. 距离成分
        if 'position' in picker_state and 'pickup_location' in order:
            distance = self._calculate_distance(
                picker_state['position'], 
                order['pickup_location']
            )
            utility += self.params['w_distance'] * distance
            
        # 2. 工作负荷成分
        workload = picker_state.get('current_workload', 0)
        max_workload = picker_state.get('max_workload', 10)
        workload_ratio = workload / max_workload if max_workload > 0 else 0
        utility += self.params['w_workload'] * workload_ratio
        
        # 3. 优先级成分
        priority_map = {'high': 1.0, 'normal': 0.5, 'low': 0.0}
        priority_value = priority_map.get(order.get('priority', 'normal'), 0.5)
        utility += self.params['w_priority'] * priority_value
        
        # 4. 设备匹配成分
        if order.get('eq_flag', 0):
            if picker.get('has_forklift', False):
                utility += self.params['w_equipment'] * 1.0
            else:
                utility += self.params['w_equipment'] * -10.0  # 严重惩罚
                
        # 5. 时间压力成分
        time_remaining = order.get('sla_sec', 3600) - order.get('elapsed_time', 0)
        time_pressure = max(0, 1 - time_remaining / 3600)  # 归一化到[0,1]
        utility += self.params['w_time_pressure'] * time_pressure
        
        return utility
        
    def calculate_inclusive_value(self, nest_id: str, 
                                alternatives: List[Tuple[float, int]]) -> float:
        """
        计算巢的包容值（Inclusive Value）
        这是Nested Logit模型的关键组成部分
        
        IV_n = ln(Σ exp(V_j/λ)) for all j in nest n
        """
        if not alternatives:
            return -np.inf
            
        lambda_param = self.params['lambda']
        
        # 提取效用值
        utilities = [u for u, _ in alternatives]
        
        # 计算包容值，避免数值溢出
        max_utility = max(utilities)
        sum_exp = sum(np.exp((u - max_utility) / lambda_param) for u in utilities)
        
        inclusive_value = max_utility + lambda_param * np.log(sum_exp)
        
        return inclusive_value
        
    def calculate_nest_probability(self, nest_id: str, 
                                  inclusive_values: Dict[str, float]) -> float:
        """
        计算选择某个巢的概率
        
        P(nest) = exp(α_nest + λ*IV_nest) / Σ exp(α_nest' + λ*IV_nest')
        """
        nest_constant = self.params['nest_constants'].get(nest_id, 0.0)
        lambda_param = self.params['lambda']
        
        # 计算分子
        nest_value = nest_constant + lambda_param * inclusive_values[nest_id]
        
        # 计算分母（所有巢的和）
        denominator = 0.0
        for n_id, iv in inclusive_values.items():
            n_const = self.params['nest_constants'].get(n_id, 0.0)
            denominator += np.exp(n_const + lambda_param * iv)
            
        # 避免数值问题
        if denominator == 0:
            return 0.0
            
        probability = np.exp(nest_value) / denominator
        
        return probability
        
    def calculate_conditional_probability(self, picker_utility: float,
                                        nest_utilities: List[float]) -> float:
        """
        计算在给定巢内选择某个拣选员的条件概率
        
        P(j|nest) = exp(V_j/λ) / Σ exp(V_k/λ) for all k in nest
        """
        lambda_param = self.params['lambda']
        
        # 避免数值溢出
        max_utility = max(nest_utilities)
        
        numerator = np.exp((picker_utility - max_utility) / lambda_param)
        denominator = sum(np.exp((u - max_utility) / lambda_param) 
                         for u in nest_utilities)
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
        
    def assign_orders(self, orders: pd.DataFrame, 
                     pickers: List[Dict],
                     picker_states: Dict[str, Dict]) -> pd.DataFrame:
        """
        使用Nested Logit模型分配订单给拣选员
        
        参数:
        - orders: 待分配的订单
        - pickers: 拣选员列表
        - picker_states: 拣选员当前状态
        
        返回:
        - 分配结果DataFrame
        """
        assignments = []
        
        # 为订单创建巢结构
        nests = self.create_nests(orders)
        
        for order_idx, order in orders.iterrows():
            # 确定订单所属的巢
            if order['sla_sec'] <= 30 * 60:
                urgency = 'urgent'
            elif order['sla_sec'] <= 60 * 60:
                urgency = 'normal'
            else:
                urgency = 'low'
            equipment = 'forklift' if order.get('eq_flag', 0) else 'normal'
            order_nest = f"{urgency}_{equipment}"
            
            # 计算每个拣选员的效用
            picker_utilities = {}
            for picker in pickers:
                picker_id = picker['id']
                if picker_id in picker_states:
                    utility = self.calculate_utility(
                        order, picker, picker_states[picker_id]
                    )
                    picker_utilities[picker_id] = utility
                    
            # 按巢组织拣选员
            nest_alternatives = {}
            for nest_id in nests:
                nest_alternatives[nest_id] = []
                
                # 根据巢的要求筛选合适的拣选员
                for picker in pickers:
                    picker_id = picker['id']
                    
                    # 检查设备匹配
                    if 'forklift' in nest_id and not picker.get('has_forklift', False):
                        continue
                        
                    if picker_id in picker_utilities:
                        nest_alternatives[nest_id].append(
                            (picker_utilities[picker_id], picker_id)
                        )
                        
            # 计算每个巢的包容值
            inclusive_values = {}
            for nest_id, alternatives in nest_alternatives.items():
                if alternatives:
                    inclusive_values[nest_id] = self.calculate_inclusive_value(
                        nest_id, alternatives
                    )
                else:
                    inclusive_values[nest_id] = -np.inf
                    
            # 计算选择每个巢的概率
            nest_probabilities = {}
            valid_nests = [n for n, iv in inclusive_values.items() if iv > -np.inf]
            
            if valid_nests:
                for nest_id in valid_nests:
                    nest_probabilities[nest_id] = self.calculate_nest_probability(
                        nest_id, inclusive_values
                    )
                    
                # 基于概率选择巢（这里可以用随机抽样或选最大概率）
                # 为了演示，选择概率最大的巢
                selected_nest = max(nest_probabilities, key=nest_probabilities.get)
                
                # 在选定的巢内选择拣选员
                if nest_alternatives[selected_nest]:
                    # 计算条件概率
                    nest_utilities = [u for u, _ in nest_alternatives[selected_nest]]
                    picker_probs = []
                    
                    for utility, picker_id in nest_alternatives[selected_nest]:
                        cond_prob = self.calculate_conditional_probability(
                            utility, nest_utilities
                        )
                        picker_probs.append((cond_prob, picker_id))
                        
                    # 选择概率最大的拣选员
                    selected_picker = max(picker_probs, key=lambda x: x[0])[1]
                    
                    # 记录分配
                    assignments.append({
                        'order_id': order['order_id'],
                        'picker_id': selected_picker,
                        'assign_ts': order['create_ts'] + np.random.uniform(10, 60),
                        'nest_id': selected_nest,
                        'nest_prob': nest_probabilities[selected_nest],
                        'picker_prob': max(picker_probs, key=lambda x: x[0])[0]
                    })
                    
                    # 更新拣选员状态
                    if selected_picker in picker_states:
                        picker_states[selected_picker]['current_workload'] += 1
                        
        return pd.DataFrame(assignments)
        
    def _calculate_distance(self, pos1: Tuple[int, int], 
                          pos2: Tuple[int, int]) -> float:
        """计算曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def estimate_parameters(self, historical_data: pd.DataFrame) -> Dict:
        """
        从历史数据估计模型参数
        这里需要使用最大似然估计或其他优化方法
        
        参数:
        - historical_data: 包含历史分配决策的数据
        
        返回:
        - 估计的参数字典
        """
        # 这是一个简化的实现框架
        # 实际应用中需要使用如statsmodels或pylogit等包进行参数估计
        
        logger.info("估计Nested Logit模型参数...")
        
        # TODO: 实现参数估计逻辑
        # 1. 构建似然函数
        # 2. 使用优化算法（如L-BFGS）最大化似然函数
        # 3. 返回估计的参数
        
        return self.params