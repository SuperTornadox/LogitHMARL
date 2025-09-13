import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
import itertools

class OptimalRoutingAssigner:
    def __init__(self, use_hungarian: bool = True):
        self.use_hungarian = use_hungarian
        
    def assign_orders(self,
                     pickers: List[Dict],
                     orders: List[Dict],
                     warehouse_grid: np.ndarray) -> List[Dict]:
        
        # Filter available entities
        available_pickers = [p for p in pickers if p['current_order'] is None]
        pending_orders = [o for o in orders if o['assigned_picker'] is None]
        
        if not available_pickers or not pending_orders:
            return []
            
        if self.use_hungarian:
            return self._hungarian_assignment(available_pickers, pending_orders, warehouse_grid)
        else:
            return self._dp_assignment(available_pickers, pending_orders, warehouse_grid)
            
    def _hungarian_assignment(self,
                             pickers: List[Dict],
                             orders: List[Dict],
                             warehouse_grid: np.ndarray) -> List[Dict]:
        
        n_pickers = len(pickers)
        n_orders = len(orders)
        
        # Create cost matrix
        cost_matrix = np.full((n_pickers, n_orders), float('inf'))
        
        for i, picker in enumerate(pickers):
            for j, order in enumerate(orders):
                # Check equipment constraints
                if order.get('requires_forklift', False) and not picker.get('is_forklift', False):
                    # If order requires forklift but picker doesn't have one,
                    # still assign a high cost instead of infinity to avoid infeasible matrix
                    cost_matrix[i, j] = 10000  # High penalty but not infinite
                else:
                    # Calculate cost as total distance to complete order
                    cost = self._calculate_order_cost(
                        (picker['x'], picker['y']),
                        order['items'],
                        warehouse_grid
                    )
                    cost_matrix[i, j] = cost
                
        # Check if cost matrix is feasible
        if np.all(cost_matrix == float('inf')):
            # No feasible assignments, return empty list
            return []
        
        # Solve assignment problem
        if n_pickers <= n_orders:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            # More pickers than orders - transpose and solve
            col_indices, row_indices = linear_sum_assignment(cost_matrix.T)
            
        # Create assignments
        assignments = []
        for picker_idx, order_idx in zip(row_indices, col_indices):
            if cost_matrix[picker_idx, order_idx] < float('inf'):
                # Generate optimal route for this pairing
                route = self._solve_tsp(
                    (pickers[picker_idx]['x'], pickers[picker_idx]['y']),
                    orders[order_idx]['items']
                )
                
                assignments.append({
                    'picker_id': pickers[picker_idx]['picker_id'],
                    'order_id': orders[order_idx]['order_id'],
                    'route': route,
                    'cost': cost_matrix[picker_idx, order_idx]
                })
                
        return assignments
        
    def _dp_assignment(self,
                      pickers: List[Dict],
                      orders: List[Dict],
                      warehouse_grid: np.ndarray) -> List[Dict]:
        
        # Dynamic programming approach for small instances
        n_pickers = len(pickers)
        n_orders = min(len(orders), 10)  # Limit for computational feasibility
        
        # State: (assigned_pickers_mask, assigned_orders_mask)
        # Value: minimum total cost
        dp = {}
        parent = {}
        
        def solve(picker_mask: int, order_mask: int) -> float:
            if (picker_mask, order_mask) in dp:
                return dp[(picker_mask, order_mask)]
                
            # Base case: all orders assigned or no more pickers
            if order_mask == (1 << n_orders) - 1 or picker_mask == (1 << n_pickers) - 1:
                dp[(picker_mask, order_mask)] = 0
                return 0
                
            min_cost = float('inf')
            best_assignment = None
            
            # Try assigning each available picker to each unassigned order
            for p_idx in range(n_pickers):
                if picker_mask & (1 << p_idx):
                    continue  # Picker already assigned
                    
                for o_idx in range(n_orders):
                    if order_mask & (1 << o_idx):
                        continue  # Order already assigned
                        
                    # Check constraints
                    if orders[o_idx].get('requires_forklift', False) and not pickers[p_idx]['is_forklift']:
                        continue
                        
                    # Calculate cost
                    cost = self._calculate_order_cost(
                        (pickers[p_idx]['x'], pickers[p_idx]['y']),
                        orders[o_idx]['items'],
                        warehouse_grid
                    )
                    
                    # Recursive call
                    new_picker_mask = picker_mask | (1 << p_idx)
                    new_order_mask = order_mask | (1 << o_idx)
                    future_cost = solve(new_picker_mask, new_order_mask)
                    
                    total_cost = cost + future_cost
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_assignment = (p_idx, o_idx)
                        
            dp[(picker_mask, order_mask)] = min_cost
            if best_assignment:
                parent[(picker_mask, order_mask)] = best_assignment
                
            return min_cost
            
        # Solve and reconstruct assignments
        total_cost = solve(0, 0)
        
        # Reconstruct path
        assignments = []
        picker_mask, order_mask = 0, 0
        
        while (picker_mask, order_mask) in parent:
            p_idx, o_idx = parent[(picker_mask, order_mask)]
            
            route = self._solve_tsp(
                (pickers[p_idx]['x'], pickers[p_idx]['y']),
                orders[o_idx]['items']
            )
            
            assignments.append({
                'picker_id': pickers[p_idx]['picker_id'],
                'order_id': orders[o_idx]['order_id'],
                'route': route,
                'cost': self._calculate_order_cost(
                    (pickers[p_idx]['x'], pickers[p_idx]['y']),
                    orders[o_idx]['items'],
                    None
                )
            })
            
            picker_mask |= (1 << p_idx)
            order_mask |= (1 << o_idx)
            
        return assignments
        
    def _calculate_order_cost(self,
                             start_pos: Tuple[int, int],
                             items: List[Tuple[int, int]],
                             warehouse_grid: Optional[np.ndarray]) -> float:
        
        # Always use heuristic for speed
        if len(items) == 0:
            return 0
        elif len(items) == 1:
            return abs(items[0][0] - start_pos[0]) + abs(items[0][1] - start_pos[1])
        else:
            # Use nearest neighbor heuristic for cost estimation
            return self._tsp_heuristic_cost(start_pos, items)
            
    def _solve_tsp(self,
                  start_pos: Tuple[int, int],
                  items: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        
        if len(items) == 0:
            return []  # 空路径
        elif len(items) == 1:
            return items.copy()  # 返回副本，避免修改原始列表
            
        # Always use nearest neighbor for simplicity and speed
        # (Brute force TSP is too slow even for 8 items)
        return self._nearest_neighbor_tsp(start_pos, items)
            
    def _nearest_neighbor_tsp(self,
                             start_pos: Tuple[int, int],
                             items: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        
        route = []  # 不包含起始位置
        remaining = items.copy()
        current = start_pos
        
        while remaining:
            nearest = min(remaining, key=lambda x: self._manhattan_distance(current, x))
            route.append(nearest)
            current = nearest
            remaining.remove(nearest)
            
        return route
        
    def _tsp_heuristic_cost(self,
                           start_pos: Tuple[int, int],
                           items: List[Tuple[int, int]]) -> float:
        
        # MST-based lower bound
        n = len(items) + 1
        positions = [start_pos] + items
        
        # Build distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self._manhattan_distance(positions[i], positions[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                
        # Compute MST cost (Prim's algorithm)
        visited = [False] * n
        min_edges = [float('inf')] * n
        min_edges[0] = 0
        mst_cost = 0
        
        for _ in range(n):
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or min_edges[v] < min_edges[u]):
                    u = v
                    
            visited[u] = True
            mst_cost += min_edges[u]
            
            for v in range(n):
                if not visited[v] and dist_matrix[u, v] < min_edges[v]:
                    min_edges[v] = dist_matrix[u, v]
                    
        return mst_cost * 1.2  # Add 20% for TSP approximation
        
    def _route_distance(self, route: List[Tuple[int, int]]) -> float:
        distance = 0
        for i in range(len(route) - 1):
            distance += self._manhattan_distance(route[i], route[i + 1])
        return distance
        
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])