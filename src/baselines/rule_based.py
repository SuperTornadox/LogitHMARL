import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class RoutingStrategy(Enum):
    S_SHAPE = "s_shape"
    RETURN = "return"
    NEAREST = "nearest"

@dataclass
class Assignment:
    picker_id: int
    order_id: int
    route: List[Tuple[int, int]]
    estimated_time: float

class RuleBasedAssigner:
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.S_SHAPE):
        self.strategy = strategy
        
    def assign_orders(self, 
                     pickers: List[Dict],
                     orders: List[Dict],
                     warehouse_grid: np.ndarray) -> List[Assignment]:
        assignments = []
        
        # Filter available pickers and pending orders
        available_pickers = [p for p in pickers if p['current_order'] is None]
        pending_orders = [o for o in orders if o['assigned_picker'] is None]
        
        if not available_pickers or not pending_orders:
            return assignments
            
        # Greedy assignment: nearest picker to each order
        for order in pending_orders:
            if not available_pickers:
                break
                
            # Find best picker for this order
            best_picker = None
            best_distance = float('inf')
            
            for picker in available_pickers:
                # Check equipment constraint
                if order.get('requires_forklift', False) and not picker['is_forklift']:
                    continue
                    
                # Calculate distance to first item in order
                if order['items']:
                    first_item = order['items'][0]
                    distance = self._manhattan_distance(
                        (picker['x'], picker['y']),
                        first_item
                    )
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_picker = picker
                        
            if best_picker:
                # Generate route for this assignment
                route = self._generate_route(
                    (best_picker['x'], best_picker['y']),
                    order['items'],
                    warehouse_grid
                )
                
                # Create assignment
                assignment = Assignment(
                    picker_id=best_picker['picker_id'],
                    order_id=order['order_id'],
                    route=route,
                    estimated_time=self._estimate_completion_time(route)
                )
                
                assignments.append(assignment)
                available_pickers.remove(best_picker)
                
        return assignments
        
    def _generate_route(self, 
                       start_pos: Tuple[int, int],
                       items: List[Tuple[int, int]],
                       warehouse_grid: np.ndarray) -> List[Tuple[int, int]]:
        
        if self.strategy == RoutingStrategy.S_SHAPE:
            return self._s_shape_route(start_pos, items, warehouse_grid)
        elif self.strategy == RoutingStrategy.RETURN:
            return self._return_route(start_pos, items, warehouse_grid)
        else:  # NEAREST
            return self._nearest_neighbor_route(start_pos, items)
            
    def _s_shape_route(self,
                      start_pos: Tuple[int, int],
                      items: List[Tuple[int, int]],
                      warehouse_grid: np.ndarray) -> List[Tuple[int, int]]:
        # S-Shape routing: traverse aisles in serpentine pattern
        # Group items by aisle
        aisles = {}
        for item in items:
            aisle = item[0] // 11  # Assuming aisle width pattern
            if aisle not in aisles:
                aisles[aisle] = []
            aisles[aisle].append(item)
            
        # Sort aisles
        sorted_aisles = sorted(aisles.keys())
        
        route = []  # 不包含起始位置
        direction = 1  # 1 for up, -1 for down
        
        for i, aisle in enumerate(sorted_aisles):
            # Sort items in aisle by y-coordinate
            aisle_items = sorted(aisles[aisle], key=lambda x: x[1] * direction)
            route.extend(aisle_items)
            direction *= -1  # Reverse direction for next aisle
            
        # 如果没有路径点，返回空列表
        return route if route else []
        
    def _return_route(self,
                     start_pos: Tuple[int, int],
                     items: List[Tuple[int, int]],
                     warehouse_grid: np.ndarray) -> List[Tuple[int, int]]:
        # Return routing: enter and exit each aisle from the same end
        aisles = {}
        for item in items:
            aisle = item[0] // 11
            if aisle not in aisles:
                aisles[aisle] = []
            aisles[aisle].append(item)
            
        route = []  # 不包含起始位置
        
        for aisle in sorted(aisles.keys()):
            # Enter aisle, pick items, return to main corridor
            aisle_items = sorted(aisles[aisle], key=lambda x: x[1])
            
            # Go to aisle entrance
            aisle_entrance = (aisle * 11 + 5, warehouse_grid.shape[0] - 5)
            route.append(aisle_entrance)
            
            # Pick items in order
            for item in aisle_items:
                route.append(item)
                
            # Return to entrance
            route.append(aisle_entrance)
            
        return route if route else []
        
    def _nearest_neighbor_route(self,
                               start_pos: Tuple[int, int],
                               items: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Greedy nearest neighbor
        route = []  # 不包含起始位置
        remaining = items.copy()
        current = start_pos
        
        while remaining:
            nearest = min(remaining, key=lambda x: self._manhattan_distance(current, x))
            route.append(nearest)
            current = nearest
            remaining.remove(nearest)
            
        return route
        
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def _estimate_completion_time(self, route: List[Tuple[int, int]]) -> float:
        # Estimate time based on route length
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self._manhattan_distance(route[i], route[i + 1])
            
        # Add picking time (2 seconds per item)
        picking_time = (len(route) - 1) * 2.0
        
        # Movement speed: 1 cell per second
        movement_time = total_distance * 1.0
        
        return movement_time + picking_time

class SShapeAssigner(RuleBasedAssigner):
    def __init__(self):
        super().__init__(RoutingStrategy.S_SHAPE)

class ReturnAssigner(RuleBasedAssigner):
    def __init__(self):
        super().__init__(RoutingStrategy.RETURN)