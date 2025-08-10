"""
Unified Warehouse System

This single file consolidates the core simulation (WarehouseSimulation)
and a correct Nested Logit model (NestedLogitModel) into one entrypoint.

Features:
- Grid/slot generation, SKU master, initial inventory allocation
- Poisson order generation with Zipf SKU lines
- Task assignment via Nested Logit (inclusive value, nest prob, conditional prob)
- SimPy-based picker execution with NetworkX A* movement
- Replenishment by ROP rule
- CSV outputs compatible with prior pipeline

CLI:
  python3 warehouse_system.py run-sim            # run full-day simulation
  python3 warehouse_system.py run-sim --sim-time 14400 --n-pickers 8

Note: This is a consolidation for convenience; original modular files
remain available for advanced workflows.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import simpy
import networkx as nx
from sklearn.cluster import KMeans


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Nested Logit (consolidated)
# -----------------------------------------------------------------------------
class NestedLogitModel:
    """
    Standard Nested Logit for picker assignment.

    Structure:
    - Upper level: choose a nest (urgency x equipment)
    - Lower level: choose a picker within the selected nest

    Utility: U = V + epsilon, where V is observable utility
    """

    def __init__(self, config: Dict | None = None):
        self.config = config or {}

        # Default parameters; can be learned offline
        self.params = {
            'lambda': 0.7,                 # (0,1], closer to 1 => MNL-like
            'w_distance': -0.5,
            'w_workload': -0.3,
            'w_priority': 1.0,
            'w_equipment': 2.0,
            'w_time_pressure': 1.5,
            'nest_constants': {
                'urgent_forklift': 0.5,
                'urgent_normal': 0.3,
                'normal_forklift': 0.2,
                'normal_normal': 0.0,
                'low_forklift': -0.2,
                'low_normal': -0.3,
            },
        }

        if 'params' in self.config:
            self.params.update(self.config['params'])

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def calculate_utility(self, order: pd.Series, picker: Dict, picker_state: Dict) -> float:
        utility = 0.0

        # Distance term
        if 'position' in picker_state and isinstance(order.get('pickup_location', None), (tuple, list)):
            distance = self._manhattan(picker_state['position'], order['pickup_location'])
            utility += self.params['w_distance'] * distance

        # Workload ratio
        workload = picker_state.get('current_workload', 0)
        max_workload = picker_state.get('max_workload', 10)
        workload_ratio = workload / max_workload if max_workload > 0 else 0
        utility += self.params['w_workload'] * workload_ratio

        # Priority
        pr_map = {'high': 1.0, 'normal': 0.5, 'low': 0.0}
        utility += self.params['w_priority'] * pr_map.get(order.get('priority', 'normal'), 0.5)

        # Equipment
        if order.get('eq_flag', 0):
            utility += self.params['w_equipment'] * (1.0 if picker.get('has_forklift', False) else -10.0)

        # Time pressure
        time_remaining = order.get('sla_sec', 3600) - order.get('elapsed_time', 0)
        time_pressure = max(0, 1 - time_remaining / 3600)
        utility += self.params['w_time_pressure'] * time_pressure

        return utility

    def _inclusive_value(self, utilities: List[float]) -> float:
        if not utilities:
            return -np.inf
        lambda_param = self.params['lambda']
        max_u = max(utilities)
        sum_exp = sum(np.exp((u - max_u) / lambda_param) for u in utilities)
        return max_u + lambda_param * np.log(sum_exp)

    def _nest_prob(self, nest_id: str, inclusive_values: Dict[str, float]) -> float:
        n_const = self.params['nest_constants'].get(nest_id, 0.0)
        lambda_param = self.params['lambda']
        numerator = np.exp(n_const + lambda_param * inclusive_values[nest_id])
        denom = sum(np.exp(self.params['nest_constants'].get(n, 0.0) + lambda_param * iv)
                    for n, iv in inclusive_values.items())
        return 0.0 if denom == 0 else numerator / denom

    def _cond_prob(self, u_j: float, utilities: List[float]) -> float:
        lambda_param = self.params['lambda']
        max_u = max(utilities)
        num = np.exp((u_j - max_u) / lambda_param)
        den = sum(np.exp((u - max_u) / lambda_param) for u in utilities)
        return 0.0 if den == 0 else num / den

    def assign_orders(self, orders: pd.DataFrame, pickers: List[Dict], picker_states: Dict[str, Dict]) -> pd.DataFrame:
        assignments = []

        # Pre-compute nest membership for each order
        for _, order in orders.iterrows():
            if order['sla_sec'] <= 30 * 60:
                urgency = 'urgent'
            elif order['sla_sec'] <= 60 * 60:
                urgency = 'normal'
            else:
                urgency = 'low'
            equipment = 'forklift' if order.get('eq_flag', 0) else 'normal'

            order_nest = f"{urgency}_{equipment}"

            # Utilities per picker
            picker_utils: Dict[str, float] = {}
            for picker in pickers:
                pid = picker['id']
                if pid in picker_states:
                    picker_utils[pid] = self.calculate_utility(order, picker, picker_states[pid])

            # Build nest alternatives
            nest_alts: Dict[str, List[Tuple[float, str]]] = {}
            all_nests = ['urgent_forklift', 'urgent_normal', 'normal_forklift', 'normal_normal', 'low_forklift', 'low_normal']
            for n in all_nests:
                nest_alts[n] = []
                need_fk = 'forklift' in n
                for picker in pickers:
                    pid = picker['id']
                    if need_fk and not picker.get('has_forklift', False):
                        continue
                    if pid in picker_utils:
                        u = picker_utils[pid]
                        if need_fk:
                            u += self.params['nest_constants'].get('urgent_forklift' if 'urgent' in n else 'normal_forklift' if 'normal' in n else 'low_forklift', 0.0)
                        nest_alts[n].append((u, pid))

            # Inclusive values
            inclusive_values: Dict[str, float] = {n: (self._inclusive_value([u for u, _ in alts]) if alts else -np.inf)
                                                  for n, alts in nest_alts.items()}

            # Valid nests
            valid_nests = [n for n, iv in inclusive_values.items() if iv > -np.inf]
            if not valid_nests:
                continue

            # Nest probabilities
            nest_probs = {n: self._nest_prob(n, inclusive_values) for n in valid_nests}
            selected_nest = max(nest_probs, key=nest_probs.get)

            # Conditional picker choice within nest
            if nest_alts[selected_nest]:
                nest_utils = [u for u, _ in nest_alts[selected_nest]]
                picker_probs = []
                for u, pid in nest_alts[selected_nest]:
                    cp = self._cond_prob(u, nest_utils)
                    picker_probs.append((cp, pid))
                chosen_picker = max(picker_probs, key=lambda x: x[0])[1]

                assignments.append({
                    'order_id': order['order_id'],
                    'picker_id': chosen_picker,
                    'assign_ts': float(order['create_ts']) + float(np.random.uniform(10, 60)),
                    'nest_id': selected_nest,
                    'nest_prob': float(nest_probs[selected_nest]),
                    'picker_prob': float(max(picker_probs, key=lambda x: x[0])[0]),
                })

        return pd.DataFrame(assignments)


# -----------------------------------------------------------------------------
# Warehouse Simulation (consolidated)
# -----------------------------------------------------------------------------
class WarehouseSimulation:
    def __init__(self, config: Dict | None = None):
        self.config = {
            'GRID_X': 32,
            'GRID_Y': 20,
            'GRID_Z': 3,
            'N_SKU': 1000,
            'ABC_SPLIT': (0.2, 0.3, 0.5),
            'ABC_QTY': (400, 120, 40),
            'SIM_TIME': 8 * 60 * 60,
            'ORDER_LAMBDA': 2 / 60,   # 2 per minute
            'ZIPF_A': 1.2,
            'N_PICKERS': 6,
            'FORKLIFT_RATIO': 0.15,
            'WALK_SPEED': 1,
            'PICK_TIME': 5,
            'LOAD_TIME': 8,
            'ROP_SAFETY_DAYS': 0.5,
            'LEAD_TIME_MEAN': 2 * 60 * 60,
            'LEAD_TIME_STD': 30 * 60,
            'REPLENISH_QTY_FACTOR': 1.5,
            'N_CLUSTERS': 8,
            'RANDOM_SEED': 42,
        }
        if config:
            self.config.update(config)

        self.rng = np.random.default_rng(self.config['RANDOM_SEED'])
        self.n_slots = self.config['GRID_X'] * self.config['GRID_Y'] * self.config['GRID_Z']

        # Storage
        self.slots: pd.DataFrame | None = None
        self.sku_df: pd.DataFrame | None = None
        self.orders: pd.DataFrame | None = None
        self.assign: pd.DataFrame | None = None
        self.picks: pd.DataFrame | None = None
        self.replenishment_log: List[Dict] = []

    # ------------------------------- Data generation -------------------------
    def generate_slots(self):
        logger.info("生成货位拓扑...")
        x, y, z = np.meshgrid(
            np.arange(self.config['GRID_X']),
            np.arange(self.config['GRID_Y']),
            np.arange(self.config['GRID_Z']),
        )
        self.slots = pd.DataFrame({
            'slot_id': np.arange(self.n_slots),
            'x': x.flatten(), 'y': y.flatten(), 'z': z.flatten(),
            'max_capacity': 999,
            'level': z.flatten(),
        })
        self.slots.to_csv("slots.csv", index=False)
        logger.info(f"生成 {self.n_slots} 个货位")

    def generate_sku_master(self):
        logger.info("生成SKU主数据...")
        self.sku_df = pd.DataFrame({'sku_id': np.arange(self.config['N_SKU'])})
        # ABC split
        abc_bins = [-1,
                    int(self.config['N_SKU'] * self.config['ABC_SPLIT'][0]),
                    int(self.config['N_SKU'] * (self.config['ABC_SPLIT'][0] + self.config['ABC_SPLIT'][1])),
                    self.config['N_SKU']]
        self.sku_df['class'] = pd.cut(self.sku_df['sku_id'], bins=abc_bins, labels=['A', 'B', 'C'])

        self.sku_df['init_qty'] = self.sku_df['class'].map({
            'A': self.config['ABC_QTY'][0],
            'B': self.config['ABC_QTY'][1],
            'C': self.config['ABC_QTY'][2],
        })
        qty_variation = self.rng.normal(1.0, 0.1, size=len(self.sku_df))
        base_qty = self.sku_df['init_qty'].astype(float)
        self.sku_df['init_qty'] = (base_qty * np.clip(qty_variation, 0.8, 1.2)).astype(int)

        # Dimensions
        self.sku_df['height'] = np.clip(self.rng.triangular(5, 20, 120, size=self.config['N_SKU']), 5, 120).astype(int)
        self.sku_df['length'] = np.clip(self.rng.triangular(5, 20, 120, size=self.config['N_SKU']), 5, 120).astype(int)
        self.sku_df['width']  = np.clip(self.rng.triangular(5, 20, 120, size=self.config['N_SKU']), 5, 120).astype(int)
        self.sku_df['volume'] = self.sku_df['height'] * self.sku_df['length'] * self.sku_df['width']
        self.sku_df['weight'] = (self.sku_df['volume'] * 0.001 * self.rng.uniform(0.5, 2.0, size=self.config['N_SKU'])).astype(int)

        # Demand estimate for ROP
        self.sku_df['daily_demand'] = self.sku_df['class'].map({
            'A': self.rng.uniform(50, 100),
            'B': self.rng.uniform(20, 50),
            'C': self.rng.uniform(5, 20),
        })
        self.sku_df.to_csv("sku_master.csv", index=False)
        logger.info(f"生成 {self.config['N_SKU']} 个SKU")

    def initial_inventory_allocation(self):
        logger.info("执行初始库存分配和货位优化...")
        assert self.slots is not None and self.sku_df is not None

        self.slots['sku'] = self.rng.choice(self.sku_df['sku_id'], size=self.n_slots)
        self.slots['quantity'] = self.slots['sku'].map(self.sku_df.set_index('sku_id')['init_qty'])

        # Velocity-based slotting: put A-class in front 1/3 rows
        high_ids = self.sku_df.query("`class`=='A'")['sku_id'].values
        front_rows = self.slots.query("y < @self.config['GRID_Y']/3").index
        if len(front_rows) > 0 and len(high_ids) > 0:
            self.slots.loc[front_rows, 'sku'] = self.rng.choice(high_ids, size=len(front_rows))
            self.slots['quantity'] = self.slots['sku'].map(self.sku_df.set_index('sku_id')['init_qty'])

        self.slots[['slot_id', 'sku', 'quantity']].to_csv("inventory_day0.csv", index=False)
        self.slots[['slot_id', 'x', 'y', 'z', 'sku']].to_csv("assigned_slots.csv", index=False)
        logger.info("库存分配完成")

    def generate_orders(self):
        logger.info("生成订单数据...")
        orders_list = []
        t = 0.0
        idx = 0
        while t < self.config['SIM_TIME']:
            t += self.rng.exponential(1 / self.config['ORDER_LAMBDA'])
            if t >= self.config['SIM_TIME']:
                break

            n_lines = self.rng.integers(1, 6)
            skus = self.rng.zipf(a=self.config['ZIPF_A'], size=n_lines) % self.config['N_SKU']
            sla_sec = self.rng.choice([30 * 60, 60 * 60, 2 * 60 * 60], p=[0.3, 0.5, 0.2])
            eq_flag = int(self.rng.random() < self.config['FORKLIFT_RATIO'])

            orders_list.append({
                'order_id': f"O{idx:05d}",
                'create_ts': t,
                'sku_list': list(map(int, skus)),
                'eq_flag': eq_flag,
                'sla_sec': int(sla_sec),
                'priority': self.rng.choice(['high', 'normal', 'low'], p=[0.2, 0.6, 0.2]),
            })
            idx += 1

        self.orders = pd.DataFrame(orders_list)
        logger.info(f"生成 {len(self.orders)} 个订单")

    # ------------------------------- Assignment ------------------------------
    def nested_logit_assignment(self):
        assert self.slots is not None and self.orders is not None
        logger.info("执行Nested-Logit任务分配...")

        # Spatial clusters (optional; used to infer pickup location)
        coords = self.slots[['x', 'y']].values
        clusters = KMeans(n_clusters=self.config['N_CLUSTERS'], random_state=self.config['RANDOM_SEED']).fit(coords)
        self.slots['cluster'] = clusters.labels_

        # Attach pickup location and cluster per order (based on main sku)
        self.orders['pickup_location'] = None
        self.orders['cluster_id'] = 0
        for idx, order in self.orders.iterrows():
            main_sku = order['sku_list'][0] if order['sku_list'] else 0
            sku_slots = self.slots[self.slots['sku'] == main_sku]
            if not sku_slots.empty:
                slot = sku_slots.iloc[0]
                self.orders.at[idx, 'pickup_location'] = (int(slot['x']), int(slot['y']))
                self.orders.at[idx, 'cluster_id'] = int(slot['cluster'])
            else:
                self.orders.at[idx, 'pickup_location'] = (
                    int(self.rng.integers(0, self.config['GRID_X'])),
                    int(self.rng.integers(0, self.config['GRID_Y'])),
                )
                self.orders.at[idx, 'cluster_id'] = int(self.rng.integers(0, self.config['N_CLUSTERS']))

        # Pickers and states
        pickers: List[Dict] = []
        picker_states: Dict[str, Dict] = {}
        for i in range(self.config['N_PICKERS']):
            pid = f"P{i}"
            picker = {
                'id': pid,
                'has_forklift': self.rng.random() < 0.3,
                'max_capacity': self.rng.integers(10, 20),
                'skill_level': self.rng.uniform(0.5, 1.5),
            }
            pickers.append(picker)
            picker_states[pid] = {
                'position': (0, 0),
                'current_workload': 0,
                'max_workload': 10,
                'assigned_orders': [],
            }

        nl = NestedLogitModel(config={'params': {
            'lambda': 0.6, 'w_distance': -0.4, 'w_workload': -0.5, 'w_priority': 1.2, 'w_equipment': 3.0, 'w_time_pressure': 1.5,
        }})

        self.assign = nl.assign_orders(self.orders, pickers, picker_states)
        if 'picked_flag' not in self.assign.columns and len(self.assign) > 0:
            self.assign['picked_flag'] = 0

        logger.info(f"使用Nested Logit模型分配了 {len(self.assign)} 个任务")

    # ------------------------------- Simulation ------------------------------
    def calculate_rop(self, sku_id: int) -> int:
        assert self.sku_df is not None
        sku_info = self.sku_df[self.sku_df['sku_id'] == sku_id].iloc[0]
        daily_demand = sku_info['daily_demand']
        lead_time_days = self.config['LEAD_TIME_MEAN'] / (24 * 60 * 60)
        safety_stock = daily_demand * self.config['ROP_SAFETY_DAYS']
        rop = daily_demand * lead_time_days + safety_stock
        return int(rop)

    def run_simulation(self):
        assert self.assign is not None and self.slots is not None
        logger.info("开始运行仿真...")

        G = nx.grid_2d_graph(self.config['GRID_X'], self.config['GRID_Y'])
        env = simpy.Environment()

        inv = self.slots.set_index('slot_id')['quantity'].to_dict()
        pick_log: List[Dict] = []

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def replenishment_process(env, inv):
            while True:
                yield env.timeout(60 * 60)
                for slot_id, quantity in list(inv.items()):
                    slot_info = self.slots[self.slots['slot_id'] == slot_id].iloc[0]
                    sku_id = int(slot_info['sku'])
                    rop = self.calculate_rop(sku_id)
                    if quantity < rop:
                        replenish_qty = int(rop * self.config['REPLENISH_QTY_FACTOR'] - quantity)
                        lead_time = max(0, self.rng.normal(self.config['LEAD_TIME_MEAN'], self.config['LEAD_TIME_STD']))
                        yield env.timeout(lead_time)
                        inv[slot_id] = inv.get(slot_id, 0) + replenish_qty
                        self.replenishment_log.append({
                            'slot_id': slot_id,
                            'sku_id': sku_id,
                            'trigger_ts': env.now - lead_time,
                            'complete_ts': env.now,
                            'quantity': replenish_qty,
                            'rop': rop,
                            'stock_before': quantity,
                            'stock_after': inv[slot_id],
                        })

        def picker_process(env, pid, job_q):
            while True:
                if job_q:
                    oid = job_q.pop(0)
                    order = self.orders[self.orders.order_id == oid].iloc[0]
                    # Build route by nearest slot holding SKU (ignoring stockouts for brevity)
                    route = []
                    for sku in order.sku_list:
                        possible = self.slots[(self.slots.sku == sku) & (self.slots.slot_id.map(lambda x: inv.get(x, 0) > 0))]
                        if not possible.empty:
                            sid = int(possible.sample(1, random_state=self.rng).iloc[0]['slot_id'])
                            route.append(sid)
                    current_pos = (0, 0)
                    for sid in route:
                        slot = self.slots[self.slots.slot_id == sid].iloc[0]
                        target = (int(slot['x']), int(slot['y']))
                        try:
                            path = nx.astar_path(G, current_pos, target, heuristic=heuristic)
                            for node in path[1:]:
                                yield env.timeout(self.config['WALK_SPEED'])
                                pick_log.append({'picker_id': pid, 'step_ts': env.now, 'x': node[0], 'y': node[1], 'load': 0, 'action': 'walk'})
                            yield env.timeout(self.config['PICK_TIME'])
                            if inv.get(sid, 0) > 0:
                                inv[sid] -= 1
                                pick_log.append({'picker_id': pid, 'step_ts': env.now, 'x': target[0], 'y': target[1], 'load': 1, 'action': 'pick'})
                            current_pos = target
                        except nx.NetworkXNoPath:
                            logger.warning(f"无法找到从 {current_pos} 到 {target} 的路径")
                    yield env.timeout(self.config['LOAD_TIME'])
                    pick_log.append({'picker_id': pid, 'step_ts': env.now, 'x': current_pos[0], 'y': current_pos[1], 'load': 0, 'action': 'load'})
                else:
                    yield env.timeout(1)

        # Build job queues
        job_queues = {f"P{i}": [] for i in range(self.config['N_PICKERS'])}
        for _, a in self.assign.iterrows():
            job_queues[str(a.picker_id)].append(str(a.order_id))

        # Start processes
        for pid, q in job_queues.items():
            env.process(picker_process(env, pid, q))
        env.process(replenishment_process(env, inv))

        env.run(until=self.config['SIM_TIME'])

        # Save picks and final inv
        self.picks = pd.DataFrame(pick_log)
        self.slots['quantity'] = self.slots['slot_id'].map(inv)
        logger.info("仿真完成")

    # ------------------------------- Output ----------------------------------
    def save_results(self):
        logger.info("保存仿真结果...")
        assert self.orders is not None and self.slots is not None and self.assign is not None
        self.orders.to_csv("orders.csv", index=False)
        self.assign.to_csv("assign.csv", index=False)
        if self.picks is not None and not self.picks.empty:
            self.picks[['picker_id', 'step_ts', 'x', 'y', 'load']].to_csv("picks.csv", index=False)
        self.slots[['slot_id', 'sku', 'quantity']].to_csv("inventory_end.csv", index=False)
        if self.replenishment_log:
            pd.DataFrame(self.replenishment_log).to_csv("replenishment_log.csv", index=False)
        logger.info("所有数据已保存")

    def run(self):
        start_time = datetime.now()
        logger.info(f"开始仓库仿真 - {start_time}")
        self.generate_slots()
        self.generate_sku_master()
        self.initial_inventory_allocation()
        self.generate_orders()
        self.nested_logit_assignment()
        self.run_simulation()
        self.save_results()
        end_time = datetime.now()
        logger.info(f"仿真完成 - 用时: {(end_time - start_time).total_seconds():.2f}秒")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Unified Warehouse System")
    sub = p.add_subparsers(dest='cmd', required=True)

    run = sub.add_parser('run-sim', help='Run full simulation and export CSVs')
    run.add_argument('--sim-time', type=int, default=8*60*60, help='Simulation time (sec)')
    run.add_argument('--n-pickers', type=int, default=6, help='Number of pickers')
    run.add_argument('--grid-x', type=int, default=32)
    run.add_argument('--grid-y', type=int, default=20)
    run.add_argument('--grid-z', type=int, default=3)
    run.add_argument('--order-lambda', type=float, default=2/60, help='Orders per second')
    run.add_argument('--seed', type=int, default=42)

    return p.parse_args()


def main():
    args = _parse_args()
    if args.cmd == 'run-sim':
        sim = WarehouseSimulation(config={
            'SIM_TIME': args.sim_time,
            'N_PICKERS': args.n_pickers,
            'GRID_X': args.grid_x,
            'GRID_Y': args.grid_y,
            'GRID_Z': args.grid_z,
            'ORDER_LAMBDA': args.order_lambda,
            'RANDOM_SEED': args.seed,
        })
        sim.run()


if __name__ == '__main__':
    main()


