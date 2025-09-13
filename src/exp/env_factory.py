from env.dynamic_warehouse_env import DynamicWarehouseEnv


def create_test_env(width, height, n_pickers, n_shelves, n_stations, order_rate, max_items):
    """根据统一配置创建动态环境（适配当前 DynamicWarehouseEnv 接口）。"""
    cfg = {
        'width': width,
        'height': height,
        'n_pickers': n_pickers,
        'n_shelves': n_shelves,
        'n_stations': n_stations,
        'n_charging_pads': 1,
        'levels_per_shelf': 3,
        'time_step': 2.0,
        'order_config': {
            'base_rate': order_rate,
            'peak_hours': [(9, 12), (14, 17)],
            'peak_multiplier': 1.6,
            'off_peak_multiplier': 0.7,
            'simulation_hours': 2,
        },
    }
    return DynamicWarehouseEnv(cfg)

