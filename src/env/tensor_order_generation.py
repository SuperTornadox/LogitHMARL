import torch
from typing import Dict


class TensorOrderGeneratorTorch:
    """Simplified nonhomogeneous Poisson order generator implemented with PyTorch.

    This is a lightweight GPU-friendly replacement for sampling arrivals per step.
    """

    def __init__(self, config: Dict, device: str = 'cpu'):
        self.device = torch.device(device)
        self.base_rate = float(config.get('base_rate', 60.0))  # orders/hour
        self.peak_hours = config.get('peak_hours', [(9, 12), (14, 17), (19, 21)])
        self.peak_multiplier = float(config.get('peak_multiplier', 1.6))
        self.off_peak_multiplier = float(config.get('off_peak_multiplier', 0.7))
        self.simulation_hours = float(config.get('simulation_hours', 24.0))
        self.n_skus = int(config.get('n_skus', 1000))

        # Simple SKU zone mapping (torch tensors)
        zones = torch.randint(0, 4, (self.n_skus,), device=self.device)
        self.sku_zone = zones

    def _rate_at(self, hour: float) -> float:
        # Peak hours boost
        for s, e in self.peak_hours:
            if s <= hour % 24 < e:
                return self.base_rate * self.peak_multiplier
        if hour % 24 >= 22 or hour % 24 < 6:
            return self.base_rate * self.off_peak_multiplier * 0.5
        return self.base_rate

    @torch.no_grad()
    def sample_window(self, t0_h: float, dt_h: float, width: int, height: int, n_stations: int) -> Dict[str, torch.Tensor]:
        """Sample orders in [t0, t0+dt] on GPU.

        Returns dictionary of tensors (may be empty), fields:
          'shelf_xy': (K,2) long, 'station_xy': (K,2) long, 'priority': (K,), 'deadline': (K,)
        """
        lam = self._rate_at(t0_h) * dt_h  # expected orders in this window
        # Poisson via torch.poisson over rate 1 replay trick
        lam_t = torch.tensor([lam], device=self.device)
        k = torch.poisson(lam_t).to(torch.long).item()
        if k <= 0:
            return {
                'shelf_xy': torch.empty((0, 2), dtype=torch.long, device=self.device),
                'station_xy': torch.empty((0, 2), dtype=torch.long, device=self.device),
                'priority': torch.empty((0,), dtype=torch.float32, device=self.device),
                'deadline': torch.empty((0,), dtype=torch.float32, device=self.device),
            }
        # Random shelves and stations (uniform for simplicity)
        # Here we only return (x,y) pairs; zone/weight/value can be extended similarly.
        sx = torch.randint(0, width, (k,), device=self.device)
        sy = torch.randint(0, height, (k,), device=self.device)
        stx = torch.randint(0, width, (k,), device=self.device)
        sty = torch.randint(0, height, (k,), device=self.device)
        pr = torch.rand((k,), device=self.device) * 0.6 + 0.4  # [0.4,1.0)
        # deadlines ~ 0.1..0.5 hours after arrival
        dl = torch.rand((k,), device=self.device) * 0.4 + 0.1
        return {
            'shelf_xy': torch.stack([sx, sy], dim=1),
            'station_xy': torch.stack([stx, sty], dim=1),
            'priority': pr,
            'deadline': t0_h + dl,
        }

