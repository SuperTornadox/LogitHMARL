import torch
from typing import Tuple

from env.tensor_utils import distance_transform_4conn, argmin4_next_action


@torch.no_grad()
def gpu_smart_navigate_batch(grid: torch.Tensor,
                             picker_xy: torch.Tensor,
                             target_xy: torch.Tensor) -> torch.Tensor:
    """GPU batch navigation helper for the existing env.

    Args:
      grid: (H,W) int tensor (0 free, 2 shelf)
      picker_xy: (N,2) int tensor (x,y)
      target_xy: (N,2) int tensor (x,y) target per picker

    Returns:
      actions: (N,) long in {0..4}
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = grid.to(device)
    H, W = g.shape
    obstacle = g.eq(2)
    src = torch.zeros_like(obstacle)
    valid = (target_xy[:, 0] >= 0) & (target_xy[:, 1] >= 0)
    if valid.any():
        xs = target_xy[valid, 0].clamp(0, W - 1).to(device)
        ys = target_xy[valid, 1].clamp(0, H - 1).to(device)
        src[ys, xs] = True
    dist = distance_transform_4conn(obstacle, src, max_iters=H + W)
    actions = argmin4_next_action(dist, picker_xy.to(device), obstacle)
    return actions.cpu()

