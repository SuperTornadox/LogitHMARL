import torch


INF = 1e9


def to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


@torch.no_grad()
def distance_transform_4conn(obstacle: torch.Tensor,
                             sources: torch.Tensor,
                             max_iters: int = 1024) -> torch.Tensor:
    """4-connected distance transform via iterative relaxation on GPU.

    Args:
      obstacle: (H,W) bool tensor; True means blocked cell.
      sources:  (H,W) bool tensor; True means source cells (distance 0).
      max_iters: max relaxation iterations; upper bound ~ H+W.

    Returns:
      dist: (H,W) float32 tensor; INF on obstacles.
    """
    assert obstacle.ndim == 2 and sources.ndim == 2 and obstacle.shape == sources.shape
    device = obstacle.device
    H, W = obstacle.shape
    dist = torch.full((H, W), INF, dtype=torch.float32, device=device)
    dist = torch.where(sources, torch.zeros_like(dist), dist)
    # Keep obstacles INF at all times
    dist = torch.where(obstacle, torch.full_like(dist, INF), dist)

    def shift(x, dy, dx, fill):
        pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))  # (left,right,top,bottom)
        y0 = max(-dy, 0); y1 = y0 + H
        x0 = max(-dx, 0); x1 = x0 + W
        xpad = torch.nn.functional.pad(x, pad, value=fill)
        return xpad[y0:y1, x0:x1]

    for _ in range(int(max_iters)):
        up = shift(dist, -1, 0, INF) + 1.0
        dn = shift(dist, 1, 0, INF) + 1.0
        lf = shift(dist, 0, -1, INF) + 1.0
        rt = shift(dist, 0, 1, INF) + 1.0
        newdist = torch.minimum(dist, torch.minimum(torch.minimum(up, dn), torch.minimum(lf, rt)))
        # Re-enforce constraints
        newdist = torch.where(obstacle, torch.full_like(newdist, INF), newdist)
        newdist = torch.where(sources, torch.zeros_like(newdist), newdist)
        if torch.equal(newdist, dist):
            break
        dist = newdist
    return dist


@torch.no_grad()
def argmin4_next_action(dist_map: torch.Tensor,
                        picker_xy: torch.Tensor,
                        obstacle: torch.Tensor) -> torch.Tensor:
    """Pick greedy action (0:UP,1:DOWN,2:LEFT,3:RIGHT,4:IDLE) that reduces distance.

    Args:
      dist_map: (H,W) float tensor distances; INF for obstacles.
      picker_xy: (N,2) int tensor of (x,y) per picker.
      obstacle: (H,W) bool tensor.

    Returns:
      actions: (N,) int64 tensor in {0..4}.
    """
    device = dist_map.device
    H, W = dist_map.shape
    N = picker_xy.shape[0]
    # Gather distance at neighbors
    x = picker_xy[:, 0].clamp(0, W - 1)
    y = picker_xy[:, 1].clamp(0, H - 1)

    def sample(nx, ny):
        nx = nx.clamp(0, W - 1)
        ny = ny.clamp(0, H - 1)
        return dist_map[ny, nx]

    up_d = sample(x, y - 1)
    dn_d = sample(x, y + 1)
    lf_d = sample(x - 1, y)
    rt_d = sample(x + 1, y)
    cur_d = sample(x, y)
    cand = torch.stack([up_d, dn_d, lf_d, rt_d, cur_d], dim=1)  # (N,5)
    # Mask moves into obstacles: set cost to INF
    def is_obs(nx, ny):
        nx = nx.clamp(0, W - 1)
        ny = ny.clamp(0, H - 1)
        return obstacle[ny, nx]
    mask_obs = torch.stack([
        is_obs(x, y - 1),
        is_obs(x, y + 1),
        is_obs(x - 1, y),
        is_obs(x + 1, y),
        torch.zeros_like(is_obs(x, y)),  # current cell not used
    ], dim=1)
    cand = torch.where(mask_obs, torch.full_like(cand, INF), cand)
    # Choose argmin among moves that strictly reduce distance; else pick IDLE
    # Build a big penalty for non-improving moves
    improve = cand[:, :4] < (cur_d.unsqueeze(1) - 1e-6)
    penalized = torch.where(improve, cand[:, :4], torch.full_like(cand[:, :4], INF))
    best4 = torch.argmin(penalized, dim=1)  # 0..3
    best4_val = torch.gather(penalized, 1, best4.view(-1, 1)).squeeze(1)
    # Use IDLE (4) if no improving neighbor
    actions = torch.where(best4_val >= INF, torch.full((N,), 4, dtype=torch.long, device=device), best4)
    return actions

