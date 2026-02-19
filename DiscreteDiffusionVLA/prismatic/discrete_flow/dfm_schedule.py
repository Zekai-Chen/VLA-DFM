"""Discrete Flow Matching schedules for kappa(t) and kappa_dot(t)."""

import math
from typing import Optional, Tuple

import torch


def kappa(t: torch.Tensor, schedule: str = "cosine") -> torch.Tensor:
    """Compute kappa(t) for a given schedule. t in [0, 1]."""
    if schedule == "cosine":
        return 1.0 - torch.cos(0.5 * math.pi * t)
    if schedule == "linear":
        return t
    if schedule == "poly2":
        return t ** 2
    raise ValueError(f"Unknown DFM schedule: {schedule}")


def kappa_dot(t: torch.Tensor, schedule: str = "cosine") -> torch.Tensor:
    """Compute derivative of kappa(t)."""
    if schedule == "cosine":
        return 0.5 * math.pi * torch.sin(0.5 * math.pi * t)
    if schedule == "linear":
        return torch.ones_like(t)
    if schedule == "poly2":
        return 2.0 * t
    raise ValueError(f"Unknown DFM schedule: {schedule}")


def time_grid(num_steps: int, eps: float = 1e-3, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (t, dt) grid with t in (eps, 1-eps)."""
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    edges = torch.linspace(0.0, 1.0 - eps, num_steps + 1, device=device)
    t = edges[:-1]
    dt = edges[1:] - edges[:-1]
    return t, dt
