"""CTMC-based discrete flow matching decoder (hazard/tau-leaping)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from .dfm_schedule import kappa, kappa_dot, time_grid
from . import parallel_decode


@torch.no_grad()
def dfm_decode(
    init_ids: torch.LongTensor,                       # [B, L]
    tokens_to_logits: Callable[[torch.LongTensor], Tuple[torch.Tensor, torch.Tensor]],
    mask_token_id: int,
    num_steps: int = 12,
    schedule: str = "cosine",
    temperature: float = 1.0,
    temperature_anneal: str = "none",
    adaptive_step: bool = True,
    step_min: float = 1e-4,
    step_max: float = 0.2,
    time_eps: float = 1e-3,
    early_exit: bool = True,
    early_exit_frac: float = 0.0,
    corrector: bool = False,
    corrector_iters: int = 1,
    corrector_remask_frac: float = 0.1,
    clamp_mask: Optional[torch.BoolTensor] = None,    # True => do not update these positions
    clamp_values: Optional[torch.LongTensor] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, dict]:
    """Run CTMC hazard/tau-leaping updates for discrete flow matching.

    Returns:
        final_ids: [B, L]
        actions_hidden_states: [B, L, D] from last model forward
    """
    if init_ids.dim() != 2:
        raise ValueError("init_ids must have shape [B, L]")

    device = init_ids.device
    cur = init_ids.clone()

    if clamp_mask is None:
        clamp_mask = torch.zeros_like(cur, dtype=torch.bool, device=device)

    if clamp_values is not None:
        clamp_values = clamp_values.to(device)
        cur = torch.where(clamp_mask, clamp_values, cur)

    t_grid, dt_grid = time_grid(num_steps, eps=time_eps, device=device)

    actions_hidden_states = None
    num_changed_per_step = []
    dt_safe_hits = 0
    dt_under_min = 0
    early_exit_iter = -1

    for step, t in enumerate(t_grid):
        # Exit if no unresolved positions remain
        unresolved = (cur == mask_token_id) & (~clamp_mask)
        if early_exit and unresolved.sum().item() == 0:
            early_exit_iter = step
            break

        logits, actions_hidden_states = tokens_to_logits(cur)
        # Optional temperature anneal
        temp = temperature
        if temperature_anneal == "linear":
            temp = temperature * (1.0 - t) + 1.0 * t
        elif temperature_anneal not in ("none", None):
            raise ValueError(f"Unknown temperature_anneal: {temperature_anneal}")

        logits = logits.float()
        if temp != 1.0:
            logits = logits / temp
        # Never sample mask token if it is within the logits vocabulary range
        if 0 <= mask_token_id < logits.size(-1):
            logits[..., mask_token_id] = -1e9
        probs = F.softmax(logits, dim=-1)

        # Sample x1_i from posterior per position
        flat_probs = probs.view(-1, probs.size(-1))
        sampled_flat = torch.multinomial(flat_probs, 1).view(cur.shape)

        # Hazard rate (scalar per step for mixture path)
        kappa_t = kappa(t, schedule=schedule)
        kdot_t = kappa_dot(t, schedule=schedule)
        denom = (1.0 - kappa_t).clamp(min=1e-8)
        hazard = (kdot_t / denom).clamp(min=0.0)

        # Step size with safety precedence
        remaining_time = (1.0 - time_eps) - t
        h = dt_grid[step]
        if adaptive_step:
            safe_h = (1.0 - kappa_t) / kdot_t.clamp(min=1e-8)
            h = torch.minimum(h, safe_h)
            if (safe_h <= h).item():
                dt_safe_hits += 1
        h = torch.minimum(h, torch.tensor(step_max, device=device))
        h = torch.minimum(h, remaining_time)
        if h < step_min:
            safe_cap = (1.0 - kappa_t) / kdot_t.clamp(min=1e-8) if adaptive_step else torch.tensor(step_min, device=device)
            if (step_min <= remaining_time) and (step_min <= safe_cap):
                h = torch.tensor(step_min, device=device)
            else:
                dt_under_min += 1

        # Update probability for CTMC jump
        p_update = 1.0 - torch.exp(-h * hazard)
        p_update = p_update.clamp(min=0.0, max=1.0)
        # Broadcast to [B, L]
        update_mask = torch.rand_like(cur.float()) < p_update
        update_mask = update_mask & (sampled_flat != cur) & (~clamp_mask)
        # Only update unresolved (masked) positions to match training corruption
        update_mask = update_mask & unresolved

        cur = torch.where(update_mask, sampled_flat, cur)
        if clamp_values is not None:
            cur = torch.where(clamp_mask, clamp_values, cur)

        num_changed = update_mask.sum().item()
        num_changed_per_step.append(num_changed)

        if early_exit and early_exit_frac > 0.0:
            num_free = (~clamp_mask).sum().item()
            if num_free > 0 and (num_changed / num_free) < early_exit_frac:
                early_exit_iter = step
                break

    if corrector:
        # Simple remask-corrector: re-mask lowest-confidence tokens and run a short MaskGIT pass
        for _ in range(corrector_iters):
            if corrector_remask_frac <= 0.0:
                break
            logits, actions_hidden_states = tokens_to_logits(cur)
            logits = logits.float()
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values  # [B, L]

            B, L = confidence.shape
            k = max(1, int(corrector_remask_frac * L))
            # mask lowest-confidence positions per batch
            sorted_conf, _ = confidence.sort(dim=1)
            threshold = sorted_conf[:, k - 1].unsqueeze(1)
            mask = confidence <= threshold
            mask = mask & (~clamp_mask)

            init_ids = torch.where(mask, mask_token_id, cur)

            if schedule.startswith("poly"):
                exponent = schedule.replace("poly", "")
                mask_schedule_method = f"pow{exponent}"
            else:
                mask_schedule_method = schedule
            corrected, actions_hidden_states = parallel_decode.decode(
                init_ids=init_ids,
                tokens_to_logits=tokens_to_logits,
                mask_token_id=mask_token_id,
                num_iter=2,
                choice_temperature=temperature,
                mask_scheduling_method=mask_schedule_method,
                use_remask=True,
            )
            cur = corrected[:, -1, :]
            if clamp_values is not None:
                cur = torch.where(clamp_mask, clamp_values, cur)

    dfm_mask_frac_final = (cur == mask_token_id).float().mean().item()
    dfm_unresolved_final = ((cur == mask_token_id) & (~clamp_mask)).sum().item()

    stats = {
        "dfm_nfe_realized": len(num_changed_per_step),
        "dfm_early_exit_iter": early_exit_iter,
        "dfm_dt_safe_hits": dt_safe_hits,
        "dfm_dt_under_min": dt_under_min,
        "dfm_num_changed_tokens": num_changed_per_step,
        "dfm_mask_frac_final": dfm_mask_frac_final,
        "dfm_unresolved_final": dfm_unresolved_final,
    }

    return cur, actions_hidden_states, stats
