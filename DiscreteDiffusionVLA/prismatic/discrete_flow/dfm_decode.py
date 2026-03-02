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
    maskgit_num_steps: int = 12,
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
    debug_level: int = 0,
    decode_mode: str = "ctmc",
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

    n_action_positions = int((~clamp_mask).sum().item())
    n_masked_initial = int(((init_ids == mask_token_id) & (~clamp_mask)).sum().item())
    if decode_mode == "maskgit" and n_action_positions > 0 and n_masked_initial != n_action_positions:
        raise RuntimeError(
            f"MaskGIT init mismatch: masked={n_masked_initial}, free={n_action_positions}. "
            "Action span or masking is incorrect."
        )

    if clamp_values is not None:
        clamp_values = clamp_values.to(device)
        cur = torch.where(clamp_mask, clamp_values, cur)

    if decode_mode not in ("ctmc", "maskgit"):
        raise ValueError(f"Unknown decode_mode: {decode_mode}")

    if decode_mode == "maskgit":
        if maskgit_num_steps <= 0:
            raise ValueError(f"maskgit_num_steps must be > 0, got {maskgit_num_steps}")
        t_grid = torch.linspace(0.0, 1.0, maskgit_num_steps, device=device)
        dt_grid = torch.zeros_like(t_grid)
    else:
        t_grid, dt_grid = time_grid(num_steps, eps=time_eps, device=device)

    actions_hidden_states = None
    num_changed_per_step = []
    step_masked_count = []
    mask_len_per_step = []
    debug_p_update = []
    debug_top1_prob = []
    debug_unresolved = []
    dt_safe_hits = 0
    dt_under_min = 0
    early_exit_iter = -1

    unknown_init = None
    if decode_mode == "maskgit":
        unknown_init = ((init_ids == mask_token_id) & (~clamp_mask)).sum(dim=1)

    for step, t in enumerate(t_grid):
        # Exit if no unresolved positions remain
        unresolved = (cur == mask_token_id) & (~clamp_mask)
        if debug_level >= 1:
            unresolved_count = int(unresolved.sum().item())
            debug_unresolved.append(unresolved_count)
            step_masked_count.append(unresolved_count)
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

        if decode_mode == "ctmc":
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
                safe_cap = (
                    (1.0 - kappa_t) / kdot_t.clamp(min=1e-8)
                    if adaptive_step
                    else torch.tensor(step_min, device=device)
                )
                if (step_min <= remaining_time) and (step_min <= safe_cap):
                    h = torch.tensor(step_min, device=device)
                else:
                    dt_under_min += 1

            # Update probability for CTMC jump
            p_update = 1.0 - torch.exp(-h * hazard)
            p_update = p_update.clamp(min=0.0, max=1.0)
            if debug_level >= 2:
                debug_p_update.append(
                    {
                        "mean": float(p_update.item()),
                        "min": float(p_update.item()),
                        "max": float(p_update.item()),
                    }
                )
                debug_unresolved.append(int(unresolved.sum().item()))
                top1_probs = probs.max(dim=-1).values
                if unresolved.any():
                    debug_top1_prob.append(float(top1_probs[unresolved].mean().item()))
                else:
                    debug_top1_prob.append(0.0)
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
        else:
            # MaskGIT-style refinement driven by kappa(t) schedule (diffusion-like)
            kappa_t = kappa(t, schedule=schedule)
            mask_ratio = (1.0 - kappa_t).clamp(min=0.0, max=1.0)
            unresolved_count = unresolved.sum(dim=1)
            total_unknown = unknown_init.to(unresolved_count.device)
            mask_len = torch.round(total_unknown.float() * mask_ratio).long()
            mask_len = torch.clamp(mask_len, min=0, max=total_unknown)
            if debug_level >= 1:
                mask_len_per_step.append(mask_len.detach().cpu().tolist())

            if debug_level >= 2:
                debug_unresolved.append(int(unresolved.sum().item()))
                top1_probs = probs.max(dim=-1).values
                if unresolved.any():
                    debug_top1_prob.append(float(top1_probs[unresolved].mean().item()))
                else:
                    debug_top1_prob.append(0.0)

            prev_cur = cur
            proposal = torch.where(unresolved, sampled_flat, cur)
            conf_all = probs.gather(2, proposal.unsqueeze(-1)).squeeze(-1)
            inf = torch.tensor(float("inf"), device=conf_all.device)
            conf_all = torch.where(clamp_mask, inf, conf_all)

            masking = torch.zeros_like(unresolved)
            if (total_unknown > 0).any():
                # Rows where mask_len == 0 -> fully resolve (all False)
                zero_rows = (mask_len == 0)
                # Rows where mask_len >= total_unknown -> keep all action positions masked
                full_rows = (mask_len >= total_unknown) & (total_unknown > 0)
                if full_rows.any():
                    masking = torch.where(full_rows.unsqueeze(1), ~clamp_mask, masking)
                # Remaining rows -> MaskGIT-style random top-k over confidence
                mid_rows = (~zero_rows) & (~full_rows) & (total_unknown > 0)
                if mid_rows.any():
                    mask_len_for_fn = torch.clamp(mask_len, min=1, max=conf_all.shape[1] - 1)
                    masking_all = parallel_decode.mask_by_random_topk(conf_all, mask_len_for_fn, temperature=1.0)
                    masking = torch.where(mid_rows.unsqueeze(1), masking_all, masking)

            cur = proposal
            cur = torch.where(masking, mask_token_id, cur)
            if clamp_values is not None:
                cur = torch.where(clamp_mask, clamp_values, cur)

            num_changed = (cur != prev_cur).sum().item()
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

    # Final fill: force-resolve any remaining masks in non-clamped positions
    unresolved = (cur == mask_token_id) & (~clamp_mask)
    if unresolved.any():
        logits, actions_hidden_states = tokens_to_logits(cur)
        logits = logits.float()
        temp = temperature
        if temperature_anneal == "linear":
            temp = 1.0
        elif temperature_anneal not in ("none", None):
            raise ValueError(f"Unknown temperature_anneal: {temperature_anneal}")
        if temp != 1.0:
            logits = logits / temp
        # Never sample mask token if it is within the logits vocabulary range
        if 0 <= mask_token_id < logits.size(-1):
            logits[..., mask_token_id] = -1e9
        probs = F.softmax(logits, dim=-1)
        flat_probs = probs.view(-1, probs.size(-1))
        sampled_flat = torch.multinomial(flat_probs, 1).view(cur.shape)
        cur = torch.where(unresolved, sampled_flat, cur)
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
        "dfm_step_changed_count": num_changed_per_step,
        "dfm_mask_frac_final": dfm_mask_frac_final,
        "dfm_unresolved_final": dfm_unresolved_final,
        "dfm_decode_mode": decode_mode,
        "dfm_n_action_positions": n_action_positions,
        "dfm_n_masked_initial": n_masked_initial,
        "dfm_step_masked_count": step_masked_count,
        "dfm_maskgit_num_steps": maskgit_num_steps if decode_mode == "maskgit" else None,
    }
    if debug_level >= 1:
        stats["dfm_unresolved_count"] = debug_unresolved
        if decode_mode == "maskgit":
            stats["dfm_mask_len_per_step"] = mask_len_per_step
    if debug_level >= 2:
        stats["dfm_p_update"] = debug_p_update
        stats["dfm_top1_prob_mean"] = debug_top1_prob

    return cur, actions_hidden_states, stats
