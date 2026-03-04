import torch
import torch.nn.functional as F
import math

from .mask_schedule import schedule as mask_schedule


def mask_by_random_topk(
    probs: torch.Tensor,
    mask_len: torch.Tensor,
    temperature: float = 1.0,
    eligible: torch.Tensor | None = None,
) -> torch.BoolTensor:
    """
    PyTorch 版 mask_by_random_topk:
    probs: [B, L], 每个位置被采样到的概率
    mask_len: [B], 每个样本要 mask 的数量
    返回 mask 布尔矩阵 [B, L]
    """
    # 1) 采样 Gumbel 噪声
    gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-20) + 1e-20)
    # 2) 计算 confidence 分数
    confidence = torch.log(probs + 1e-20) + temperature * gumbel  # [B, L]
    if eligible is not None:
        inf = torch.tensor(float("inf"), device=probs.device)
        confidence = torch.where(eligible, confidence, inf)
    # 3) 对每行找第 k 小阈值
    sorted_conf, _ = confidence.sort(dim=1)  # [B, L]
    B, L = probs.shape
    k = mask_len.clamp(min=1, max=L-1)      # [B]
    batch_idx = torch.arange(B, device=probs.device)
    threshold = sorted_conf[batch_idx, k]  # [B]

    # 4) 低于阈值的置为 True（继续被 mask）
    mask = confidence < threshold.unsqueeze(1)  # [B, L]
    if eligible is not None:
        mask = mask & eligible
    return mask


def top_k_logits(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.size(-1))
    val, idx = logits.topk(k, dim=-1)
    mask = torch.full_like(logits, float('-inf')).to(device=logits.device)
    mask.scatter_(-1, idx, val)
    return mask


def decode(
    init_ids: torch.LongTensor,             # [B, L], 初始序列（含 mask_token_id）
    tokens_to_logits,                       # fn(seq_ids: [B, L]) -> logits [B, L, V]
    mask_token_id: int,
    start_iter: int = 0,
    num_iter: int = 12,
    choice_temperature: float = 1.0,
    mask_scheduling_method="cosine",
    use_remask: bool = False,               # 是否使用重 mask 概率
    token_critic: torch.nn.Module = None,  # TokenCritic 模型，输出 [B, L] 的分数
    critic_noise_scale: float = 1.0,       # Critic 加噪声比例
    clamp_mask: torch.Tensor | None = None,
    clamp_values: torch.Tensor | None = None,
    return_stats: bool = False,
):
    """
    非自回归 MaskGIT 推理
    返回 final_seqs: [B, num_iter, L]，每轮迭代的 sampled 序列
    """
    B, L = init_ids.shape
    device = init_ids.device

    if clamp_mask is None:
        clamp_mask = torch.zeros_like(init_ids, dtype=torch.bool, device=device)

    # 记录初始未知（mask）数量，用于调度（仅对可更新位置）
    unknown_init = ((init_ids == mask_token_id) & (~clamp_mask)).sum(dim=1)  # [B]

    # State init
    cur_seqs = init_ids.clone()                         # [B, L]
    if clamp_values is not None:
        clamp_values = clamp_values.to(device)
        if clamp_values.dim() == 1:
            clamp_values = clamp_values.unsqueeze(0).expand(init_ids.size(0), -1)
        cur_seqs = torch.where(clamp_mask, clamp_values, cur_seqs)
    final_seqs = torch.zeros(B, num_iter, L, dtype=init_ids.dtype, device=device)
    final_seqs[:, 0, :] = init_ids

    step_masked_count = []
    step_changed_count = []
    mask_len_per_step = []

    # 迭代解码
    for step in range(start_iter, num_iter):
        prev_cur = cur_seqs
        # 1) 得到 logits & 概率分布
        logits, actions_hidden_states = tokens_to_logits(cur_seqs)             # [B, L, V]
        probs = F.softmax(logits, dim=-1)               # [B, L, V]

        # 2) 并行 categorical 采样
        #    展平后采样，再 reshape
        flat_probs = probs.view(-1, probs.size(-1))     # [B*L, V]
        sampled_flat = torch.multinomial(flat_probs, 1)  # [B*L, 1]
        sampled = sampled_flat.view(B, L)               # [B, L]

        # 3) 仅在 mask 位置更新
        unknown_map = (cur_seqs == mask_token_id) & (~clamp_mask)  # [B, L]
        sampled = torch.where(unknown_map, sampled, cur_seqs)
        if clamp_values is not None:
            sampled = torch.where(clamp_mask, clamp_values, sampled)

        # 4) 计算下轮 mask 数量
        ratio = torch.tensor(float(step + 1) / num_iter, device=device)              # scalar
        mask_len = mask_schedule(ratio, unknown_init, mask_scheduling_method)
        if mask_len.dtype.is_floating_point:
            if torch.max(mask_len).item() <= 1.0 + 1e-6:
                mask_len = torch.round(unknown_init.float() * mask_len)
            else:
                mask_len = torch.round(mask_len)
        if mask_len.dim() == 0:
            mask_len = mask_len.expand_as(unknown_init)
        mask_len = mask_len.to(dtype=torch.long, device=device)
        mask_len = torch.clamp(mask_len, min=0)
        mask_len = torch.minimum(mask_len, unknown_init)
        if step == (num_iter - 1):
            mask_len = torch.zeros_like(mask_len)
        mask_len_per_step.append(mask_len.detach().cpu().tolist())

        # —————————————— 4) 计算每个位置得分 scores ——————————————
        if token_critic is not None:
            # 用 Critic 得分 + 加噪声
            # 假设 token_critic 返回 [B, L] raw scores
            raw_crit = token_critic(actions_hidden_states)              # [B, L]
            scores = - raw_crit
            # 加均匀噪声，看 step 越大噪声越小
            scores = scores + (torch.rand_like(scores).cuda() - 0.5) * critic_noise_scale * (1.0 - ratio)
            selected_probs = scores
        else:
            # 5) 计算每个位置被选中的概率：probs.gather
            selected_probs = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)  # [B, L]

        if use_remask:
            # 6) 引入“重 mask 概率”
            #    p_remask 从 1 线性降到 0：早期更容易重新 mask，后期更稳定
            p_remask = 1.0 - ratio
            #    对已知位置（~unknown_map）降低它们的置信度
            selected_probs = torch.where(
                unknown_map,
                selected_probs,
                selected_probs * p_remask
            )  # [B, L]
        else:
            # 已知位置（初始非 mask）设为 1.0，降低下轮 mask 概率
            selected_probs = torch.where(unknown_map, selected_probs, torch.ones_like(selected_probs))

        # clamp positions should never be masked
        selected_probs = torch.where(clamp_mask, torch.ones_like(selected_probs), selected_probs)

        # 6) 用 Gumbel+top-k 策略决定下轮仍要被 mask 的位置
        masking = torch.zeros_like(unknown_map)
        zero_rows = mask_len == 0
        full_rows = (mask_len >= unknown_init) & (unknown_init > 0)
        if full_rows.any():
            masking = torch.where(full_rows.unsqueeze(1), ~clamp_mask, masking)
        mid_rows = (~zero_rows) & (~full_rows) & (unknown_init > 0)
        if mid_rows.any():
            mask_len_for_fn = torch.clamp(mask_len, min=1, max=selected_probs.shape[1] - 1)
            masking_all = mask_by_random_topk(
                selected_probs,
                mask_len_for_fn,
                temperature=choice_temperature,
                eligible=~clamp_mask,
            )                                               # [B, L]
            masking = torch.where(mid_rows.unsqueeze(1), masking_all, masking)

        # 7) 构造 next seqs：被 mask 的位置继续用 mask_token
        next_seqs = torch.where(masking, mask_token_id, sampled)  # [B, L]
        if clamp_values is not None:
            next_seqs = torch.where(clamp_mask, clamp_values, next_seqs)
        cur_seqs = next_seqs

        # 8) 存储本轮结果
        final_seqs[:, step, :] = sampled
        step_masked_count.append(int(((cur_seqs == mask_token_id) & (~clamp_mask)).sum().item()))
        step_changed_count.append(int((cur_seqs != prev_cur).sum().item()))

        # TODO: Important 选最后一轮 final_iters[:, -1, :] 作为最终输出, 或对多轮结果做融合

    if return_stats:
        stats = {
            "mask_len_per_step": mask_len_per_step,
            "step_masked_count": step_masked_count,
            "step_changed_count": step_changed_count,
            "nfe_realized": len(step_changed_count),
        }
        return final_seqs, actions_hidden_states, stats
    return final_seqs, actions_hidden_states


def legacy_decode(
    init_ids: torch.LongTensor,             # [B, L], 初始序列（含 mask_token_id）
    tokens_to_logits,                       # fn(seq_ids: [B, L]) -> logits [B, L, V]
    mask_token_id: int,
    start_iter: int = 0,
    num_iter: int = 12,
    choice_temperature: float = 1.0,
    mask_scheduling_method="cosine",
    use_remask: bool = False,               # 是否使用重 mask 概率
    token_critic: torch.nn.Module = None,  # TokenCritic 模型，输出 [B, L] 的分数
    critic_noise_scale: float = 1.0,       # Critic 加噪声比例
):
    """
    Legacy MaskGIT decode (exactly matches 2026-02-27 branch).
    返回 final_seqs: [B, num_iter, L]，每轮迭代的 sampled 序列
    """
    B, L = init_ids.shape
    device = init_ids.device

    # 记录初始未知（mask）数量，用于调度
    unknown_init = (init_ids == mask_token_id).sum(dim=1)  # [B]

    # State init
    cur_seqs = init_ids.clone()                         # [B, L]
    final_seqs = torch.zeros(B, num_iter, L, dtype=init_ids.dtype, device=device)
    final_seqs[:, 0, :] = init_ids

    # 迭代解码
    for step in range(start_iter, num_iter):
        # 1) 得到 logits & 概率分布
        logits, actions_hidden_states = tokens_to_logits(cur_seqs)             # [B, L, V]
        probs = F.softmax(logits, dim=-1)               # [B, L, V]

        # 2) 并行 categorical 采样
        #    展平后采样，再 reshape
        flat_probs = probs.view(-1, probs.size(-1))     # [B*L, V]
        sampled_flat = torch.multinomial(flat_probs, 1)  # [B*L, 1]
        sampled = sampled_flat.view(B, L)               # [B, L]

        # 3) 仅在 mask 位置更新
        unknown_map = cur_seqs == mask_token_id         # [B, L]
        sampled = torch.where(unknown_map, sampled, cur_seqs)

        # 4) 计算下轮 mask 数量
        ratio = torch.tensor(float(step + 1) / num_iter, device=device)              # scalar
        # 调度函数：给定 ratio、初始未知数，返回 mask_ratio
        mask_ratio = mask_schedule(ratio, unknown_init, mask_scheduling_method)  # [B]
        mask_len = torch.floor(unknown_init.float() * mask_ratio).long()
        # 保证至少 1 且最多 unknown_init-1
        mask_len = torch.clamp(mask_len, min=1, max=(unknown_init - 1).item())

        # —————————————— 4) 计算每个位置得分 scores ——————————————
        if token_critic is not None:
            # 用 Critic 得分 + 加噪声
            # 假设 token_critic 返回 [B, L] raw scores
            raw_crit = token_critic(actions_hidden_states)              # [B, L]
            scores = - raw_crit
            # 加均匀噪声，看 step 越大噪声越小
            scores = scores + (torch.rand_like(scores).cuda() - 0.5) * critic_noise_scale * (1.0 - ratio)
            selected_probs = scores
        else:
            # 5) 计算每个位置被选中的概率：probs.gather
            selected_probs = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)  # [B, L]

        if use_remask:
            # 6) 引入“重 mask 概率”
            #    p_remask 从 1 线性降到 0：早期更容易重新 mask，后期更稳定
            p_remask = 1.0 - ratio
            #    对已知位置（~unknown_map）降低它们的置信度
            selected_probs = torch.where(
                unknown_map,
                selected_probs,
                selected_probs * p_remask
            )  # [B, L]
        else:
            # 已知位置（初始非 mask）设为极大值，避开下轮 mask
            inf = torch.tensor(float("inf"), device=device)
            selected_probs = torch.where(unknown_map, selected_probs, inf)

        # 6) 用 Gumbel+top-k 策略决定下轮仍要被 mask 的位置
        masking = mask_by_random_topk(
            selected_probs,
            mask_len,
            temperature=choice_temperature * (1.0 - ratio),
        )                                               # [B, L]

        # 7) 构造 next seqs：被 mask 的位置继续用 mask_token
        next_seqs = torch.where(masking, mask_token_id, sampled)  # [B, L]
        cur_seqs = next_seqs

        # 8) 存储本轮结果
        final_seqs[:, step, :] = sampled

        # TODO: Important 选最后一轮 final_iters[:, -1, :] 作为最终输出, 或对多轮结果做融合

    return final_seqs, actions_hidden_states
