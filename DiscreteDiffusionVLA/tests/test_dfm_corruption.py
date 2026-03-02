import torch

from prismatic.extern.hf.modeling_prismatic import PrismaticForConditionalGeneration


def test_mixture_mask_frac_same_matches_kappa():
    torch.manual_seed(0)
    B, L = 512, 64
    loss_mask_full = torch.ones(B, L, dtype=torch.bool)
    kappa_t = torch.full((B,), 0.7)

    masked = PrismaticForConditionalGeneration._sample_mixture_mask(loss_mask_full, kappa_t)
    frac_same = 1.0 - masked.float().mean().item()
    assert abs(frac_same - 0.7) < 0.03
