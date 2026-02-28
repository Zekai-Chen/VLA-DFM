import torch

from prismatic.extern.hf.modeling_prismatic import PrismaticForConditionalGeneration


def test_dfm_generalized_kl_matches_manual():
    logits = torch.tensor([[[2.0, 0.0, -1.0]]], requires_grad=True)  # (B=1, T-1=1, V=3)
    x1 = torch.tensor([[0]])
    xt = torch.tensor([[2]])  # mask token
    action_mask = torch.tensor([[True]])
    kappa_t = torch.tensor([0.2])
    kdot_t = torch.tensor([0.5])

    loss = PrismaticForConditionalGeneration._dfm_generalized_kl_loss(
        shift_logits=logits,
        x1=x1,
        xt=xt,
        action_mask=action_mask,
        kappa_t=kappa_t,
        kdot_t=kdot_t,
        action_begin=0,
        action_end=2,
        mask_id=2,
        weight_clip=20.0,
    )

    with torch.no_grad():
        log_p = torch.log_softmax(logits, dim=-1)
        log_p_x1 = log_p[0, 0, 0]
        log_p_xt = log_p[0, 0, 2]
        p_xt = torch.exp(log_p_xt)
        w = kdot_t / (1.0 - kappa_t)
        expected = -w * (p_xt + log_p_x1)

    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
