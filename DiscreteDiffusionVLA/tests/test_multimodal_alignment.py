import torch

from prismatic.extern.hf.modeling_prismatic import PrismaticForConditionalGeneration


def test_multimodal_token_id_alignment():
    token_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    patch_embeds = torch.zeros(2, 3, 8)
    mm_ids = PrismaticForConditionalGeneration._build_multimodal_token_ids(
        token_ids, patch_embeds, patch_fill_id=0
    )
    assert mm_ids.shape == (2, 8)
    assert torch.equal(mm_ids[:, 0], token_ids[:, 0])
    assert torch.equal(mm_ids[:, 1:4], torch.zeros(2, 3, dtype=mm_ids.dtype))
    assert torch.equal(mm_ids[:, 4:], token_ids[:, 1:])


def test_multimodal_mask_alignment():
    mask = torch.tensor([[True, False, True, False, True], [False, True, False, True, False]])
    patch_embeds = torch.zeros(2, 2, 8)
    mm_mask = PrismaticForConditionalGeneration._expand_mask_with_patches(mask, patch_embeds)
    assert mm_mask.shape == (2, 7)
    assert torch.equal(mm_mask[:, 0], mask[:, 0])
    assert torch.equal(mm_mask[:, 1:3], torch.zeros(2, 2, dtype=mm_mask.dtype))
    assert torch.equal(mm_mask[:, 3:], mask[:, 1:])
