"""Utils for training/fine-tuning scripts."""

import torch

from prismatic.vla.constants import ACTION_DIM


def get_current_action_mask(token_ids, action_begin: int, action_end: int):
    is_action = (token_ids >= action_begin) & (token_ids < action_end)
    action_cumsum = torch.cumsum(is_action, dim=1)
    mask = (1 <= action_cumsum) & (action_cumsum <= ACTION_DIM)
    return is_action & mask


def get_next_actions_mask(token_ids, action_begin: int, action_end: int):
    is_action = (token_ids >= action_begin) & (token_ids < action_end)
    action_cumsum = torch.cumsum(is_action, dim=1)
    mask = action_cumsum > ACTION_DIM
    return is_action & mask


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss
