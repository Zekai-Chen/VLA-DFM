"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions.
Inherits from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained,
but exactly replicate the logic in `prismatic.models.vlms.prismatic.py`.
"""

import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from prismatic.discrete_flow import (
    dfm_decode,
    kappa,
    kappa_dot,
    mask_schedule,
    parallel_decode,
)
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# Set up logger
logger = logging.getLogger(__name__)


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    """
    Vision backbone for Prismatic models that handles image feature extraction.

    Supports both single backbone (e.g., SigLIP) and fused backbone (e.g., SigLIP + DINOv2) configurations.
    For fused backbones, features from both models are concatenated along the feature dimension.
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        """
        Initialize the vision backbone.

        Args:
            use_fused_vision_backbone: Whether to use two backbones and fuse their features
            image_sizes: List of image sizes for each backbone
            timm_model_ids: List of TIMM model IDs to use for each backbone
            timm_override_act_layers: List of activation layer overrides for each backbone
        """
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1  # Default value, can be overridden later

        # Validate number of (fused) vision backbones
        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic models only support up to 2 (fused) vision backbones!")

        # Create primary featurizer
        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        # Create secondary featurizer if using fused backbone
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch LayerScale modules for HF compatibility
        self._patch_layer_scales()

    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        """
        Create a TIMM-based featurizer model with appropriate configurations.

        Args:
            model_id: The TIMM model ID to load
            img_size: Input image size for the model
            act_layer: Override for the activation layer type

        Returns:
            A configured featurizer model
        """
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        # Monkey-patch the forward function to extract the second-to-last layer features
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer

    def _patch_layer_scales(self) -> None:
        """
        Patch all LayerScale modules to be compatible with HF's parameter naming.

        HF Transformers overwrites parameters with names containing 'gamma',
        so we need to rename and modify the forward method.
        """
        # Patch primary featurizer
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        # Patch secondary featurizer if it exists
        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def get_num_patches(self) -> int:
        """
        Returns the number of vision patches output by the vision backbone.

        Returns:
            Number of patches per image
        """
        return self.featurizer.patch_embed.num_patches

    def get_num_images_in_input(self) -> int:
        """
        Returns the number of input images for the vision backbone.

        Returns:
            Number of images expected in the input
        """
        return self.num_images_in_input

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """
        Sets the number of input images for the vision backbone.

        Args:
            num_images_in_input: Number of images to expect in the input
        """
        self.num_images_in_input = num_images_in_input

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone.

        If `self.use_fused_vision_backbone == True`, uses both SigLIP and DINOv2 transformers to extract visual features
        (otherwise uses SigLIP only). Allows multi-image inputs (but only for fused vision backbone).

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
        """
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            return torch.cat([patches, patches_fused], dim=2)

        else:
            assert self.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)

            # Process each image and collect patches
            all_patches = []
            for img in images:
                # Split each image further into two stacks of channels (each with 3 channels)
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # Get patches from both SigLIP and DINOv2 vision transformers
                patches = self.featurizer(img_regular)
                patches_fused = self.fused_featurizer(img_fused)

                # Concatenate SigLIP and DINOv2 patches along the hidden dimension
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            # Concatenate all patches along the patch dimension
            return torch.cat(all_patches, dim=1)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None

    # Additions for Discrete Diffusion
    labels: Optional[torch.LongTensor] = None

    # Additions for DFM logging
    dfm_stats: Optional[Dict[str, torch.FloatTensor]] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        # config.text_config.vocab_size自行向上取整到64的整数倍, 32064, 还有空间,无需调整
        # self.language_model.resize_token_embeddings(config.text_config.vocab_size)

        self.use_discrete_diffusion = config.use_discrete_diffusion
        self.use_discrete_flow_matching = getattr(config, "use_discrete_flow_matching", False)
        self.mask_token_id = config.mask_token_id
        self.action_vocab_anchor = getattr(config, "action_vocab_anchor", "pad")

        if self.use_discrete_diffusion and self.use_discrete_flow_matching:
            raise ValueError("Cannot enable both discrete diffusion and discrete flow matching.")

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size

        # Validate action vocab range configuration (pad anchor only).
        self._validate_action_vocab()

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        """
        Replace embeddings in input_embeddings at positions where all_actions_mask is True
        with embeddings from noisy_action_features, using vectorized operations.

        Args:
            input_embeddings: Tensor of shape (B, S, D)
            all_actions_mask: Boolean tensor of shape (B, S)
            noisy_action_features: Tensor of shape (B, K, D) where K is the number of True values in mask per sample

        Returns:
            Modified input_embeddings tensor
        """
        # Clone input to avoid modifying the original tensor
        new_input_embeddings = input_embeddings.clone()

        # Create a tensor with the same shape of input_embeddings to hold the noisy action features
        repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

        # Create batch indices for splicing
        batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

        # Get indices where mask is True for each sample
        masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

        # Move the noisy action features into their correct positions
        repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features

        # Combine original input embeddings and noisy action embeddings using the mask
        new_input_embeddings = torch.where(
            all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
        )

        return new_input_embeddings

    def _action_vocab_range(self) -> Tuple[int, int, int]:
        """Return (action_begin, action_end, n_bins) for current config."""
        n_bins = getattr(self.config, "n_action_bins", None)
        if n_bins is None and hasattr(self, "bin_centers"):
            n_bins = int(self.bin_centers.shape[0])
        if n_bins is None:
            raise ValueError("n_action_bins must be set on config or via bin_centers.")
        n_bins = int(n_bins)
        anchor = getattr(self.config, "action_vocab_anchor", "pad")
        if anchor == "pad":
            action_end = int(self.pad_token_id)
        elif anchor == "vocab_size":
            action_end = int(self.vocab_size)
        else:
            raise ValueError(f"Unknown action_vocab_anchor: {anchor}")
        action_begin = int(action_end - n_bins)
        return action_begin, action_end, n_bins

    def _validate_action_vocab(self) -> None:
        """Validate that special tokens do not overlap action bins."""
        if not hasattr(self.config, "n_action_bins") and not hasattr(self, "bin_centers"):
            return
        action_begin, action_end, _ = self._action_vocab_range()
        if action_begin < 0:
            raise ValueError(f"Action vocab begin ({action_begin}) is negative; check n_action_bins/pad_token_id.")
        anchor = getattr(self.config, "action_vocab_anchor", "pad")
        if anchor == "pad":
            if action_begin <= self.pad_token_id < action_end:
                raise ValueError(
                    f"pad_token_id ({self.pad_token_id}) overlaps action range [{action_begin}, {action_end})."
                )
            if action_begin <= self.mask_token_id < action_end:
                raise ValueError(
                    f"mask_token_id ({self.mask_token_id}) overlaps action range [{action_begin}, {action_end})."
                )

    def _process_action_masks(self, labels):
        """Helper to get action masks from labels"""
        action_begin, action_end, _ = self._action_vocab_range()
        current_action_mask = get_current_action_mask(labels, action_begin, action_end)
        next_actions_mask = get_next_actions_mask(labels, action_begin, action_end)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    def _compute_language_embeddings(self, input_embeddings, attention_mask, all_actions_mask):
        """Compute a safe language embedding summary for FiLM conditioning."""
        if attention_mask is None:
            attention_mask = input_embeddings.new_ones(input_embeddings.shape[:2], dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        language_mask = attention_mask & (~all_actions_mask)
        if not torch.any(language_mask):
            return input_embeddings.new_zeros((input_embeddings.shape[0], 1, input_embeddings.shape[2]))

        denom = language_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        summed = (input_embeddings * language_mask.unsqueeze(-1)).sum(dim=1)
        return (summed / denom).unsqueeze(1)

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        """Process vision features with optional FiLM conditioning"""
        if use_film:
            # FiLM: Infuse language inputs into visual features
            patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
        else:
            patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

        # Project patch embeddings into language embedding space
        return self.projector(patch_features)

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """Process proprioceptive features and append to vision features"""
        if proprio_projector is not None and proprio is not None:
            # projected_patch_embeddings: (bsz, num_patches * num_images, llm_dim)
            # proprio: (bsz, proprio_dim) or (propro_dim,)
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            # For simplicity, just append proprio token to the end of projected vision patch tokens
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        """Build multimodal embeddings and attention mask"""
        # Update attention mask
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )

        return multimodal_embeddings, multimodal_attention_mask

    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        """Build multimodal labels with IGNORE_INDEX for patch embeddings"""
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None

    @staticmethod
    def _build_multimodal_token_ids(token_ids, projected_patch_embeddings, patch_fill_id: int = 0):
        """Build multimodal token IDs with patch fill tokens inserted after BOS."""
        if token_ids is None:
            return None
        patch_len = projected_patch_embeddings.shape[1] if projected_patch_embeddings is not None else 0
        if patch_len > 0:
            patch_ids = torch.full(
                (token_ids.shape[0], patch_len),
                fill_value=patch_fill_id,
                dtype=token_ids.dtype,
                device=token_ids.device,
            )
            return torch.cat([token_ids[:, :1], patch_ids, token_ids[:, 1:]], dim=1)
        return token_ids

    @staticmethod
    def _expand_mask_with_patches(mask, projected_patch_embeddings):
        """Expand a (B, L) mask to include patch positions after BOS."""
        if mask is None:
            return None
        patch_len = projected_patch_embeddings.shape[1] if projected_patch_embeddings is not None else 0
        if patch_len > 0:
            patch_mask = torch.zeros(
                (mask.shape[0], patch_len), device=mask.device, dtype=mask.dtype
            )
            return torch.cat([mask[:, :1], patch_mask, mask[:, 1:]], dim=1)
        return mask

    def _get_eos_pos(self, all_actions_mask):
        """Prepare loss mask for discrete diffusion"""
        loss_mask_full = all_actions_mask.clone()  # (B, seq_len)

        # 找到每个序列中最后一个 True 的下标，并把下一个位置也置为 True
        batch_size, seq_len = loss_mask_full.shape
        # 将序列反向，再 argmax 就能找到“最后一个 True”在原序列中的位置
        last_true = seq_len - 1 - torch.argmax(
            loss_mask_full.flip(dims=[1]).int(), dim=1
        )  # (B,)

        # 下一个位置
        next_pos = last_true + 1  # (B,)
        # 只保留那些 next_pos < seq_len 的有效索引
        valid = next_pos < seq_len  # (B,) 布尔向量
        batch_idx = torch.arange(batch_size, device=loss_mask_full.device)[valid]

        # 把 EOS token 位置也设为 True
        loss_mask_full[batch_idx, next_pos[valid]] = True
        return next_pos[valid]

    def apply_mask_diffusion(
        self,
        input_ids: torch.LongTensor,                 # (B, L)
        input_embeddings: torch.Tensor,              # (B, L, D)
        labels: torch.LongTensor,                    #  (B, L), target input_ids (有-100)
        loss_mask_full: torch.BoolTensor,            # (B, L), True 表示可 mask 的位置（非 padding、非语言 token）
        mask_token_id: int,
        eos_pos: Optional[torch.LongTensor] = None,  # (B,), 可选的 EOS token 位置（如果有）
        no_mask_token_prob: float = 0.0,             # Optional probability to “unmask” 已 mask 的一部分
    ):
        """
        输入：
        - input_ids: 原始 token id 序列
        - input_embeddings: 原始 embedding 序列
        - loss_mask_full: 全局可 mask 位掩码（包括动作 token 位置）
        - mask_token_id: 用于填充的 special mask token id
        - no_mask_token_prob: 可选概率，把已 mask 掉的位置再随机 unmask
        返回：
        - masked_input_ids: 用 mask_token_id 替代被 mask 掉的位置的 input_ids
        - labels: 原 input_ids，在非被 mask 位置用 -100 屏蔽（CrossEntropyLoss 忽略）
        - new_input_embeddings: 对应替换了 action token 的新 embeddings
        - loss_mask: float mask，用于后续 loss 加权（1 表示预测该位置，0 表示忽略）
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 1) 计算每个样本总共可 mask 的 token 数量
        #    total_unknown = loss_mask_full.sum(dim=1)  # (B,)
        total_unknown = loss_mask_full.float().sum(dim=1)  # (B,)

        # 2) 随机采一个 time ratio in [0,1)
        rand_time = torch.rand(B, device=device)

        # 3) 根据 schedule 计算 mask ratio、再算出每个样本要 mask 的 token 数
        #    mask_ratios: tensor (B,), 取值 in (0,1]
        mask_ratios = mask_schedule(rand_time, total_unknown, method="cosine")  # [B]
        #    num_mask: at least 1
        num_mask = torch.clamp((total_unknown * mask_ratios).round(), min=1).long()  # [B]

        # 4) 为每个位置打随机分数，非可-mask 位置打上大数，保证它永远不被选中
        #    vals: (B, L) ~ Uniform(0,1)
        vals = torch.rand(B, L, device=device)
        #    large = 1e8
        large = float('inf')
        #    只有 loss_mask_full==True 的位置保留原分数，其他位置加大
        vals = torch.where(loss_mask_full, vals, vals + large)  # inf 表示不可选

        # 5) 按行排序、取前 num_mask
        perm = vals.argsort(dim=1)                     # (B, L)
        ranks = perm.argsort(dim=1)
        # masked_mask: bool (B, L)，True 表示该位置被 mask 掉
        masked_mask = ranks < num_mask[:, None]        # (B, L), 先 top-k mask

        # 6) 可选：no_mask_token_prob，再次随机取消一部分已 mask 的位置
        if no_mask_token_prob > 0:
            # 从 masked_mask 中随机抽取一部分不再 mask
            # 生成同形状的 [0,1) 随机数
            prob = torch.rand(B, L, device=device)
            # 在已经 mask 的位置上，若 prob < no_mask_token_prob 则 unmask
            unmask = (prob < no_mask_token_prob) & masked_mask
            masked_mask = masked_mask & (~unmask)

        # # Set to True in eos_pos
        # if eos_pos is not None:
        #     # eos_pos: (B,), 只在这些位置上 mask 掉
        #     eos_mask = torch.zeros_like(masked_mask, dtype=torch.bool, device=device)
        #     eos_mask[torch.arange(B, device=device), eos_pos] = True
        #     masked_mask = masked_mask | eos_mask

        # 7) 构造 labels: 被 mask 掉的位置保留原 id，其他位置设为 -100
        ignore_labels = torch.full_like(labels, fill_value=IGNORE_INDEX, dtype=labels.dtype, device=device)
        masked_labels = torch.where(masked_mask, labels, ignore_labels)

        masked_input_ids = torch.where(masked_mask, mask_token_id, input_ids)
        # 8) 构造新的 input_embeddings: masked 位置替换成 用 embedding lookup 或直接替换
        #    假设你后面是用 inputs_embeds，所以直接对 embeddings 替换
        #    masked_input_embeddings: (B, L, D)
        masked_input_embeddings = input_embeddings.clone()
        #    获取 mask token embedding
        mask_emb = self.get_input_embeddings()(torch.tensor([mask_token_id], device=device))  # (1, D)
        #    扩展到 (B, L, D)
        mask_emb = mask_emb.view(1, 1, -1).expand(B, L, -1)
        #    替换
        masked_input_embeddings = torch.where(masked_mask.unsqueeze(-1), mask_emb, masked_input_embeddings)

        # 10) 返回
        #     loss_mask: float 跟 JAX 里一致，用于后续加权 loss (1. for masked positions, 0. elsewhere)
        loss_mask = masked_mask.float()

        return masked_input_ids, masked_input_embeddings, masked_labels, loss_mask

    @staticmethod
    def _sample_mixture_mask(loss_mask_full: torch.BoolTensor, kappa_t: torch.Tensor) -> torch.BoolTensor:
        """Sample per-coordinate Bernoulli mask for mixture path corruption."""
        p_mask = (1.0 - kappa_t).view(-1, 1)  # (B, 1)
        rand = torch.rand(loss_mask_full.shape, device=loss_mask_full.device)
        return (rand < p_mask) & loss_mask_full

    def apply_mask_flow_matching(
        self,
        input_ids: torch.LongTensor,                 # (B, L)
        input_embeddings: torch.Tensor,              # (B, L, D)
        labels: torch.LongTensor,                    # (B, L), target input_ids (has -100)
        loss_mask_full: torch.BoolTensor,            # (B, L), True indicates maskable positions
        mask_token_id: int,
        schedule: str = "cosine",
        time_eps: float = 1e-3,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ):
        """
        Apply mask-only corruption following a DFM mixture path.
        Returns masked inputs/labels plus loss mask and kappa stats.
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Sample time t in [t_min, t_max], clamped to (eps, 1-eps)
        t_low = max(t_min, time_eps)
        t_high = min(t_max, 1.0 - time_eps)
        if t_high <= t_low:
            raise ValueError("Invalid DFM time range after applying eps clamp.")
        t = torch.rand(B, device=device) * (t_high - t_low) + t_low
        kappa_t = kappa(t, schedule=schedule)
        kdot_t = kappa_dot(t, schedule=schedule)

        masked_mask = self._sample_mixture_mask(loss_mask_full, kappa_t)

        ignore_labels = torch.full_like(labels, fill_value=IGNORE_INDEX, dtype=labels.dtype, device=device)
        masked_labels = torch.where(masked_mask, labels, ignore_labels)

        masked_input_ids = torch.where(masked_mask, mask_token_id, input_ids)

        masked_input_embeddings = input_embeddings.clone()
        mask_emb = self.get_input_embeddings()(torch.tensor([mask_token_id], device=device))
        mask_emb = mask_emb.view(1, 1, -1).expand(B, L, -1)
        masked_input_embeddings = torch.where(masked_mask.unsqueeze(-1), mask_emb, masked_input_embeddings)

        loss_mask = masked_mask.float()

        return masked_input_ids, masked_input_embeddings, masked_labels, loss_mask, kappa_t, kdot_t, t

    @staticmethod
    def _dfm_generalized_kl_loss(
        *,
        shift_logits: torch.Tensor,          # (B, T-1, V_full)
        x1: torch.Tensor,                    # (B, T-1)
        xt: torch.Tensor,                    # (B, T-1)
        action_mask: torch.Tensor,           # (B, T-1)
        kappa_t: torch.Tensor,               # (B,)
        kdot_t: torch.Tensor,                # (B,)
        action_begin: int,
        action_end: int,
        mask_id: int,
        weight_clip: float = 20.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Generalized-KL loss for mixture path on action coordinates."""
        B, _, _ = shift_logits.shape
        am = action_mask.bool()

        action_ids = torch.arange(action_begin, action_end, device=shift_logits.device)
        allowed_ids = torch.cat([action_ids, torch.tensor([mask_id], device=shift_logits.device)], dim=0)
        K = allowed_ids.numel()

        logits = shift_logits.index_select(dim=-1, index=allowed_ids)

        x1_safe = torch.where(am, x1, torch.full_like(x1, action_begin))
        xt_safe = torch.where(am, xt, torch.full_like(xt, action_begin))

        x1_idx = x1_safe - action_begin
        xt_idx = torch.where(xt_safe == mask_id, torch.full_like(xt_safe, K - 1), xt_safe - action_begin)

        log_p = torch.log_softmax(logits, dim=-1)
        log_p_x1 = log_p.gather(-1, x1_idx.unsqueeze(-1)).squeeze(-1)
        log_p_xt = log_p.gather(-1, xt_idx.unsqueeze(-1)).squeeze(-1)
        p_xt = torch.exp(log_p_xt)

        delta = (xt_safe == x1_safe).to(log_p.dtype)

        denom = (1.0 - kappa_t).clamp(min=eps)
        w = (kdot_t / denom).clamp(min=0.0, max=weight_clip).view(B, 1)

        loss_pos = -w * (p_xt - delta + (1.0 - delta) * log_p_x1)

        return (loss_pos * am).sum() / am.sum().clamp(min=1.0)

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        proprio=None,
        proprio_projector=None,
        noisy_actions=None,
        noisy_action_projector=None,
        diffusion_timestep_embeddings=None,
        use_film: bool = False,
        dfm_schedule: Optional[str] = None,
        dfm_time_eps: float = 1e-3,
        dfm_t_min: float = 0.0,
        dfm_t_max: float = 1.0,
        dfm_loss_mode: Optional[str] = None,
        dfm_weight_clip: float = 20.0,
        dfm_train_mode: Optional[str] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )  # True
        output_projector_features = output_projector_features if output_projector_features is not None else False  # False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # True

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None
        dfm_loss_mask = None
        dfm_weight = None
        multimodal_labels = None
        dfm_stats = None
        dfm_action_token_count = None
        dfm_t = None
        multimodal_x1_labels = None
        multimodal_xt_ids = None
        multimodal_actions_mask = None

        # Resolve DFM defaults
        dfm_schedule = dfm_schedule or getattr(self.config, "dfm_schedule", "cosine")
        dfm_loss_mode = dfm_loss_mode or getattr(self.config, "dfm_loss_mode", "generalized_kl")
        dfm_train_mode = dfm_train_mode or "flow"

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"

            # Get input embeddings (from language model embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

            # Extract action masks
            all_actions_mask = self._process_action_masks(labels)

            # Extract a safe language summary for FiLM conditioning
            language_embeddings = None
            if use_film:
                language_embeddings = self._compute_language_embeddings(
                    input_embeddings, attention_mask, all_actions_mask
                )  # (B, 1, llm_dim)

            # Get visual features  pixel_values [8, 12, 224, 224]
            projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

            # Add proprioceptive state if provided
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

            # [Diffusion] Add diffusion timestep embedding if provided
            if diffusion_timestep_embeddings is not None:
                # For simplicity, just append diffusion timestep embedding to the end of projected vision patch tokens
                projected_patch_embeddings = torch.cat(
                    (projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
                )

            # Process action embeddings
            dfm_loss_mask = None
            dfm_weight = None

            if noisy_actions is not None:
                # Get mask corresponding to all action tokens
                all_actions_mask = self._process_action_masks(labels)

                # Reshape noisy actions into individual action tokens
                # noisy_actions: (B, chunk_len, action_dim) -> (B, chunk_len * action_dim, 1)
                B = noisy_actions.shape[0]
                noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)

                # Project noisy action tokens into language model embedding space
                noisy_action_features = noisy_action_projector(noisy_actions)  # (B, chunk_len * action_dim, llm_dim)

                # Replace embeddings of the action tokens with noisy action embeddings
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, noisy_action_features
                )
            elif self.use_discrete_diffusion:
                # (Later on, the positional embeddings will be added to them)
                loss_mask_full = all_actions_mask  # (B, seq_len)
                eos_pos = self._get_eos_pos(all_actions_mask)  # (B,)
                input_ids, input_embeddings, labels, loss_mask = self.apply_mask_diffusion(
                    input_ids=input_ids,
                    input_embeddings=input_embeddings,
                    labels=labels,
                    loss_mask_full=loss_mask_full,
                    mask_token_id=self.mask_token_id,
                    no_mask_token_prob=0.0,
                )

                # Set labels to EOS_TOKEN in eos_pos
                labels[torch.arange(labels.shape[0]), eos_pos] = STOP_INDEX

            elif self.use_discrete_flow_matching:
                labels_x1 = labels.clone() if labels is not None else None
                loss_mask_full = all_actions_mask
                dfm_action_token_count = loss_mask_full.sum(dim=1)
                if dfm_train_mode == "diffusion_like":
                    eos_pos = self._get_eos_pos(all_actions_mask)  # (B,)
                    (
                        input_ids,
                        input_embeddings,
                        labels,
                        dfm_loss_mask,
                    ) = self.apply_mask_diffusion(
                        input_ids=input_ids,
                        input_embeddings=input_embeddings,
                        labels=labels,
                        loss_mask_full=loss_mask_full,
                        mask_token_id=self.mask_token_id,
                        no_mask_token_prob=0.0,
                    )
                    mask_ratio = dfm_loss_mask.sum(dim=1) / dfm_action_token_count.clamp(min=1.0)
                    kappa_t = (1.0 - mask_ratio).clamp(min=0.0, max=1.0)
                    kdot_t = torch.zeros_like(kappa_t)
                    dfm_weight = torch.ones_like(kappa_t)
                    dfm_t = None
                    dfm_loss_mode = "masked_ce"
                    # Supervise STOP/EOS to match discrete diffusion semantics.
                    labels[torch.arange(labels.shape[0]), eos_pos] = STOP_INDEX
                    dfm_loss_mask[torch.arange(dfm_loss_mask.shape[0]), eos_pos] = 1.0
                else:
                    (
                        input_ids,
                        input_embeddings,
                        labels,
                        dfm_loss_mask,
                        kappa_t,
                        kdot_t,
                        dfm_t,
                    ) = self.apply_mask_flow_matching(
                        input_ids=input_ids,
                        input_embeddings=input_embeddings,
                        labels=labels,
                        loss_mask_full=loss_mask_full,
                        mask_token_id=self.mask_token_id,
                        schedule=dfm_schedule,
                        time_eps=dfm_time_eps,
                        t_min=dfm_t_min,
                        t_max=dfm_t_max,
                    )
                    denom = (1.0 - kappa_t).clamp(min=1e-8)
                    dfm_weight = (kdot_t / denom).clamp(min=0.0, max=dfm_weight_clip)
                    mask_ratio = (1.0 - kappa_t).clamp(min=0.0, max=1.0)
                if os.environ.get("VLA_DFM_DEBUG", "0") == "1":
                    self._dfm_debug_step = getattr(self, "_dfm_debug_step", 0) + 1
                    debug_every = int(os.environ.get("VLA_DFM_DEBUG_EVERY", "200"))
                    if self._dfm_debug_step % debug_every == 0:
                        is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
                        if is_rank0:
                            with torch.no_grad():
                                supervised_mask = labels != IGNORE_INDEX
                                action_mask = loss_mask_full
                                supervised_in_action = supervised_mask & action_mask
                                supervised_outside_action = supervised_mask & (~action_mask)
                                action_count = action_mask.sum().item()
                                supervised_count = supervised_in_action.sum().item()
                                outside_count = supervised_outside_action.sum().item()
                                supervised_frac = supervised_count / max(action_count, 1)
                                per_batch_frac = (
                                    dfm_loss_mask.sum(dim=1) / dfm_action_token_count.clamp(min=1.0)
                                ).mean().item()
                                t_mean = dfm_t.mean().item() if dfm_t is not None else float("nan")
                                t_min = dfm_t.min().item() if dfm_t is not None else float("nan")
                                t_max = dfm_t.max().item() if dfm_t is not None else float("nan")
                                mr_mean = mask_ratio.mean().item()
                                mr_min = mask_ratio.min().item()
                                mr_max = mask_ratio.max().item()
                                w_mean = dfm_weight.mean().item()
                                w_min = dfm_weight.min().item()
                                w_max = dfm_weight.max().item()
                                logger.info(
                                    "[DFM DEBUG] supervised_action_tokens=%d/%d (%.4f) "
                                    "per_batch_supervised_frac=%.4f supervised_outside_action=%d "
                                    "dfm_loss_mask_sum=%d t_mean=%.4f t_min=%.4f t_max=%.4f "
                                    "mask_ratio_mean=%.4f mask_ratio_min=%.4f mask_ratio_max=%.4f "
                                    "w_mean=%.4f w_min=%.4f w_max=%.4f",
                                    supervised_count,
                                    action_count,
                                    supervised_frac,
                                    per_batch_frac,
                                    outside_count,
                                    dfm_loss_mask.sum().item(),
                                    t_mean,
                                    t_min,
                                    t_max,
                                    mr_mean,
                                    mr_min,
                                    mr_max,
                                    w_mean,
                                    w_min,
                                    w_max,
                                )
                                if supervised_count == 0:
                                    logger.warning(
                                        "[DFM DEBUG] No supervised action tokens (all -100). Check masking/labels."
                                    )
                                if outside_count > 0:
                                    logger.warning(
                                        "[DFM DEBUG] Found supervised tokens outside action mask. Check loss_mask_full/labels."
                                    )

            else:
                # Replace the embeddings of the action tokens with zeros
                # (Later on, the positional embeddings will be added to them)
                all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)  [8, 93, 1]
                input_embeddings = input_embeddings * ~all_actions_mask

            # Build multimodal embeddings & attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # Build labels for multimodal sequence if needed  # labels shape [8, 93]
            multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)
            if self.use_discrete_flow_matching and labels_x1 is not None:
                multimodal_x1_labels = self._build_multimodal_labels(labels_x1, projected_patch_embeddings)
                multimodal_xt_ids = self._build_multimodal_token_ids(
                    input_ids, projected_patch_embeddings, patch_fill_id=0
                )
                multimodal_actions_mask = self._expand_mask_with_patches(all_actions_mask, projected_patch_embeddings)
                if os.environ.get("VLA_DFM_DEBUG", "0") == "1":
                    if multimodal_x1_labels is not None and multimodal_xt_ids is not None:
                        assert (
                            multimodal_x1_labels.shape == multimodal_xt_ids.shape == multimodal_actions_mask.shape
                        ), "DFM multimodal tensors are misaligned."

            # Dispatch to language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        # Override loss for DFM if applicable
        lm_loss = language_model_output.loss
        if self.use_discrete_flow_matching and (dfm_loss_mask is not None):
            # Align loss mask with multimodal sequence (insert patch positions)
            patch_len = projected_patch_embeddings.shape[1] if projected_patch_embeddings is not None else 0
            if patch_len > 0:
                patch_mask = torch.zeros(
                    (dfm_loss_mask.shape[0], patch_len), device=dfm_loss_mask.device, dtype=dfm_loss_mask.dtype
                )
                multimodal_loss_mask = torch.cat([dfm_loss_mask[:, :1], patch_mask, dfm_loss_mask[:, 1:]], dim=1)
            else:
                multimodal_loss_mask = dfm_loss_mask

            skip_dfm_loss_override = dfm_train_mode == "diffusion_like"
            if not skip_dfm_loss_override:
                logits = language_model_output.logits
                # Shift for causal LM loss
                shift_logits = logits[:, :-1, :]
                if dfm_loss_mode == "generalized_kl":
                    if multimodal_x1_labels is None or multimodal_xt_ids is None or multimodal_actions_mask is None:
                        raise ValueError("Missing multimodal bookkeeping for generalized_kl loss.")
                    x1 = multimodal_x1_labels[:, 1:]
                    xt = multimodal_xt_ids[:, 1:]
                    am = multimodal_actions_mask[:, 1:]
                    action_begin, action_end, _ = self._action_vocab_range()
                    if action_begin <= self.mask_token_id < action_end:
                        raise ValueError(
                            "mask_token_id overlaps action vocab range; generalized_kl requires distinct mask token."
                        )
                    lm_loss = self._dfm_generalized_kl_loss(
                        shift_logits=shift_logits,
                        x1=x1,
                        xt=xt,
                        action_mask=am,
                        kappa_t=kappa_t,
                        kdot_t=kdot_t,
                        action_begin=action_begin,
                        action_end=action_end,
                        mask_id=self.mask_token_id,
                        weight_clip=dfm_weight_clip,
                    )
                else:
                    vocab_size = logits.shape[-1]
                    shift_labels = multimodal_labels[:, 1:]
                    shift_loss_mask = multimodal_loss_mask[:, 1:]

                    token_losses = torch.nn.functional.cross_entropy(
                        shift_logits.reshape(-1, vocab_size),
                        shift_labels.reshape(-1),
                        reduction="none",
                        ignore_index=IGNORE_INDEX,
                    ).view(shift_labels.shape)

                    if dfm_loss_mode == "masked_ce":
                        weight = torch.ones_like(shift_loss_mask)
                    else:
                        weight = dfm_weight.view(-1, 1).expand_as(shift_loss_mask)

                    masked_loss = token_losses * shift_loss_mask * weight
                    denom = (shift_loss_mask * weight).sum().clamp(min=1.0)
                    lm_loss = masked_loss.sum() / denom

            # DFM stats for logging (both diffusion_like + flow)
            with torch.no_grad():
                if dfm_action_token_count is None:
                    mask_frac = multimodal_loss_mask.sum(dim=1) / multimodal_loss_mask.shape[1]
                else:
                    mask_frac = dfm_loss_mask.sum(dim=1) / dfm_action_token_count.clamp(min=1.0)
                mask_ratio = (1.0 - kappa_t).clamp(min=0.0, max=1.0)
                dfm_stats = {
                    "kappa_mean": kappa_t.mean().detach(),
                    "t_mean": dfm_t.mean().detach() if dfm_t is not None else torch.tensor(float("nan")),
                    "t_min": dfm_t.min().detach() if dfm_t is not None else torch.tensor(float("nan")),
                    "t_max": dfm_t.max().detach() if dfm_t is not None else torch.tensor(float("nan")),
                    "mask_ratio_mean": mask_ratio.mean().detach(),
                    "mask_ratio_min": mask_ratio.min().detach(),
                    "mask_ratio_max": mask_ratio.max().detach(),
                    "mask_frac_mean": mask_frac.mean().detach(),
                    "w_mean": dfm_weight.mean().detach(),
                    "w_min": dfm_weight.min().detach(),
                    "w_max": dfm_weight.max().detach(),
                    "frac_w_clipped": (dfm_weight >= dfm_weight_clip).float().mean().detach(),
                    "num_supervised_tokens": dfm_loss_mask.sum().detach(),
                }
                if dfm_loss_mode == "generalized_kl" and multimodal_x1_labels is not None:
                    shift_logits = language_model_output.logits[:, :-1, :]
                    shift_x1 = multimodal_x1_labels[:, 1:]
                    shift_xt = multimodal_xt_ids[:, 1:]
                    shift_am = multimodal_actions_mask[:, 1:].bool()
                    action_begin, action_end, _ = self._action_vocab_range()
                    action_ids = torch.arange(action_begin, action_end, device=shift_logits.device)
                    allowed_ids = torch.cat(
                        [action_ids, torch.tensor([self.mask_token_id], device=shift_logits.device)], dim=0
                    )
                    K = allowed_ids.numel()
                    logits = shift_logits.index_select(dim=-1, index=allowed_ids)
                    log_p = torch.log_softmax(logits, dim=-1)
                    x1_idx = torch.where(
                        shift_am, shift_x1 - action_begin, torch.zeros_like(shift_x1)
                    )
                    xt_idx = torch.where(
                        shift_am,
                        torch.where(
                            shift_xt == self.mask_token_id,
                            torch.full_like(shift_xt, K - 1),
                            shift_xt - action_begin,
                        ),
                        torch.zeros_like(shift_xt),
                    )
                    log_p_x1 = log_p.gather(-1, x1_idx.unsqueeze(-1)).squeeze(-1)
                    log_p_xt = log_p.gather(-1, xt_idx.unsqueeze(-1)).squeeze(-1)
                    p_xt = torch.exp(log_p_xt)
                    masked_xt = shift_xt == self.mask_token_id
                    denom = shift_am.sum().clamp(min=1.0)
                    frac_same = ((shift_xt == shift_x1) & shift_am).sum().float() / denom
                    mean_logp_x1_masked = (
                        log_p_x1[shift_am & masked_xt].mean() if (shift_am & masked_xt).any() else torch.tensor(0.0)
                    )
                    mean_p_xt_masked = (
                        p_xt[shift_am & masked_xt].mean() if (shift_am & masked_xt).any() else torch.tensor(0.0)
                    )
                    jump_coeff = (kdot_t / (1.0 - kappa_t).clamp(min=1e-8)).clamp(max=dfm_weight_clip)
                    dfm_stats.update(
                        {
                            "frac_same": frac_same.detach(),
                            "jump_coeff_mean": jump_coeff.mean().detach(),
                            "jump_coeff_p95": torch.quantile(jump_coeff, 0.95).detach(),
                            "mean_logp_x1_masked": mean_logp_x1_masked.detach(),
                            "mean_p_xt_masked": mean_p_xt_masked.detach(),
                        }
                    )

            # Debug loss parity check (optional)
            if os.environ.get("VLA_DFM_DEBUG", "0") == "1":
                debug_every = int(os.environ.get("VLA_DFM_DEBUG_EVERY", "200"))
                debug_step = getattr(self, "_dfm_debug_step", 0)
                if debug_step % debug_every == 0:
                    is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
                    if is_rank0:
                        with torch.no_grad():
                            hf_loss = language_model_output.loss.detach()
                            if skip_dfm_loss_override:
                                # Compute debug-only DFM loss with shifted tensors (no grad)
                                logits = language_model_output.logits
                                vocab_size = logits.shape[-1]
                                shift_logits = logits[:, :-1, :]
                                shift_labels = multimodal_labels[:, 1:]
                                shift_loss_mask = multimodal_loss_mask[:, 1:]
                                token_losses = torch.nn.functional.cross_entropy(
                                    shift_logits.reshape(-1, vocab_size),
                                    shift_labels.reshape(-1),
                                    reduction="none",
                                    ignore_index=IGNORE_INDEX,
                                ).view(shift_labels.shape)
                                if dfm_loss_mode == "masked_ce":
                                    weight = torch.ones_like(shift_loss_mask)
                                else:
                                    weight = dfm_weight.view(-1, 1).expand_as(shift_loss_mask)
                                masked_loss = token_losses * shift_loss_mask * weight
                                denom = (shift_loss_mask * weight).sum().clamp(min=1.0)
                                dfm_loss_check = masked_loss.sum() / denom
                            else:
                                dfm_loss_check = lm_loss.detach()
                            logger.info(
                                "[DFM DEBUG] hf_loss=%.6f dfm_loss_check=%.6f skip_dfm_loss_override=%s",
                                hf_loss.item(),
                                dfm_loss_check.item(),
                                str(skip_dfm_loss_override),
                            )

        return PrismaticCausalLMOutputWithPast(
            loss=lm_loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
            labels=labels if labels is not None else None,
            dfm_stats=dfm_stats,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # check if config has topk_filter_thres
        if hasattr(config, "topk_filter_thres"):
            self.topk_filter_thres = config.topk_filter_thres
        else:
            self.topk_filter_thres = 0.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        action_begin, _, _ = self._action_vocab_range()
        ARBITRARY_ACTION_TOKEN_IDX = action_begin
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions

    def _run_diffusion_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        noise,
        action_head,
        projected_patch_embeddings,
        labels,
        attention_mask,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        noisy_action_projector,
    ):
        """Run diffusion-based action prediction"""
        # Clone embedding for reuse in each timestep
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        curr_noisy_actions = noise

        # Reverse diffusion: Iteratively denoise to generate action prediction
        for t in action_head.noise_scheduler.timesteps:
            # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action
            # embedding, and diffusion timestep embedding)
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )  # (B, llm_dim)
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

            # [Diffusion] Replace the embeddings of the action tokens with noisy actions
            # (Later on, the positional embeddings will be added to them)

            # For simplicity, append diffusion timestep embedding to the end of projected vision tokens
            projected_patch_embeddings = torch.cat(
                (orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
            )

            # Reshape and project noisy actions into language embedding space
            B = curr_noisy_actions.shape[0]
            orig_curr_noisy_actions_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

            # Replace action token embeddings with noisy action embeddings
            input_embeddings = self._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, noisy_action_features
            )

            # Build multimodal embeddings and attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # Forward pass through language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract hidden states for action portion of response
            last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                :,
            ]  # (B, act_chunk_len, D)

            # Predict noise and update noisy actions: x_t -> x_{t-1}
            noise_pred = action_head.predict_noise(actions_hidden_states)
            curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        # Return final actions
        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states

    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]  # (B, act_chunk_len, D)

        # Handle different prediction methods
        if action_head is not None:
            # L1 regression prediction
            normalized_actions = action_head.predict_action(actions_hidden_states)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # Discrete token-based prediction
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            action_begin, action_end, _ = self._action_vocab_range()
            discretized_actions = action_end - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states

    def _discrete_diffusion_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        input_ids=None,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
        assert input_ids is not None, "Input IDs must be provided for discrete diffusion prediction!"

        # Handle different prediction methods
        if action_head is not None:
            pass
            # # L1 regression prediction
            # normalized_actions = action_head.predict_action(actions_hidden_states)
            # normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            # normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:

            def tokens_to_logits(suffix_seq: torch.LongTensor) -> torch.Tensor:

                prefix = masked_input_ids[:, :1+NUM_PROMPT_TOKENS]
                suffix = masked_input_ids[:, 1+NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK:]
                full_seqs = torch.cat([prefix, suffix_seq, suffix], dim=1)

                # Get input embeddings and action masks
                input_embeddings = self.get_input_embeddings()(full_seqs)

                # Build multimodal embeddings and attention mask
                multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                    input_embeddings, projected_patch_embeddings, attention_mask
                )

                language_model_output = self.language_model(
                    input_ids=None,
                    attention_mask=multimodal_attention_mask,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=multimodal_embeddings,
                    labels=None,
                    use_cache=None,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

                logits = language_model_output.logits
                # topk_filter_thres = self.topk_filter_thres  # suggest 0.0
                # filtered_logits = parallel_decode.top_k_logits(
                #     logits, topk_filter_thres)
                filtered_logits = logits

                full_logits = filtered_logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                    :self.vocab_size
                ]
                action_begin, action_end, _ = self._action_vocab_range()
                neg_inf = torch.finfo(full_logits.dtype).min
                if action_begin > 0:
                    full_logits[..., :action_begin] = neg_inf
                if action_end < full_logits.shape[-1]:
                    full_logits[..., action_end:] = neg_inf

                # Extract hidden states for action tokens
                last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
                actions_hidden_states = last_hidden_states[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                    :,
                ]  # (B, act_chunk_len, D)

                return full_logits, actions_hidden_states          # suffix 部分 [B, L, V]

            # Set all action tokens to MASK_TOKEN
            # Note: Keep EOS/STOP tokens unchanged since their all_actions_mask is False
            mask_token_id = self.mask_token_id
            # Warn once if mask token collides with action-token range
            if not getattr(self, "_dfm_mask_collision_warned", False):
                action_begin, action_end, _ = self._action_vocab_range()
                if action_begin <= mask_token_id < action_end:
                    logger.warning(
                        "mask_token_id (%d) overlaps action-token range [%d, %d]. "
                        "DFM decoding will forbid mask-token sampling, but training/tokenizer config should be fixed.",
                        mask_token_id,
                        action_begin,
                        action_end - 1,
                    )
                self._dfm_mask_collision_warned = True
            masked_input_ids = torch.where(
                all_actions_mask, torch.tensor(mask_token_id, device=input_ids.device), input_ids
            )

            cur_seqs = masked_input_ids[
                :, 1+NUM_PROMPT_TOKENS:1+NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK
            ]  # (B, seq_len)

            final_iters, actions_hidden_states = parallel_decode.decode(
                init_ids=cur_seqs,
                tokens_to_logits=tokens_to_logits,
                mask_token_id=self.mask_token_id,
                num_iter=12,
                choice_temperature=1.0,  # to_test
                mask_scheduling_method="cosine",
                use_remask=False,
            )

            predicted_action_token_ids = final_iters[:, -1, :].cpu().numpy()
            action_begin, action_end, _ = self._action_vocab_range()
            discretized_actions = action_end - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states

    def _discrete_flow_matching_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        input_ids=None,
        dfm_num_steps: int = 12,
        dfm_schedule: str = "cosine",
        dfm_temperature: float = 1.0,
        dfm_temperature_anneal: str = "none",
        dfm_adaptive_step: bool = True,
        dfm_step_min: float = 1e-4,
        dfm_step_max: float = 0.2,
        dfm_time_eps: float = 1e-3,
        dfm_early_exit: bool = True,
        dfm_early_exit_frac: float = 0.0,
        dfm_corrector: bool = False,
        dfm_corrector_iters: int = 1,
        dfm_corrector_remask_frac: float = 0.1,
        dfm_clamp_mask: bool = False,
        dfm_clamp_values: Optional[torch.LongTensor] = None,
        return_debug: bool = False,
        dfm_debug_level: int = 1,
        dfm_decode_mode: str = "ctmc",
    ):
        """CTMC discrete flow matching prediction."""
        assert input_ids is not None, "Input IDs must be provided for DFM prediction!"

        if action_head is not None:
            # Continuous action heads not supported in DFM prediction
            pass
        else:

            def tokens_to_logits(suffix_seq: torch.LongTensor) -> torch.Tensor:
                prefix = masked_input_ids[:, : 1 + NUM_PROMPT_TOKENS]
                suffix = masked_input_ids[:, 1 + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK :]
                full_seqs = torch.cat([prefix, suffix_seq, suffix], dim=1)

                input_embeddings = self.get_input_embeddings()(full_seqs)

                multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                    input_embeddings, projected_patch_embeddings, attention_mask
                )

                language_model_output = self.language_model(
                    input_ids=None,
                    attention_mask=multimodal_attention_mask,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=multimodal_embeddings,
                    labels=None,
                    use_cache=None,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

                logits = language_model_output.logits

                full_logits = logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                    : self.vocab_size,
                ]
                action_begin, action_end, _ = self._action_vocab_range()
                neg_inf = torch.finfo(full_logits.dtype).min
                if action_begin > 0:
                    full_logits[..., :action_begin] = neg_inf
                if action_end < full_logits.shape[-1]:
                    full_logits[..., action_end:] = neg_inf

                last_hidden_states = language_model_output.hidden_states[-1]
                actions_hidden_states = last_hidden_states[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                    :,
                ]

                return full_logits, actions_hidden_states

            mask_token_id = self.mask_token_id
            masked_input_ids = torch.where(
                all_actions_mask, torch.tensor(mask_token_id, device=input_ids.device), input_ids
            )

            cur_seqs = masked_input_ids[
                :, 1 + NUM_PROMPT_TOKENS : 1 + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK
            ]
            init_snapshot = None
            if return_debug and dfm_debug_level >= 2:
                init_snapshot = cur_seqs[0, :16].detach().cpu().tolist()

            clamp_values = None
            if dfm_clamp_values is not None:
                clamp_values = dfm_clamp_values.to(cur_seqs.device)
                if clamp_values.dim() == 1:
                    clamp_values = clamp_values.unsqueeze(0).expand(cur_seqs.size(0), -1)
                if clamp_values.shape != cur_seqs.shape:
                    raise ValueError(
                        f"dfm_clamp_values must have shape {tuple(cur_seqs.shape)}, got {tuple(clamp_values.shape)}"
                    )

            clamp_mask = None
            if isinstance(dfm_clamp_mask, torch.Tensor):
                clamp_mask = dfm_clamp_mask.to(cur_seqs.device)
            elif clamp_values is not None:
                clamp_mask = clamp_values != mask_token_id
            elif dfm_clamp_mask:
                clamp_mask = cur_seqs != mask_token_id

            final_ids, actions_hidden_states, dfm_stats = dfm_decode(
                init_ids=cur_seqs,
                tokens_to_logits=tokens_to_logits,
                mask_token_id=mask_token_id,
                num_steps=dfm_num_steps,
                schedule=dfm_schedule,
                temperature=dfm_temperature,
                temperature_anneal=dfm_temperature_anneal,
                adaptive_step=dfm_adaptive_step,
                step_min=dfm_step_min,
                step_max=dfm_step_max,
                time_eps=dfm_time_eps,
                early_exit=dfm_early_exit,
                early_exit_frac=dfm_early_exit_frac,
                corrector=dfm_corrector,
                corrector_iters=dfm_corrector_iters,
                corrector_remask_frac=dfm_corrector_remask_frac,
                clamp_mask=clamp_mask,
                clamp_values=clamp_values,
                debug_level=dfm_debug_level if return_debug else 0,
                decode_mode=dfm_decode_mode,
            )
            # Telemetry: fraction of decoded tokens in action vocab range
            action_begin, action_end, n_bins = self._action_vocab_range()
            in_action = (final_ids >= action_begin) & (final_ids < action_end)
            dfm_stats["dfm_in_action_frac_final"] = in_action.float().mean().item()
            self.last_dfm_stats = dfm_stats
            if return_debug:
                self.last_dfm_debug = None

            predicted_action_token_ids = final_ids.cpu().numpy()
            discretized_actions = action_end - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

            debug = None
            if return_debug:
                action_start = 1 + NUM_PROMPT_TOKENS
                action_span_end = action_start + ACTION_DIM * NUM_ACTIONS_CHUNK
                prefix = input_ids[:, :action_start]
                suffix = input_ids[:, action_span_end:]
                full_seq = torch.cat([prefix, final_ids, suffix], dim=1)
                action_span_mask = torch.zeros_like(full_seq, dtype=torch.bool)
                action_span_mask[:, action_start:action_span_end] = True
                changed_off_action = (full_seq != input_ids) & (~action_span_mask)
                changed_off_action_count = int(changed_off_action.sum().item())
                stop_token_corrupted = bool((full_seq[:, -1] != input_ids[:, -1]).any().item())

                # Action stats
                actions_tensor = torch.as_tensor(normalized_actions)
                nan_frac = torch.isnan(actions_tensor).float().mean().item()
                actions_safe = torch.nan_to_num(actions_tensor, nan=0.0)
                action_min = actions_safe.min().item()
                action_max = actions_safe.max().item()
                action_mean = actions_safe.mean().item()
                action_std = actions_safe.std().item()
                clip_frac = ((actions_tensor <= -1.0) | (actions_tensor >= 1.0)).float().mean().item()
                debug = {
                    "mask_token_id": int(mask_token_id),
                    "mask_frac_final": dfm_stats.get("dfm_mask_frac_final"),
                    "valid_action_frac_final": dfm_stats.get("dfm_in_action_frac_final"),
                    "num_unique_action_tokens": int(final_ids.unique().numel()),
                    "action_stats": {
                        "min": action_min,
                        "max": action_max,
                        "mean": action_mean,
                        "std": action_std,
                        "nan_frac": nan_frac,
                        "clip_frac": clip_frac,
                    },
                    "dfm_stats": dfm_stats,
                    "changed_off_action_count": changed_off_action_count,
                    "stop_token_corrupted": stop_token_corrupted,
                    "action_vocab_range": {
                        "low": int(action_begin),
                        "high": int(action_end),
                        "n_bins": int(n_bins),
                    },
                }
                if dfm_debug_level >= 2:
                    debug["token_snapshot"] = {
                        "init": init_snapshot,
                        "final": final_ids[0, :16].detach().cpu().tolist(),
                    }
                self.last_dfm_debug = debug

        if return_debug:
            return normalized_actions, actions_hidden_states, debug
        return normalized_actions, actions_hidden_states

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        use_discrete_diffusion: bool = False,
        use_discrete_flow_matching: bool = False,
        dfm_num_steps: int = 12,
        dfm_schedule: str = "cosine",
        dfm_temperature: float = 1.0,
        dfm_temperature_anneal: str = "none",
        dfm_adaptive_step: bool = True,
        dfm_step_min: float = 1e-4,
        dfm_step_max: float = 0.2,
        dfm_time_eps: float = 1e-3,
        dfm_early_exit: bool = True,
        dfm_early_exit_frac: float = 0.0,
        dfm_corrector: bool = False,
        dfm_corrector_iters: int = 1,
        dfm_corrector_remask_frac: float = 0.1,
        dfm_clamp_mask: bool = False,
        dfm_clamp_values: Optional[torch.LongTensor] = None,
        return_debug: bool = False,
        dfm_debug_level: int = 1,
        dfm_decode_mode: str = "ctmc",
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            use_discrete_diffusion: Whether to use discrete diffusion for action prediction
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        if use_discrete_diffusion and use_discrete_flow_matching:
            raise ValueError("Cannot enable both discrete diffusion and discrete flow matching.")

        # Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

        # Prepare inputs by adding necessary tokens
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # Get input embeddings and action masks
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # Extract a safe language summary for FiLM conditioning
        language_embeddings = None
        if use_film:
            language_embeddings = self._compute_language_embeddings(
                input_embeddings, attention_mask, all_actions_mask
            )  # (B, 1, llm_dim)

        # Process vision features
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # Use diffusion if provided, otherwise use regression or discrete prediction
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        debug = None
        if use_diffusion:
            assert use_discrete_diffusion is False, "Discrete diffusion has not been supported in this method!"
            assert use_discrete_flow_matching is False, "DFM is not supported with diffusion action head!"
            # Sample random noise with shape equal to output action, used as the starting state for reverse diffusion
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            # Run diffusion-based prediction
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:
            if use_discrete_flow_matching:
                dfm_result = self._discrete_flow_matching_prediction(
                    input_embeddings,
                    all_actions_mask,
                    projected_patch_embeddings,
                    attention_mask,
                    labels,
                    NUM_PATCHES,
                    NUM_PROMPT_TOKENS,
                    action_head,
                    input_ids=input_ids,
                    dfm_num_steps=dfm_num_steps,
                    dfm_schedule=dfm_schedule,
                    dfm_temperature=dfm_temperature,
                    dfm_temperature_anneal=dfm_temperature_anneal,
                    dfm_adaptive_step=dfm_adaptive_step,
                    dfm_step_min=dfm_step_min,
                    dfm_step_max=dfm_step_max,
                    dfm_time_eps=dfm_time_eps,
                    dfm_early_exit=dfm_early_exit,
                    dfm_early_exit_frac=dfm_early_exit_frac,
                    dfm_corrector=dfm_corrector,
                    dfm_corrector_iters=dfm_corrector_iters,
                    dfm_corrector_remask_frac=dfm_corrector_remask_frac,
                    dfm_clamp_mask=dfm_clamp_mask,
                    dfm_clamp_values=dfm_clamp_values,
                    return_debug=return_debug,
                    dfm_debug_level=dfm_debug_level,
                    dfm_decode_mode=dfm_decode_mode,
                )
                if return_debug:
                    normalized_actions, actions_hidden_states, debug = dfm_result
                else:
                    normalized_actions, actions_hidden_states = dfm_result
            elif use_discrete_diffusion:
                normalized_actions, actions_hidden_states = self._discrete_diffusion_prediction(
                    input_embeddings,
                    all_actions_mask,
                    projected_patch_embeddings,
                    attention_mask,
                    labels,
                    NUM_PATCHES,
                    NUM_PROMPT_TOKENS,
                    action_head,
                    input_ids=input_ids,
                )
            else:
                # Run regression or discrete token-based prediction
                normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
                    input_embeddings,
                    all_actions_mask,
                    projected_patch_embeddings,
                    attention_mask,
                    labels,
                    NUM_PATCHES,
                    NUM_PROMPT_TOKENS,
                    action_head,
                )

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        if return_debug:
            return actions, actions_hidden_states, debug
        return actions, actions_hidden_states

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]


class DiscreteDiffusionForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX  # TODO: IMPORTANT
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        action_begin, _, _ = self._action_vocab_range()
        ARBITRARY_ACTION_TOKEN_IDX = action_begin
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions

    def _run_diffusion_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        noise,
        action_head,
        projected_patch_embeddings,
        labels,
        attention_mask,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        noisy_action_projector,
    ):
        """Run diffusion-based action prediction"""
        # Clone embedding for reuse in each timestep
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        curr_noisy_actions = noise

        # Reverse diffusion: Iteratively denoise to generate action prediction
        for t in action_head.noise_scheduler.timesteps:
            # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action
            # embedding, and diffusion timestep embedding)
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )  # (B, llm_dim)
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

            # [Diffusion] Replace the embeddings of the action tokens with noisy actions
            # (Later on, the positional embeddings will be added to them)

            # For simplicity, append diffusion timestep embedding to the end of projected vision tokens
            projected_patch_embeddings = torch.cat(
                (orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
            )

            # Reshape and project noisy actions into language embedding space
            B = curr_noisy_actions.shape[0]
            orig_curr_noisy_actions_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

            # Replace action token embeddings with noisy action embeddings
            input_embeddings = self._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, noisy_action_features
            )

            # Build multimodal embeddings and attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # Forward pass through language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract hidden states for action portion of response
            last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                :,
            ]  # (B, act_chunk_len, D)

            # Predict noise and update noisy actions: x_t -> x_{t-1}
            noise_pred = action_head.predict_noise(actions_hidden_states)
            curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        # Return final actions
        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states

    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]  # (B, act_chunk_len, D)

        # Handle different prediction methods
        if action_head is not None:
            # L1 regression prediction
            normalized_actions = action_head.predict_action(actions_hidden_states)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # Discrete token-based prediction
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            action_begin, action_end, _ = self._action_vocab_range()
            discretized_actions = action_end - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

        # Prepare inputs by adding necessary tokens
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # Get input embeddings and action masks
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # Extract a safe language summary for FiLM conditioning
        language_embeddings = None
        if use_film:
            language_embeddings = self._compute_language_embeddings(
                input_embeddings, attention_mask, all_actions_mask
            )  # (B, 1, llm_dim)

        # Process vision features
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # Use diffusion if provided, otherwise use regression or discrete prediction
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            # Sample random noise with shape equal to output action, used as the starting state for reverse diffusion
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            # Run diffusion-based prediction
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:
            # Run regression or discrete token-based prediction
            normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                action_head,
            )

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
