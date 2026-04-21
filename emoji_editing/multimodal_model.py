"""High-performance multimodal encoder for emoji edit instructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, CLIPVisionModel


@dataclass
class EmojiEditMultimodalConfig:
    """Configuration for the multimodal conditioning stack."""

    text_model_name: str = "intfloat/multilingual-e5-base"
    vision_model_name: str = "openai/clip-vit-base-patch32"
    attn_implementation: str = "sdpa"
    fusion_dim: int = 768
    num_query_tokens: int = 8
    fusion_layers: int = 4
    fusion_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    freeze_text_backbone: bool = False
    freeze_vision_backbone: bool = False
    gradient_checkpointing: bool = True
    contrastive_loss_weight: float = 1.0
    emotion_loss_weight: float = 0.35
    vendor_loss_weight: float = 0.35
    task_loss_weight: float = 0.2
    sentiment_loss_weight: float = 0.1
    label_smoothing: float = 0.0


def _masked_mean(hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _load_transformer_backbone(model_name: str, attn_implementation: str | None) -> nn.Module:
    kwargs: dict[str, Any] = {}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return AutoModel.from_pretrained(model_name, **kwargs)
    except TypeError:
        return AutoModel.from_pretrained(model_name)


def _load_clip_vision_backbone(model_name: str, attn_implementation: str | None) -> CLIPVisionModel:
    kwargs: dict[str, Any] = {}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return CLIPVisionModel.from_pretrained(model_name, **kwargs)
    except TypeError:
        return CLIPVisionModel.from_pretrained(model_name)


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class FeedForward(nn.Module):
    """Transformer MLP block."""

    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class QueryFusionBlock(nn.Module):
    """Q-former style block using self-attention plus text/image cross attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.text_norm = nn.LayerNorm(dim)
        self.text_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.image_norm = nn.LayerNorm(dim)
        self.image_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        query_tokens: Tensor,
        text_tokens: Tensor,
        image_tokens: Tensor,
        text_key_padding_mask: Tensor | None,
    ) -> Tensor:
        x = query_tokens
        self_attn_input = self.self_norm(x)
        x = x + self.self_attn(
            self_attn_input,
            self_attn_input,
            self_attn_input,
            need_weights=False,
        )[0]

        text_query = self.text_norm(x)
        x = x + self.text_cross_attn(
            text_query,
            text_tokens,
            text_tokens,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )[0]

        image_query = self.image_norm(x)
        x = x + self.image_cross_attn(
            image_query,
            image_tokens,
            image_tokens,
            need_weights=False,
        )[0]

        x = x + self.ffn(self.ffn_norm(x))
        return x


class EmojiEditMultimodalEncoder(nn.Module):
    """Source image + edit instruction encoder optimized for GPU training."""

    def __init__(
        self,
        config: EmojiEditMultimodalConfig,
        num_emotions: int,
        num_sentiments: int,
        num_task_types: int,
        num_vendors: int,
    ) -> None:
        super().__init__()
        self.config = config

        self.text_backbone = _load_transformer_backbone(
            config.text_model_name,
            attn_implementation=config.attn_implementation,
        )
        self.vision_backbone = _load_clip_vision_backbone(
            config.vision_model_name,
            attn_implementation=config.attn_implementation,
        )

        if config.gradient_checkpointing and hasattr(self.text_backbone, "gradient_checkpointing_enable"):
            self.text_backbone.gradient_checkpointing_enable()
        if config.gradient_checkpointing and hasattr(self.vision_backbone, "gradient_checkpointing_enable"):
            self.vision_backbone.gradient_checkpointing_enable()

        if config.freeze_text_backbone:
            _freeze_module(self.text_backbone)
        if config.freeze_vision_backbone:
            _freeze_module(self.vision_backbone)

        text_width = int(self.text_backbone.config.hidden_size)
        vision_width = int(self.vision_backbone.config.hidden_size)

        self.text_projection = nn.Linear(text_width, config.fusion_dim)
        self.image_projection = nn.Linear(vision_width, config.fusion_dim)
        self.target_projection = nn.Linear(vision_width, config.fusion_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, config.num_query_tokens, config.fusion_dim) * 0.02)
        self.fusion_layers = nn.ModuleList(
            [
                QueryFusionBlock(
                    dim=config.fusion_dim,
                    num_heads=config.fusion_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.fusion_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.fusion_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim),
            nn.GELU(),
            nn.LayerNorm(config.fusion_dim),
        )

        self.task_head = nn.Linear(config.fusion_dim, num_task_types)
        self.vendor_head = nn.Linear(config.fusion_dim, num_vendors)
        self.emotion_head = nn.Linear(config.fusion_dim, num_emotions)
        self.sentiment_head = nn.Linear(config.fusion_dim, num_sentiments)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_tokens = self.text_projection(text_outputs.last_hidden_state)
        pooled = getattr(text_outputs, "pooler_output", None)
        if pooled is None:
            pooled = _masked_mean(text_outputs.last_hidden_state, attention_mask)
        return text_tokens, pooled

    def encode_image_tokens(self, images: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.vision_backbone(pixel_values=images, return_dict=True)
        tokens = self.image_projection(outputs.last_hidden_state)
        pooled = outputs.pooler_output
        return tokens, pooled

    def encode_target_embeddings(self, target_images: Tensor) -> Tensor:
        outputs = self.vision_backbone(pixel_values=target_images, return_dict=True)
        projected = self.target_projection(outputs.pooler_output)
        return F.normalize(projected, dim=-1)

    def fuse(self, text_tokens: Tensor, image_tokens: Tensor, attention_mask: Tensor) -> Tensor:
        query = self.query_tokens.expand(text_tokens.size(0), -1, -1)
        key_padding_mask = attention_mask == 0
        for layer in self.fusion_layers:
            query = layer(
                query_tokens=query,
                text_tokens=text_tokens,
                image_tokens=image_tokens,
                text_key_padding_mask=key_padding_mask,
            )
        return self.final_norm(query)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        source_images: Tensor,
        target_images: Tensor | None = None,
        task_type_ids: Tensor | None = None,
        target_vendor_ids: Tensor | None = None,
        target_emotion_ids: Tensor | None = None,
        target_sentiment_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        text_tokens, _ = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)
        image_tokens, _ = self.encode_image_tokens(source_images)

        conditioning_tokens = self.fuse(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            attention_mask=attention_mask,
        )
        pooled = self.output_projection(conditioning_tokens.mean(dim=1))
        fused_embedding = F.normalize(pooled, dim=-1)

        outputs = {
            "conditioning_tokens": conditioning_tokens,
            "fused_embedding": fused_embedding,
            "task_logits": self.task_head(pooled),
            "vendor_logits": self.vendor_head(pooled),
            "emotion_logits": self.emotion_head(pooled),
            "sentiment_logits": self.sentiment_head(pooled),
        }

        if target_images is None:
            return outputs

        with torch.set_grad_enabled(not self.config.freeze_vision_backbone):
            target_embedding = self.encode_target_embeddings(target_images)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * fused_embedding @ target_embedding.t()
        labels = torch.arange(logits.size(0), device=logits.device)

        label_smoothing = self.config.label_smoothing
        contrastive_loss = 0.5 * (
            F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            + F.cross_entropy(logits.t(), labels, label_smoothing=label_smoothing)
        )

        total_loss = contrastive_loss * self.config.contrastive_loss_weight
        loss_dict = {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss.detach(),
        }

        if task_type_ids is not None:
            task_loss = F.cross_entropy(outputs["task_logits"], task_type_ids, label_smoothing=label_smoothing)
            total_loss = total_loss + task_loss * self.config.task_loss_weight
            loss_dict["task_loss"] = task_loss.detach()
        if target_vendor_ids is not None:
            vendor_loss = F.cross_entropy(outputs["vendor_logits"], target_vendor_ids, label_smoothing=label_smoothing)
            total_loss = total_loss + vendor_loss * self.config.vendor_loss_weight
            loss_dict["vendor_loss"] = vendor_loss.detach()
        if target_emotion_ids is not None:
            emotion_loss = F.cross_entropy(outputs["emotion_logits"], target_emotion_ids, label_smoothing=label_smoothing)
            total_loss = total_loss + emotion_loss * self.config.emotion_loss_weight
            loss_dict["emotion_loss"] = emotion_loss.detach()
        if target_sentiment_ids is not None:
            sentiment_loss = F.cross_entropy(
                outputs["sentiment_logits"],
                target_sentiment_ids,
                label_smoothing=label_smoothing,
            )
            total_loss = total_loss + sentiment_loss * self.config.sentiment_loss_weight
            loss_dict["sentiment_loss"] = sentiment_loss.detach()

        loss_dict["loss"] = total_loss
        outputs.update(loss_dict)
        outputs["target_embedding"] = target_embedding
        return outputs
