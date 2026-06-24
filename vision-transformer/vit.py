"""
Vision Transformer (ViT) — Dosovitskiy et al., 2020
https://arxiv.org/abs/2010.11929
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Section 3.1 — split image into P×P patches, linearly project each to embed_dim.

    Input:  (B, C, H, W)
    Output: (B, num_patches, embed_dim)   where num_patches = (H/P) * (W/P)
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # TODO: linear projection of flattened patches (Eq. 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """
    Section 3 / Appendix A — standard multi-head self-attention (MSA).

    Input:  (B, N, embed_dim)
    Output: (B, N, embed_dim)
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # TODO: Q, K, V projections + output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlock(nn.Module):
    """
    Section 3 / Eq. 2-3 — one encoder layer: LayerNorm → MSA → residual,
    then LayerNorm → MLP → residual.

    Input:  (B, N, embed_dim)
    Output: (B, N, embed_dim)
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # TODO: norm1, attn, norm2, mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VisionTransformer(nn.Module):
    """
    Section 3 — full ViT.

    1. PatchEmbedding
    2. Prepend CLS token  (Eq. 1)
    3. Add position embeddings  (Eq. 1, learned 1-D)
    4. depth × TransformerBlock
    5. LayerNorm
    6. MLP head on CLS token output

    Input:  (B, C, H, W)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        # TODO: patch_embed, cls_token, pos_embed, blocks, norm, head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
