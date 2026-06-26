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
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)        # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, embed_dim)
        return x

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
        
        self.scale = self.head_dim ** -0.5  # scaling factor for dot-product attention
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # query, key, value projections
        self.out_proj = nn.Linear(embed_dim, embed_dim) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape #batch size, number of tokens / Sequence length, embedding dimension
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # (B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q,k,v = qkv.unbind(0)  # each is (B, num_heads, N, head_dim)

        attention = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N )
        attention = attention.softmax(dim=-1)  # (B, num_heads, N, N)
        out = attention @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)  # (B, N, D)
        return out
        


class TransformerBlock(nn.Module):
    """
    Section 3 / Eq. 2-3 — one encoder layer: LayerNorm → MSA → residual,
    then LayerNorm → MLP → residual.

    Input:  (B, N, embed_dim)
    Output: (B, N, embed_dim)
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.mhsa = MultiHeadAttention(embed_dim, num_heads)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, (embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear((embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layernorm1(x)
        out = self.mhsa(out)
        x = x + out  # residual connection
        out = self.layernorm2(x)
        out = self.mlp(out)
        x = x + out  # residual connection
        return x


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
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # learnable [CLS] token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # learnable position embeddings, +1 for the CLS token
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed  # (B, num_patches + 1, embed_dim)
        for block in self.transformer_blocks:
            x = block(x)  # (B, num_patches + 1, embed_dim)
        cls_output = x[:, 0]  # (B, embed_dim) — take the output corresponding to the [CLS] token
        cls_output = self.mlp_head(cls_output)  # (B, num_classes)
        return cls_output
