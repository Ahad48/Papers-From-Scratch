import torch
import pytest
from vit import PatchEmbedding, MultiHeadAttention, TransformerBlock, VisionTransformer


def test_patch_embedding_output_shape():
    """(B, C, H, W) -> (B, num_patches, embed_dim)"""
    embed = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)
    x = torch.zeros(2, 3, 224, 224)
    out = embed(x)
    # 224/16 = 14, so 14*14 = 196 patches
    assert out.shape == (2, 196, 768)


def test_patch_embedding_different_patch_size():
    """patch_size=32: 224/32=7, so 7*7=49 patches"""
    embed = PatchEmbedding(img_size=224, patch_size=32, in_channels=3, embed_dim=512)
    x = torch.zeros(1, 3, 224, 224)
    out = embed(x)
    assert out.shape == (1, 49, 512)


def test_patch_embedding_num_patches_attribute():
    """num_patches attribute must equal (img_size // patch_size) ** 2"""
    embed = PatchEmbedding(img_size=32, patch_size=4, in_channels=3, embed_dim=64)
    assert embed.num_patches == 64  # (32/4)^2


def test_multihead_attention_output_shape():
    """(B, N, D) in -> (B, N, D) out — shape preserved"""
    attn = MultiHeadAttention(embed_dim=64, num_heads=4)
    x = torch.zeros(2, 10, 64)
    out = attn(x)
    assert out.shape == (2, 10, 64)


def test_multihead_attention_invalid_heads_raises():
    """embed_dim not divisible by num_heads must raise AssertionError"""
    with pytest.raises(AssertionError):
        MultiHeadAttention(embed_dim=65, num_heads=4)


def test_transformer_block_output_shape():
    """residual path: (B, N, D) in -> (B, N, D) out"""
    block = TransformerBlock(embed_dim=64, num_heads=4, mlp_ratio=4)
    x = torch.zeros(2, 10, 64)
    out = block(x)
    assert out.shape == (2, 10, 64)


def test_vit_output_shape():
    """minimal config: (B, C, H, W) -> (B, num_classes)"""
    model = VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=64, depth=2, num_heads=4, mlp_ratio=4,
    )
    x = torch.zeros(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_vit_b16_config():
    """ViT-B/16 paper config (Table 1): embed=768, depth=12, heads=12"""
    model = VisionTransformer(
        img_size=224, patch_size=16, in_channels=3, num_classes=1000,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
    )
    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 1000)


def test_vit_batch_size_one():
    """single-image batch still produces correct output shape"""
    model = VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=64, depth=2, num_heads=4, mlp_ratio=4,
    )
    x = torch.zeros(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 10)
