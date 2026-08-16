import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from meshmae_fc import Block, AdjacencyExpert


class TFMRIVertexEncoder(nn.Module):

    def __init__(self, channels=10, num_heads=6, encoder_depth=6, embed_dim=384,
                 patch_size=64, drop_path=0.1, path_mode="dual"):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embedding = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim))
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c h p -> b h (p c)", p=patch_size),
            nn.Linear(channels * patch_size, embed_dim),
            nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder_cls_token_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4., qkv_bias=True,
                  norm_layer=nn.LayerNorm, drop_path=drop_path, path_mode=path_mode)
            for _ in range(encoder_depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.region_assigner = AdjacencyExpert()
        self.attn_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn_pool = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        nn.init.normal_(self.attn_query, std=0.02)

    def forward(self, feats, centers, hop_matrix):
        b = feats.shape[0]
        cop = torch.mean(centers, dim=2)
        pos_emb = self.pos_embedding(cop)
        tokens = self.to_patch_embedding(feats)
        attention_bias = self.region_assigner.bias_generator(hop_matrix)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cpos = self.encoder_cls_token_pos.expand(b, -1, -1)
        tokens = tokens + torch.cat([cpos, pos_emb], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens, attention_bias, has_cls_token=True)
        tokens = self.norm(tokens)
        patch = tokens[:, 1:]
        q = self.attn_query.expand(patch.size(0), -1, -1)
        out, _ = self.attn_pool(q, patch, patch, need_weights=False)
        return out.squeeze(1)
