import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from meshmae_backbone import Block, AdjacencyExpert


class RegressionHead(nn.Module):

    def __init__(self, embed_dim=384, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class Mesh_regressor(nn.Module):

    def __init__(self, channels=10, num_heads=6, encoder_depth=6, embed_dim=384,
                 patch_size=64, drop_path=0.1, path_mode="dual"):
        super().__init__()
        self.embed_dim = embed_dim
        self.path_mode = path_mode

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim))
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

        self.head = RegressionHead(embed_dim=embed_dim, hidden_dim=256, dropout_rate=0.3)
        self.apply(self._init_new)

    def _init_new(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, faces, feats, centers, coordinates, hop_matrix):
        batch = feats.shape[0]
        center_of_patches = torch.mean(centers, dim=2)
        pos_emb = self.pos_embedding(center_of_patches)
        tokens = self.to_patch_embedding(feats)

        attention_bias = self.region_assigner.bias_generator(hop_matrix)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        cls_pos = self.encoder_cls_token_pos.expand(batch, -1, -1)
        pos_emb_full = torch.cat([cls_pos, pos_emb], dim=1)
        tokens = tokens + pos_emb_full

        for blk in self.blocks:
            tokens = blk(tokens, attention_bias, has_cls_token=True)
        tokens = self.norm(tokens)

        global_feature = tokens[:, 1:, :].mean(dim=1)
        return self.head(global_feature)
