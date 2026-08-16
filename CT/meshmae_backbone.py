from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import Attention, LayerScale

from chamfer_dist import ChamferDistanceL1


class AdjacencyExpert(nn.Module):

    def __init__(self):
        super().__init__()
        self.bias_generator = self.AdaptiveBiasGenerator()

    class AdaptiveBiasGenerator(nn.Module):

        def __init__(self, initial_A=2.0, initial_sigma=2.0):
            super().__init__()
            self.log_A = nn.Parameter(torch.tensor(initial_A).log())
            self.log_sigma = nn.Parameter(torch.tensor(initial_sigma).log())

        def forward(self, hop_matrix):
            A = self.log_A.exp()
            sigma = self.log_sigma.exp()
            hop_matrix = hop_matrix.float().masked_fill(hop_matrix < 0, 1e9)
            return A * torch.exp(-hop_matrix.pow(2) / (2 * sigma.pow(2) + 1e-6))


class RelativeWindowMultiheadAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_bias: Optional[torch.Tensor]) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_bias is not None:
            attn_scores = attn_scores + attention_bias.unsqueeze(1)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        output = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.out_proj(output))


class PerspectiveTransform(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.sem_proj = nn.Linear(dim, dim)
        self.geo_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        return self.sem_proj(x), self.geo_proj(x)


class Block(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        mlp_layer: nn.Module = Mlp,
        init_values: Optional[float] = None,
        max_geom_weight: float = 0.9,
        path_mode: str = "dual",
    ):
        super().__init__()
        assert path_mode in ("dual", "local", "global")
        self.path_mode = path_mode
        self.norm1 = norm_layer(dim)

        self.perspective_transform = PerspectiveTransform(dim)

        self.standard_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop
        )

        self.region_attn = RelativeWindowMultiheadAttention(
            dim, num_heads=num_heads, dropout=attn_drop
        )

        self.max_geom_weight = max_geom_weight
        self.raw_gate = nn.Parameter(torch.tensor(.0))

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=proj_drop
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, region_logits: Optional[torch.Tensor],
                has_cls_token: bool) -> torch.Tensor:
        shortcut1 = x
        y = self.norm1(x)
        y_std, y_geom = self.perspective_transform(y)

        standard_attn_output = self.standard_attn(y_std)

        if self.path_mode == "global":
            combined_attn_output = standard_attn_output
        else:
            if has_cls_token:
                y_patch = y_geom[:, 1:, :]
            else:
                y_patch = y_geom
            region_attn_output_patch = self.region_attn(y_patch, region_logits)

            if self.path_mode == "local":
                if has_cls_token:
                    cls_attn = standard_attn_output[:, :1, :]
                    combined_attn_output = torch.cat([cls_attn, region_attn_output_patch], dim=1)
                else:
                    combined_attn_output = region_attn_output_patch
            else:
                w_geom = self.max_geom_weight * torch.sigmoid(self.raw_gate)
                w_std = 1.0 - w_geom
                if has_cls_token:
                    standard_attn_cls = standard_attn_output[:, :1, :]
                    standard_attn_patch = standard_attn_output[:, 1:, :]
                    combined_patch_output = w_std * standard_attn_patch + w_geom * region_attn_output_patch
                    combined_attn_output = torch.cat([standard_attn_cls, combined_patch_output], dim=1)
                else:
                    combined_attn_output = w_std * standard_attn_output + w_geom * region_attn_output_patch

        x = shortcut1 + self.drop_path1(self.ls1(combined_attn_output))

        y_norm2 = self.norm2(x)
        x = y_norm2 + self.drop_path2(self.ls2(self.mlp(y_norm2)))

        return x


class Mesh_mae(nn.Module):

    def __init__(self, masking_ratio=0.5, channels=14, num_heads=12, encoder_depth=12, embed_dim=768,
                 decoder_num_heads=16, decoder_depth=6, decoder_embed_dim=512,
                 patch_size=1024, num_patches=1024, norm_layer=nn.LayerNorm, weight=0.2,
                 path_mode="dual"):
        super(Mesh_mae, self).__init__()
        patch_dim = channels
        self.num_patches = num_patches
        self.weight = weight
        self.points_per_patch = 45
        self.embed_dim = embed_dim
        self.path_mode = path_mode

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        self.decoer_pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, decoder_embed_dim)
        )
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),
            nn.Linear(patch_dim * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder_cls_token_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=norm_layer,
                  path_mode=path_mode)
            for _ in range(encoder_depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.region_assigner = AdjacencyExpert()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=norm_layer,
                  path_mode=path_mode)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.to_points = nn.Linear(decoder_embed_dim, patch_size * 9)
        self.to_pointsnew = nn.Linear(decoder_embed_dim, self.points_per_patch * 3)
        self.to_points_seg = nn.Linear(decoder_embed_dim, 9)
        self.to_features = nn.Linear(decoder_embed_dim, patch_size * channels)
        self.to_features_seg = nn.Linear(decoder_embed_dim, channels)
        self.build_loss_func()
        self.initialize_weights()

    def build_loss_func(self):
        self.loss_func_cdl1 = ChamferDistanceL1().cuda()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m in [self.to_pointsnew, self.to_features, self.to_points]:
                torch.nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, faces, feats, centers, Fs, cordinates, hop_matrix, ratio=0.25):
        patch_num = self.num_patches
        points_num_per_patch = self.points_per_patch

        feats_patches = feats
        centers_patches = centers
        faces_per_patch = centers_patches.shape[2]

        center_of_patches = torch.mean(centers_patches, dim=2)
        batch, channel, num_patches, *_ = feats_patches.shape
        cordinates_patches = cordinates

        pos_emb = self.pos_embedding(center_of_patches)
        encoder_cls_token_pos = self.encoder_cls_token_pos.repeat(batch, 1, 1)
        tokens = self.to_patch_embedding(feats_patches)

        num_masked = int(ratio * patch_num)

        rand_indices = torch.rand(batch, patch_num, device=feats.device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(batch, device=feats.device)[:, None]
        tokens_unmasked = tokens[batch_range, unmasked_indices]

        idx = unmasked_indices
        P = hop_matrix.shape[-1]
        hop_rows = torch.gather(hop_matrix, 1, idx.unsqueeze(-1).expand(-1, -1, P))
        K = idx.shape[1]
        hop_unmasked = torch.gather(hop_rows, 2, idx.unsqueeze(1).expand(-1, K, -1))
        unmasked_logits = self.region_assigner.bias_generator(hop_unmasked)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        tokens_unmasked = torch.cat((cls_tokens, tokens_unmasked), dim=1)
        pos_emb_a = torch.cat((encoder_cls_token_pos, pos_emb[batch_range, unmasked_indices]), dim=1)
        tokens_unmasked = tokens_unmasked + pos_emb_a

        for blk in self.blocks:
            tokens_unmasked = blk(tokens_unmasked, unmasked_logits, has_cls_token=True)
        tokens_unmasked = self.norm(tokens_unmasked)

        encoder_output_patches = self.decoder_embed(tokens_unmasked[:, 1:, :])

        decoder_tokens = self.mask_token.repeat(batch, self.num_patches, 1)
        decoder_pos_emb = self.decoer_pos_embedding(center_of_patches)

        decoder_tokens[batch_range, unmasked_indices] = encoder_output_patches.to(decoder_tokens.dtype)
        decoded_tokens = decoder_tokens + decoder_pos_emb

        for blk in self.decoder_blocks:
            decoded_tokens = blk(decoded_tokens, region_logits=None, has_cls_token=False)
        decoded_tokens = self.decoder_norm(decoded_tokens)

        pred_tokens = decoded_tokens[batch_range, masked_indices]
        pred_vertices_coordinates = self.to_pointsnew(pred_tokens)

        faces_values_per_patch = feats_patches.shape[-1]
        pred_vertices_coordinates = torch.reshape(pred_vertices_coordinates,
                                                  (batch, num_masked, points_num_per_patch, 3)).contiguous()

        center = torch.mean(centers_patches[batch_range, masked_indices], dim=2)
        pred_vertices_coordinates = pred_vertices_coordinates + center.unsqueeze(2).repeat(
            1, 1, points_num_per_patch, 1)
        pred_vertices_coordinates = torch.reshape(pred_vertices_coordinates,
                                                  (batch * num_masked, points_num_per_patch, 3)).contiguous()

        cordinates_patches = cordinates_patches[batch_range, masked_indices]
        cordinates_patches = torch.reshape(cordinates_patches, (batch, num_masked, -1, 3)).contiguous()

        cordinates_unique = torch.unique(cordinates_patches, dim=2)
        cordinates_unique = torch.reshape(cordinates_unique, (batch * num_masked, -1, 3)).contiguous()

        masked_feats_patches = feats_patches[batch_range, :, masked_indices]

        pred_faces_features = self.to_features(pred_tokens)
        pred_faces_features = torch.reshape(pred_faces_features, (batch, num_masked, channel, faces_values_per_patch))

        shape_con_loss = self.loss_func_cdl1(pred_vertices_coordinates, cordinates_unique)
        feats_con_loss = F.mse_loss(pred_faces_features, masked_feats_patches)

        loss = shape_con_loss + self.weight * feats_con_loss

        return loss, feats_con_loss, shape_con_loss
