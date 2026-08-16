from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import Attention, LayerScale


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
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, attention_bias):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        s = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_bias is not None:
            s = s + attention_bias.unsqueeze(1)
        a = F.softmax(s, dim=-1)
        a = self.attn_drop(a)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.out_proj(o))


class PerspectiveTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sem_proj = nn.Linear(dim, dim)
        self.geo_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.sem_proj(x), self.geo_proj(x)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, proj_drop=0.1, attn_drop=0.1,
                 drop_path=0.1, norm_layer=nn.LayerNorm, act_layer=nn.GELU, mlp_layer=Mlp,
                 init_values=None, max_geom_weight=0.9, path_mode="dual"):
        super().__init__()
        assert path_mode in ("dual", "local", "global"), f"path_mode must be dual|local|global, got {path_mode}"
        self.path_mode = path_mode
        self.norm1 = norm_layer(dim)
        self.perspective_transform = PerspectiveTransform(dim)
        self.standard_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=proj_drop)
        self.region_attn = RelativeWindowMultiheadAttention(dim, num_heads=num_heads, dropout=attn_drop)
        self.max_geom_weight = max_geom_weight
        self.raw_gate = nn.Parameter(torch.tensor(.0))
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, region_logits, has_cls_token):
        shortcut1 = x
        y = self.norm1(x)
        y_std, y_geom = self.perspective_transform(y)
        standard_attn_output = self.standard_attn(y_std)
        if self.path_mode == "global":
            combined = standard_attn_output
        else:
            if has_cls_token:
                y_patch = y_geom[:, 1:, :]
            else:
                y_patch = y_geom
            region_attn_output_patch = self.region_attn(y_patch, region_logits)
            if self.path_mode == "local":
                if has_cls_token:
                    combined = torch.cat([standard_attn_output[:, :1, :], region_attn_output_patch], dim=1)
                else:
                    combined = region_attn_output_patch
            else:
                w_geom = self.max_geom_weight * torch.sigmoid(self.raw_gate)
                w_std = 1.0 - w_geom
                if has_cls_token:
                    combined = torch.cat([standard_attn_output[:, :1, :],
                                          w_std * standard_attn_output[:, 1:, :] + w_geom * region_attn_output_patch], dim=1)
                else:
                    combined = w_std * standard_attn_output + w_geom * region_attn_output_patch
        x = shortcut1 + self.drop_path1(self.ls1(combined))
        y_norm2 = self.norm2(x)
        x = y_norm2 + self.drop_path2(self.ls2(self.mlp(y_norm2)))
        return x


class HopEncoder(nn.Module):

    def __init__(self, channels=10, num_heads=6, encoder_depth=6, embed_dim=384,
                 patch_size=64, drop_path=0.1, path_mode="dual"):
        super().__init__()
        self.embed_dim = embed_dim
        self.path_mode = path_mode
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

    def forward(self, feats, centers, hop_matrix, return_tokens=False):
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
        patch_tokens = tokens[:, 1:, :]
        if return_tokens:
            return patch_tokens
        return patch_tokens.mean(dim=1)


_ENC_PREFIXES = ("pos_embedding.", "to_patch_embedding.", "cls_token", "encoder_cls_token_pos",
                 "blocks.", "norm.", "region_assigner.")


def _filter_encoder_keys(sd):
    out = {}
    for k, v in sd.items():
        if any(k.startswith(p) for p in _ENC_PREFIXES):
            out[k] = v
    return out


class AttentionPool(nn.Module):

    def __init__(self, dim, n_queries=8, num_heads=6, drop=0.1):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.zeros(1, n_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b = x.shape[0]
        q = self.queries.expand(b, -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(out).reshape(b, -1)


class BilateralFCNet(nn.Module):

    def __init__(self, n_out, lh_ckpt, rh_ckpt, embed_dim=384, channels=10,
                 num_heads=6, encoder_depth=6, load_pretrained=True,
                 n_queries=8, path_mode="dual"):
        super().__init__()
        self.path_mode = path_mode
        self.left = HopEncoder(embed_dim=embed_dim, channels=channels,
                               num_heads=num_heads, encoder_depth=encoder_depth, path_mode=path_mode)
        self.right = HopEncoder(embed_dim=embed_dim, channels=channels,
                                num_heads=num_heads, encoder_depth=encoder_depth, path_mode=path_mode)
        self.pool_l = AttentionPool(embed_dim, n_queries, num_heads)
        self.pool_r = AttentionPool(embed_dim, n_queries, num_heads)
        feat_dim = embed_dim * n_queries * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, n_out))
        if load_pretrained:
            self._load_pretrained(lh_ckpt, rh_ckpt)
        else:
            print("[init] random encoder (no pretrained load) -- ablation control", flush=True)
        self._init_head()

    def _load_pretrained(self, lh, rh):
        for model, ckpt_path, name in [(self.left, lh, "left"), (self.right, rh, "right")]:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            filt = _filter_encoder_keys(sd)
            msg = model.load_state_dict(filt, strict=False)
            enc_missing = [m for m in msg.missing_keys if any(m.startswith(p) for p in _ENC_PREFIXES)]
            bg = model.region_assigner.bias_generator
            print(f"[load {name}] enc_keys={len(filt)} enc_missing={len(enc_missing)} "
                  f"A={float(bg.log_A.exp()):.3f} sigma={float(bg.log_sigma.exp()):.3f}", flush=True)

    def _init_head(self):
        head_modules = (list(self.fusion.modules()) + list(self.pool_l.modules())
                        + list(self.pool_r.modules()))
        for m in head_modules:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, feats_l, centers_l, hop_l, feats_r, centers_r, hop_r):
        tl = self.left(feats_l, centers_l, hop_l, return_tokens=True)
        tr = self.right(feats_r, centers_r, hop_r, return_tokens=True)
        el = self.pool_l(tl)
        er = self.pool_r(tr)
        return self.fusion(torch.cat([el, er], dim=1))
