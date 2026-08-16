import torch
import torch.nn as nn
from meshmae_tfmri_vertex import TFMRIVertexEncoder
from meshmae_fc import _filter_encoder_keys


class TfMRIBilateralNet(nn.Module):
    def __init__(self, n_sub, n_vertices, lh_ckpt, rh_ckpt, embed_dim=384, channels=10,
                 num_heads=6, encoder_depth=6, load_pretrained=True, path_mode="dual"):
        super().__init__()
        self.n_sub = n_sub; self.n_vertices = n_vertices
        self.l_enc = TFMRIVertexEncoder(embed_dim=embed_dim, channels=channels, num_heads=num_heads,
                                        encoder_depth=encoder_depth, path_mode=path_mode)
        self.r_enc = TFMRIVertexEncoder(embed_dim=embed_dim, channels=channels, num_heads=num_heads,
                                        encoder_depth=encoder_depth, path_mode=path_mode)
        self.fusion = nn.Identity()
        self.head_l = nn.Linear(embed_dim * 2, n_sub * n_vertices)
        self.head_r = nn.Linear(embed_dim * 2, n_sub * n_vertices)
        if load_pretrained:
            for enc, ckpt, name in [(self.l_enc, lh_ckpt, "L"), (self.r_enc, rh_ckpt, "R")]:
                ck = torch.load(ckpt, map_location="cpu")
                sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
                filt = _filter_encoder_keys(sd)
                enc.load_state_dict(filt, strict=False)
                bg = enc.region_assigner.bias_generator
                print(f"[bilateral {name}] enc_keys={len(filt)} A={float(bg.log_A.exp()):.3f} "
                      f"sigma={float(bg.log_sigma.exp()):.3f}", flush=True)
        else:
            print("[bilateral] random encoders", flush=True)
        for h in [self.head_l, self.head_r]:
            nn.init.xavier_uniform_(h.weight)
            if h.bias is not None:
                nn.init.constant_(h.bias, 0)

    def forward(self, lf, lc, lh, rf, rc, rh):
        el = self.l_enc(lf, lc, lh)
        er = self.r_enc(rf, rc, rh)
        fused = self.fusion(torch.cat([el, er], dim=1))
        pl = self.head_l(fused).view(-1, self.n_sub, self.n_vertices)
        pr = self.head_r(fused).view(-1, self.n_sub, self.n_vertices)
        return pl, pr
