"""train_ados.py -- ADOS calibrated severity regression, two-stage transfer.

Stage 1 (frozen): the pre-trained encoders stay frozen; the bilateral patch
embeddings are read out by per-hemisphere moment alignment (PCA to 50 dims,
whitening truncated to 38 directions, rescaling to the input-feature
covariance) followed by an RBF-SVR.
Stage 2 (finetune): the encoder is fine-tuned progressively under a
differentiable readout (head-only probe, then the LayerNorm and
perspective-transform parameters of the last two encoder blocks), and the
finetuned embeddings go through the SAME moment-alignment + SVR readout.
Selection: the finetuned model is adopted only if its validation PCC exceeds
the frozen model by more than FT_MIN_DELTA on an inner validation site
(rotating over the remaining sites); the held-out test site is evaluated once
with the selected stage. The finetune stage runs without NAN normalization.
--probe-only reproduces the frozen-only pipeline.
"""
import os, sys, json, argparse, random, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.svm import SVR

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dataset_offline import OfflineFeatureDataset, compute_nan_stats
from meshmae_regressor import Mesh_regressor

PCA_DIM = 50
WHITEN_K = 38
RIDGE_L = 1e-3

SEED = 1
BATCH = 16
N_WORKER = 4
PROBE_LR, PROBE_MAX_EP, PROBE_PAT = 1e-3, 40, 8
FT_LR, FT_MAX_EP, FT_PAT = 5e-5, 35, 10
FT_WARMUP, FT_WARMUP_LR, FT_MIN_DELTA = 3, 1e-6, 0.005

_DECODER_PREFIX = ("decoder", "decoer_pos_embedding", "to_points", "to_features", "mask_token", "loss_func")


def check_ckpt_keys(net, sd, ckpt_path, tag):
    m, u = net.load_state_dict(sd, strict=False)
    enc_missing = [k for k in m if not k.startswith("head")]
    unex_ok = [k for k in u if k.startswith(_DECODER_PREFIX)]
    unex_other = [k for k in u if not k.startswith(_DECODER_PREFIX)]
    print(f"[ckpt:{tag}] missing(encoder)={len(enc_missing)} "
          f"unexpected(decoder-only)={len(unex_ok)} unexpected(other)={len(unex_other)}", flush=True)
    if enc_missing or unex_other:
        print(f"  [FAIL] missing={enc_missing[:5]} unexpected={unex_other[:5]}", flush=True)
        sys.exit(2)


def load_labels(label_dir):
    lab = np.load(f"{label_dir}/labels.npy")
    sids = np.load(f"{label_dir}/subject_ids.npy", allow_pickle=True)
    m = {str(s): float(v) for s, v in zip(sids, lab)}
    return m, [str(s) for s in sids]


def site_of(sid):
    return str(sid).split("_")[0]


def build_net(ckpt, device):
    net = Mesh_regressor(channels=10, num_heads=6, encoder_depth=6,
                         embed_dim=384, patch_size=64, drop_path=0.1).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    check_ckpt_keys(net, sd, ckpt, tag=os.path.basename(os.path.dirname(ckpt)))
    net.eval()
    cap = {}
    net._pool = cap
    net._pool_hook = net.norm.register_forward_hook(
        lambda mod, inp, out: cap.__setitem__("x", out))
    return net


@torch.no_grad()
def embed_with_net(net, ds, device, batch=16):
    net.eval()
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True)
    embs, sids = [], []
    for b in loader:
        feats, center, coord, faces, Fs, _, sid, hop = b
        net(faces.float().to(device), feats.float().to(device), center.float().to(device),
            coord.float().to(device), hop.float().to(device))
        embs.append(net._pool["x"][:, 1:, :].mean(dim=1).cpu().numpy())
        sids.extend([str(s) for s in sid])
    return np.concatenate(embs, 0).astype(np.float32), np.asarray(sids, dtype=object)


def embed_hemisphere(feat_dir, ckpt, device, batch=16, nan_stats=None):
    ds = OfflineFeatureDataset(feat_dir, canonical=True, ram_cache=True, dist="geodesic", nan_stats=nan_stats)
    net = build_net(ckpt, device)
    out = embed_with_net(net, ds, device, batch)
    del net
    torch.cuda.empty_cache()
    return out


class PairedHemisphereDataset(Dataset):
    """Paired LH/RH samples over a fixed subject list, indexed into full-dir datasets."""

    def __init__(self, ds_l, ds_r, subjs, y_map):
        lq = {s: i for i, s in enumerate(ds_l.sids)}
        rq = {s: i for i, s in enumerate(ds_r.sids)}
        self.ds_l, self.ds_r = ds_l, ds_r
        self.subjs = list(subjs)
        self.y_map = y_map
        self.idx_l = [lq[s] for s in self.subjs]
        self.idx_r = [rq[s] for s in self.subjs]

    def __getitem__(self, k):
        fl, cl, cdl, fal, _, _, _, hl = self.ds_l[self.idx_l[k]]
        fr, cr, cdr, far, _, _, _, hr = self.ds_r[self.idx_r[k]]
        return (fl, cl, cdl, fal, hl), (fr, cr, cdr, far, hr), \
            np.float32(self.y_map[self.subjs[k]]), self.subjs[k]

    def __len__(self):
        return len(self.subjs)


def subj_stats(data_path):
    d = np.load(data_path, mmap_mode="r"); n = d.shape[0]; o = np.zeros((n, 50), np.float32)
    for i in range(n):
        f = np.asarray(d[i], dtype=np.float32).reshape(10, -1)
        o[i] = np.concatenate([f.mean(1), f.std(1),
                               np.percentile(f, [10, 50, 90], axis=1).T.flatten()])
    return o


def fit_moment_alignment(Etr, Rtr, dim=PCA_DIM, ridge_l=RIDGE_L, whiten_k=WHITEN_K):
    muE = Etr.mean(0)
    Ec = Etr - muE
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    V = Vt[:dim].T
    Ep = Ec @ V
    muEp = Ep.mean(0)
    Epc = Ep - muEp
    covE = (Epc.T @ Epc) / len(Etr) + ridge_l * np.trace((Epc.T @ Epc) / len(Etr)) / dim * np.eye(dim)
    wE, VE = np.linalg.eigh(covE)
    wc = np.clip(wE, 1e-8, None)
    if whiten_k is not None and whiten_k < dim:
        wc[:-whiten_k] = 1.0
    WE = (VE @ np.diag(1.0 / np.sqrt(wc))) @ VE.T
    muR = Rtr.mean(0)
    Rc = Rtr - muR
    covR = (Rc.T @ Rc) / len(Rtr) + ridge_l * np.trace((Rc.T @ Rc) / len(Rtr)) / dim * np.eye(dim)
    wR, VR = np.linalg.eigh(covR)
    WR = (VR @ np.diag(np.sqrt(np.clip(wR, 0, None)))) @ VR.T

    def transform(X):
        return ((X - muE) @ V - muEp) @ WE @ WR + muR
    return transform


def metrics(p, t):
    p = p.astype(float); t = t.astype(float)
    pcc = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 and t.std() > 1e-9 else 0.0
    mae = float(np.abs(p - t).mean()); ss = ((t - t.mean()) ** 2).sum()
    return pcc, mae, float(1 - ((p - t) ** 2).sum() / max(ss, 1e-9))


def standardize(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def readout_predict(EMB, RAW, y, tr, ev):
    """Fit the readout chain (moment alignment + standardization + RBF-SVR) on rows tr, predict rows ev."""
    tL = fit_moment_alignment(EMB[tr][:, :384], RAW[tr][:, :50])
    tR = fit_moment_alignment(EMB[tr][:, 384:], RAW[tr][:, 50:])
    Xtr = np.concatenate([tL(EMB[tr][:, :384]), tR(EMB[tr][:, 384:])], 1)
    Xev = np.concatenate([tL(EMB[ev][:, :384]), tR(EMB[ev][:, 384:])], 1)
    Xtr, Xev = standardize(Xtr, Xev)
    return SVR(kernel="rbf", C=1.0, gamma="scale").fit(Xtr, y[tr]).predict(Xev)


def progressive_finetune(paired, tri, vai, ckpt_l, ckpt_r, dev, smoke=False):
    """Stage 2 machinery: probe (readout only) -> finetune (LayerNorm and
    perspective-transform of the last two blocks), Huber loss, validation-based
    state selection. Each fold restarts from the pretrained weights. Returns the
    two nets restored to the best state plus a small log dict."""
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    net_l, net_r = build_net(ckpt_l, dev), build_net(ckpt_r, dev)
    nets = (net_l, net_r)
    fusion = nn.Sequential(nn.LayerNorm(2 * 384), nn.Linear(2 * 384, 256), nn.GELU(),
                           nn.Dropout(0.3), nn.Linear(256, 1)).to(dev)

    def trainable_params():
        return [p for m in (*nets, fusion) for p in m.parameters() if p.requires_grad]

    def set_trainable(mode):
        for net in nets:
            for p in net.parameters():
                p.requires_grad = False
            if mode == "selective":
                for blk in net.blocks[-2:]:
                    for name, p in blk.named_parameters():
                        if any(k in name for k in ("norm1", "norm2", "perspective_transform")):
                            p.requires_grad = True
        for p in fusion.parameters():
            p.requires_grad = True
        return sum(p.numel() for p in trainable_params())

    def capture():
        return ([{k: v.detach().clone() for k, v in net.state_dict().items()} for net in nets],
                {k: v.detach().clone() for k, v in fusion.state_dict().items()})

    def restore(st):
        for net, sd in zip(nets, st[0]):
            net.load_state_dict(sd)
        fusion.load_state_dict(st[1])

    def run_epoch(loader, opt=None):
        is_train = opt is not None
        for m in (*nets, fusion):
            m.train() if is_train else m.eval()
        tot, n = 0.0, 0
        preds = []
        for (fl, cl, cdl, fal, hl), (fr, cr, cdr, far, hr), yb, _ in loader:
            yb = yb.float().to(dev)
            with torch.set_grad_enabled(is_train):
                net_l(fal.float().to(dev), fl.float().to(dev), cl.float().to(dev),
                      cdl.float().to(dev), hl.float().to(dev))
                net_r(far.float().to(dev), fr.float().to(dev), cr.float().to(dev),
                      cdr.float().to(dev), hr.float().to(dev))
                z = torch.cat([net_l._pool["x"][:, 1:, :].mean(1),
                               net_r._pool["x"][:, 1:, :].mean(1)], 1)
                pred = fusion(z).reshape(-1)
                loss = nn.functional.huber_loss(pred, yb, delta=1.0, reduction="mean")
            if is_train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params(), 1.0)
                opt.step()
            tot += float(loss.item()) * yb.shape[0]; n += yb.shape[0]
            preds.append(pred.detach().float().cpu().numpy())
        return tot / max(n, 1), np.concatenate(preds) if preds else np.zeros(0)

    tl = DataLoader(Subset(paired, tri.tolist()), batch_size=BATCH, shuffle=True, drop_last=True,
                    num_workers=N_WORKER, pin_memory=True)
    vl = DataLoader(Subset(paired, vai.tolist()), batch_size=BATCH, shuffle=False,
                    num_workers=N_WORKER, pin_memory=True)
    va_y = np.array([paired.y_map[paired.subjs[i]] for i in vai.tolist()], dtype=float)

    n_tr = set_trainable("head")
    opt = AdamW(trainable_params(), lr=PROBE_LR, weight_decay=0.05)
    probe_best = {"pcc": -9.0, "state": None, "ep": -1}
    since = 0
    for ep in range(3 if smoke else PROBE_MAX_EP):
        l, _ = run_epoch(tl, opt=opt)
        _, pv = run_epoch(vl)
        c = metrics(pv, va_y)[0]
        better = c > probe_best["pcc"] + 1e-6
        if better:
            probe_best = {"pcc": c, "state": capture(), "ep": ep}; since = 0
        else:
            since += 1
        print(f"  [probe] ep{ep:02d} loss={l:.4f} val_pcc={c:+.4f}{' *' if better else ''}", flush=True)
        if (not smoke) and since >= PROBE_PAT:
            break
    print(f"  [probe] best pcc={probe_best['pcc']:+.4f} @ ep{probe_best['ep']} (trainable={n_tr})", flush=True)

    n_tr = set_trainable("selective")
    max_ft = 3 if smoke else FT_MAX_EP
    warmup = min(max(FT_WARMUP, 0), max_ft)
    opt = AdamW(trainable_params(), lr=FT_WARMUP_LR if warmup > 0 else FT_LR, weight_decay=0.05)
    sched = None
    ft_best = {"pcc": -9.0, "state": None, "ep": -1}
    since = 0
    for ep in range(max_ft):
        if ep < warmup:
            frac = ep / max(warmup - 1, 1)
            for g in opt.param_groups:
                g["lr"] = FT_WARMUP_LR + frac * (FT_LR - FT_WARMUP_LR)
        elif sched is None:
            for g in opt.param_groups:
                g["lr"] = FT_LR
            sched = CosineAnnealingLR(opt, T_max=max(1, max_ft - warmup))
        l, _ = run_epoch(tl, opt=opt)
        if sched is not None:
            sched.step()
        _, pv = run_epoch(vl)
        c = metrics(pv, va_y)[0]
        better = c > ft_best["pcc"] + 1e-6
        if better:
            ft_best = {"pcc": c, "state": capture(), "ep": ep}; since = 0
        elif ep >= warmup:
            since += 1
        print(f"  [ft] ep{ep:02d} loss={l:.4f} val_pcc={c:+.4f}{' *' if better else ''}", flush=True)
        if (not smoke) and ep >= warmup and since >= FT_PAT:
            break
    print(f"  [ft] best pcc={ft_best['pcc']:+.4f} @ ep{ft_best['ep']} (trainable={n_tr})", flush=True)

    if ft_best["pcc"] > probe_best["pcc"] + FT_MIN_DELTA:
        restore(ft_best["state"]); stage = "ft"
    else:
        restore(probe_best["state"]); stage = "probe"
    log = {"probe_val_pcc": probe_best["pcc"], "probe_ep": probe_best["ep"],
           "ft_val_pcc": ft_best["pcc"], "ft_ep": ft_best["ep"], "selected_stage": stage}
    return net_l, net_r, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-lh", required=True,
                    help="path to the pretrained left-hemisphere encoder checkpoint")
    ap.add_argument("--ckpt-rh", required=True,
                    help="path to the pretrained right-hemisphere encoder checkpoint")
    ap.add_argument("--lh-dir", required=True,
                    help="ADOS left-hemisphere offline feature directory")
    ap.add_argument("--rh-dir", required=True,
                    help="ADOS right-hemisphere offline feature directory")
    ap.add_argument("--label-dir", required=True,
                    help="label directory (labels.npy / subject_ids.npy)")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--nan", action="store_true",
                    help="apply NAN normalization to encoder inputs (stats from each fold's training subjects)")
    ap.add_argument("--probe-only", action="store_true",
                    help="skip the finetune stage (frozen readout only)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick check: first site only, 3 epochs per stage")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
          f"probe_only={args.probe_only}", flush=True)
    for c in (args.ckpt_lh, args.ckpt_rh):
        if not os.path.exists(c):
            print(f"[FAIL] checkpoint not found: {c}", flush=True); sys.exit(2)

    ds_l = OfflineFeatureDataset(args.lh_dir, canonical=True, ram_cache=True, dist="geodesic")
    ds_r = OfflineFeatureDataset(args.rh_dir, canonical=True, ram_cache=True, dist="geodesic")
    net = build_net(args.ckpt_lh, dev)
    EL, SL = embed_with_net(net, ds_l, dev)
    del net; torch.cuda.empty_cache()
    net = build_net(args.ckpt_rh, dev)
    ER, SR = embed_with_net(net, ds_r, dev)
    del net; torch.cuda.empty_cache()
    print(f"[embed] LH {EL.shape} RH {ER.shape}", flush=True)

    lab_map, _ = load_labels(args.label_dir)
    rp = {s: i for i, s in enumerate(SR)}
    common = [s for s in SL if s in rp and s in lab_map]
    lp = {s: i for i, s in enumerate(SL)}
    EMB = np.concatenate([EL[[lp[s] for s in common]], ER[[rp[s] for s in common]]], 1).astype(np.float32)
    ls, rs = ds_l.sids, ds_r.sids
    lq = {s: i for i, s in enumerate(ls)}; rq = {s: i for i, s in enumerate(rs)}
    RL = subj_stats(f"{args.lh_dir}/data.npy")
    RR = subj_stats(f"{args.rh_dir}/data.npy")
    RAW = np.concatenate([RL[[lq[s] for s in common]], RR[[rq[s] for s in common]]], 1)
    y = np.array([lab_map[s] for s in common], dtype=np.float32)
    sites = np.array([site_of(s) for s in common])
    if not np.isfinite(EMB).all() or not np.isfinite(RAW).all():
        print("[FAIL] embeddings or input statistics contain NaN/Inf", flush=True); sys.exit(2)
    uniq = [s for s in sorted(set(sites.tolist())) if int((sites == s).sum()) > 20]
    print(f"N={len(common)} n>20 sites={ {s: int((sites == s).sum()) for s in uniq} }", flush=True)

    ldir = args.lh_dir; rdir = args.rh_dir
    l_data = np.load(ldir + "/data.npy", mmap_mode="r")
    l_cmap = np.load(ldir + "/canonical_mapping.npy")
    r_data = np.load(rdir + "/data.npy", mmap_mode="r")
    r_cmap = np.load(rdir + "/canonical_mapping.npy")

    outer = uniq[:1] if args.smoke else uniq
    paired = None if args.probe_only else PairedHemisphereDataset(ds_l, ds_r, common, lab_map)
    res, res_frz, sel = {}, {}, {}
    for oi, s in enumerate(outer):
        tr = np.where(sites != s)[0]; te = np.where(sites == s)[0]
        if args.nan:
            mu_l, sigma_l = compute_nan_stats(l_data, l_cmap, [lq[c] for c in common[tr]])
            mu_r, sigma_r = compute_nan_stats(r_data, r_cmap, [rq[c] for c in common[tr]])
            ELf, SLf = embed_hemisphere(ldir, args.ckpt_lh, dev, nan_stats={"mu": mu_l, "sigma": sigma_l})
            ERf, SRf = embed_hemisphere(rdir, args.ckpt_rh, dev, nan_stats={"mu": mu_r, "sigma": sigma_r})
            lp2 = {str(x): i for i, x in enumerate(SLf)}; rp2 = {str(x): i for i, x in enumerate(SRf)}
            EMB_f = np.concatenate([ELf[[lp2[c] for c in common]], ERf[[rp2[c] for c in common]]], 1).astype(np.float32)
        else:
            EMB_f = EMB

        if args.probe_only:
            sel[s] = {"phase": "frozen"}
            res_frz[s] = res[s] = metrics(readout_predict(EMB_f, RAW, y, tr, te), y[te])
            continue

        others = [u for u in uniq if u != s]
        v = others[oi % len(others)]
        tri = np.where((sites != s) & (sites != v))[0]
        vai = np.where(sites == v)[0]
        print(f"\n=== test={s}(n={len(te)}) val={v}(n={len(vai)}) train={len(tri)} ===", flush=True)

        if args.nan:
            mu_l, sigma_l = compute_nan_stats(l_data, l_cmap, [lq[c] for c in common[tri]])
            mu_r, sigma_r = compute_nan_stats(r_data, r_cmap, [rq[c] for c in common[tri]])
            ELv, SLv = embed_hemisphere(ldir, args.ckpt_lh, dev, nan_stats={"mu": mu_l, "sigma": sigma_l})
            ERv, SRv = embed_hemisphere(rdir, args.ckpt_rh, dev, nan_stats={"mu": mu_r, "sigma": sigma_r})
            lv2 = {str(x): i for i, x in enumerate(SLv)}; rv2 = {str(x): i for i, x in enumerate(SRv)}
            EMB_v = np.concatenate([ELv[[lv2[c] for c in common]], ERv[[rv2[c] for c in common]]], 1).astype(np.float32)
        else:
            EMB_v = EMB

        # ---- Stage 1: frozen ----
        frozen_val = metrics(readout_predict(EMB_v, RAW, y, tri, vai), y[vai])[0]

        # ---- Stage 2: finetune ----
        net_l, net_r, ftlog = progressive_finetune(paired, tri, vai, args.ckpt_lh, args.ckpt_rh, dev,
                                                   smoke=args.smoke)
        ELt, SLt = embed_with_net(net_l, ds_l, dev)
        ERt, SRt = embed_with_net(net_r, ds_r, dev)
        lt2 = {str(x): i for i, x in enumerate(SLt)}; rt2 = {str(x): i for i, x in enumerate(SRt)}
        EMB_t = np.concatenate([ELt[[lt2[c] for c in common]], ERt[[rt2[c] for c in common]]], 1).astype(np.float32)
        del net_l, net_r
        torch.cuda.empty_cache()
        ft_val = metrics(readout_predict(EMB_t, RAW, y, tri, vai), y[vai])[0]

        phase = "ft" if ft_val > frozen_val + FT_MIN_DELTA else "frozen"
        print(f"[{s}] val PCC frozen={frozen_val:+.4f} ft={ft_val:+.4f} -> phase={phase}", flush=True)
        EMB_sel = EMB_t if phase == "ft" else EMB_f
        res[s] = metrics(readout_predict(EMB_sel, RAW, y, tr, te), y[te])
        res_frz[s] = metrics(readout_predict(EMB_f, RAW, y, tr, te), y[te])
        sel[s] = {"phase": phase, "val_site": v, "frozen_val": float(frozen_val),
                  "ft_val": float(ft_val), "gradient_stage": ftlog}

    print(f"\n{'site':<8}{'n':>4}{'PCC':>9}{'MAE':>8}{'COD':>9}", flush=True)
    for s in outer:
        n = int((sites == s).sum())
        print(f"{s:<8}{n:>4}{res[s][0]:>+9.3f}{res[s][1]:>8.3f}{res[s][2]:>+9.3f}", flush=True)
    P = [res[s][0] for s in outer]; M = [res[s][1] for s in outer]; C = [res[s][2] for s in outer]
    print(f"{'mean':<8}{'':>4}{np.mean(P):>+9.3f}{np.mean(M):>8.3f}{np.mean(C):>+9.3f}"
          f"   (PCC +/-{np.std(P):.3f})", flush=True)
    if not args.probe_only:
        Pf = [res_frz[s][0] for s in outer]; Mf = [res_frz[s][1] for s in outer]; Cf = [res_frz[s][2] for s in outer]
        print(f"[frozen reference] PCC {np.mean(Pf):+.3f}+/-{np.std(Pf):.3f} | "
              f"MAE {np.mean(Mf):.3f} | COD {np.mean(Cf):+.3f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "task": "ADOS calibrated severity, leave-one-site-out",
        "protocol": ("two-stage: frozen encoder vs finetuned encoder under the same "
                     "moment-alignment + RBF-SVR readout, selected on inner-site "
                     f"val PCC (margin {FT_MIN_DELTA})" if not args.probe_only else "frozen only"),
        "n_subjects": len(common), "sites": {s: int((sites == s).sum()) for s in uniq},
        "pca_dim": PCA_DIM, "whiten_k": WHITEN_K, "ridge_l": RIDGE_L,
        "svr": "rbf C=1.0 gamma=scale",
        "ckpt_lh": args.ckpt_lh, "ckpt_rh": args.ckpt_rh, "nan": args.nan,
        "per_site": {s: list(res[s]) for s in outer},
        "per_site_frozen": {s: list(res_frz[s]) for s in outer},
        "phase_selection": sel,
        "summary": {"pcc_mean": float(np.mean(P)), "pcc_sd": float(np.std(P)),
                    "mae_mean": float(np.mean(M)), "cod_mean": float(np.mean(C))},
    }
    if not args.probe_only:
        out["summary_frozen"] = {"pcc_mean": float(np.mean(Pf)), "pcc_sd": float(np.std(Pf)),
                                 "mae_mean": float(np.mean(Mf)), "cod_mean": float(np.mean(Cf))}
    with open(f"{args.out_dir}/ados_loso.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_dir}/ados_loso.json", flush=True)


if __name__ == "__main__":
    main()
