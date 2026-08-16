import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

_THIS_DIR = Path(__file__).resolve().parent
_CT_ROOT = os.environ.get("CSML_CT_ROOT", str(_THIS_DIR))
if _CT_ROOT not in sys.path:
    sys.path.insert(0, _CT_ROOT)

from torch.utils.data import DataLoader
from dataset_offline import compute_nan_stats
from dataset_paired_offline import PairedOfflineFeatureDataset

_fusion_mlp = None
_PROBE_ONLY = False
from meshmae_regressor import Mesh_regressor


PHENO_28 = [
    "MMSE_Score", "PSQI_Score", "PicSeq_Unadj", "CardSort_Unadj", "Flanker_Unadj",
    "PMAT24_A_SI", "ReadEng_Unadj", "PicVocab_Unadj", "ProcSpeed_Unadj", "DDisc_SV_6mo_40K",
    "VSPLOT_TC", "SCPT_SEN", "ListSort_Unadj", "CogFluidComp_Unadj", "CogEarlyComp_Unadj",
    "CogTotalComp_Unadj", "CogCrystalComp_Unadj", "Sadness_Unadj", "PercHostil_Unadj",
    "Emotion_Task_Acc", "Language_Task_Acc", "Relational_Task_Acc", "WM_Task_Acc",
    "Social_Task_Perc_Random", "Endurance_Unadj", "Strength_Unadj", "NEOFAC_A", "Noise_Comp",
]

HCP_LH_DIR = None  # set from --lh-dir in main()
HCP_RH_DIR = None  # set from --rh-dir in main()
COHORT_DIR = None  # set from --cohort-dir in main()


class HuberLoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        return F.huber_loss(pred, target, delta=1.0, reduction="mean")


def compute_train_zscore(df_train, phenotype):
    y = df_train[phenotype].values.astype(np.float64)
    mu = float(y.mean())
    sigma = float(y.std())
    if sigma < 1e-8:
        sigma = 1.0
    return mu, sigma


def build_z_map(df, mu, sigma, phenotype):
    out = {}
    for _, row in df.iterrows():
        s = row["Subject"]
        y_raw = float(row[phenotype])
        z = (y_raw - mu) / sigma
        out[s] = (y_raw, z)
    return out


def set_trainable_texstyle(nets):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = False
        for p in net.head.parameters():
            p.requires_grad = True
        for blk in [net.blocks[-2], net.blocks[-1]]:
            for name, p in blk.named_parameters():
                if any(k in name for k in ("norm1", "norm2", "perspective_transform")):
                    p.requires_grad = True
    for p in _fusion_mlp.parameters():
        p.requires_grad = True
    return sum(p.numel() for net in nets for p in net.parameters() if p.requires_grad) + \
           sum(p.numel() for p in _fusion_mlp.parameters() if p.requires_grad)


def set_trainable_head_only(nets):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = False
        for p in net.head.parameters():
            p.requires_grad = True
    for p in _fusion_mlp.parameters():
        p.requires_grad = True
    return sum(p.numel() for net in nets for p in net.parameters() if p.requires_grad) + \
           sum(p.numel() for p in _fusion_mlp.parameters() if p.requires_grad)


def trainable_parameters(nets):
    params = [
        p for net in nets for p in net.parameters()
        if p.requires_grad
    ]
    params.extend(p for p in _fusion_mlp.parameters() if p.requires_grad)
    if not params:
        raise RuntimeError("no trainable parameters; check bilateral fusion setup")
    return params


def capture_training_state(nets):
    state = {
        f"net{i}_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in net.state_dict().items()
        }
        for i, net in enumerate(nets)
    }
    state["fusion_mlp_state_dict"] = {
        key: value.detach().cpu().clone()
        for key, value in _fusion_mlp.state_dict().items()
    }
    return state


def restore_training_state(nets, state):
    for i, net in enumerate(nets):
        net.load_state_dict(state[f"net{i}_state_dict"])
    _fusion_mlp.load_state_dict(state["fusion_mlp_state_dict"])


def build_model_pairs(ckpt_paths, model_kw, device, load_ckpt=True):
    pairs = []
    for ckpt in ckpt_paths:
        net = Mesh_regressor(**model_kw).to(device)
        pairs.append((net, ckpt if load_ckpt else None))
    for net, ckpt in pairs:
        if ckpt is None:
            print("  [init] from scratch: no pretrained checkpoint loaded", flush=True)
            continue
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        m, u = net.load_state_dict(sd, strict=False)
        enc_missing = [k for k in m if not k.startswith("head")]
        if enc_missing:
            print(f"  [WARN] ckpt={ckpt} encoder missing {len(enc_missing)}: {enc_missing[:3]}",
                  flush=True)
    return pairs


def _sid_to_index(feat_dir, subjs):
    sids_arr = [str(int(s)) for s in np.load(f"{feat_dir}/subject_ids.npy", allow_pickle=True)]
    sid2idx = {}
    for i, s in enumerate(sids_arr):
        sid2idx.setdefault(s, i)
    return [sid2idx[s] for s in subjs]


def _compute_nan_for_dir(feat_dir, train_subjs):
    train_idx = _sid_to_index(feat_dir, train_subjs)
    data_mmap = np.load(f"{feat_dir}/data.npy", mmap_mode="r")
    cmap = np.load(f"{feat_dir}/canonical_mapping.npy")
    mu_j, sig_j = compute_nan_stats(data_mmap, cmap, train_idx)
    return {"mu": mu_j, "sigma": sig_j}, len(train_idx)


def build_datasets(exp_cfg, splits_seed, z_map):
    dist = exp_cfg.get("dist", "hop")
    nan_on = exp_cfg.get("nan", False)

    nan_stats_l = nan_stats_r = None
    if nan_on:
        train_subjs = splits_seed["train"]
        nan_stats_l, n_l = _compute_nan_for_dir(HCP_LH_DIR, train_subjs)
        nan_stats_r, n_r = _compute_nan_for_dir(HCP_RH_DIR, train_subjs)
        print(f"[NAN] paired train-only stats: L on {n_l}, R on {n_r} subjects", flush=True)

    ds_dict = {}
    for part, subjs in [("train", splits_seed["train"]),
                        ("val",   splits_seed["val"]),
                        ("test",  splits_seed["test"])]:
        ph_map = {s: z_map[s][1] for s in subjs}
        ds = PairedOfflineFeatureDataset(
            HCP_LH_DIR, HCP_RH_DIR, subjs, ph_map,
            ram_cache=True, nan=nan_on, dist=dist,
            nan_stats_l=nan_stats_l, nan_stats_r=nan_stats_r,
        )
        ds_dict[part] = (ds, subjs)
    return ds_dict


def make_loader(ds, batch, n_worker, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=n_worker, pin_memory=True, drop_last=shuffle)


def to_dev(batch, device):
    (lf, lc, lcd, lf2, lFs, lhop, rf, rc, rcd, rf2, rFs, rhop, label, sid) = batch
    lf = lf.float().to(device, non_blocking=True); lc = lc.float().to(device, non_blocking=True)
    lcd = lcd.float().to(device, non_blocking=True); lhop = lhop.to(device, non_blocking=True)
    rf = rf.float().to(device, non_blocking=True); rc = rc.float().to(device, non_blocking=True)
    rcd = rcd.float().to(device, non_blocking=True); rhop = rhop.to(device, non_blocking=True)
    y = torch.as_tensor(label, dtype=torch.float32, device=device)
    inputs = (lf2, lf, lc, lcd, lhop, rf2, rf, rc, rcd, rhop)
    return inputs, y, sid


def forward_pairs(nets, inputs):
    lf2, lf, lc, lcd, lhop, rf2, rf, rc, rcd, rhop = inputs
    fl = nets[0](lf2, lf, lc, lcd, lhop)
    fr = nets[1](rf2, rf, rc, rcd, rhop)
    return _fusion_mlp(torch.cat([fl, fr], dim=1))


def run_epoch(nets, loader, device, crit, opt=None):
    is_train = opt is not None
    for net in nets:
        if is_train and _PROBE_ONLY:
            net.eval()
        else:
            net.train() if is_train else net.eval()
    _fusion_mlp.train() if is_train else _fusion_mlp.eval()

    losses = {"huber": 0.0, "total": 0.0}
    preds_list, sids_list = [], []
    n_seen = 0
    for batch in loader:
        inputs, y, sid = to_dev(batch, device)
        if is_train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = forward_pairs(nets, inputs)
            huber = crit(pred, y); total = huber
        if is_train:
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for net in nets for p in net.parameters() if p.requires_grad], 1.0
            )
            opt.step()
        bs = y.shape[0]
        losses["huber"] += float(huber.item()) * bs
        losses["total"] += float(total.item()) * bs
        n_seen += bs
        preds_list.append(pred.detach().reshape(-1).cpu().numpy())
        sids_list.extend(sid)
    for k in losses:
        losses[k] = losses[k] / max(n_seen, 1)
    return losses, np.concatenate(preds_list), sids_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lh-ckpt", required=True,
                    help="path to the pretrained left-hemisphere encoder checkpoint")
    ap.add_argument("--rh-ckpt", required=True,
                    help="path to the pretrained right-hemisphere encoder checkpoint")
    ap.add_argument("--lh-dir", required=True,
                    help="left-hemisphere offline feature directory")
    ap.add_argument("--rh-dir", required=True,
                    help="right-hemisphere offline feature directory")
    ap.add_argument("--cohort-dir", required=True,
                    help="cohort directory (cohort.csv / splits.csv)")
    ap.add_argument("--dist", choices=["hop", "geodesic"], default="geodesic",
                    help="attention distance of the pretrained encoder")
    ap.add_argument("--path-mode", choices=["dual", "local", "global"], default="dual",
                    help="encoder pathway configuration (must match ckpt)")
    ap.add_argument("--nan", action="store_true",
                    help="apply NAN normalization (stats from the split's training subjects); use with NAN pretrained ckpts")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--phenotype", required=True, choices=PHENO_28)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--n-worker", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--probe-only", action="store_true",
                    help="head-only probe only, no finetune phase")
    ap.add_argument("--from-scratch", action="store_true",
                    help="do not load a pretrained ckpt (randomly initialized encoder)")
    ap.add_argument("--probe-epochs", type=int, default=50)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-patience", type=int, default=10)
    ap.add_argument("--ft-epochs", type=int, default=45)
    ap.add_argument("--ft-patience", type=int, default=15)
    ap.add_argument("--min-delta", type=float, default=1e-6)
    ap.add_argument("--ft-warmup-epochs", type=int, default=3)
    ap.add_argument("--ft-warmup-start-lr", type=float, default=1e-6)
    ap.add_argument("--results-root", required=True, help="result root directory")
    ap.add_argument("--run-name", required=True,
                    help="run name used in the output directory name")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: 3 ep, no ckpt save, just verify pipeline")
    ap.add_argument("--embed-dim", type=int, default=384)
    ap.add_argument("--encoder-depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--channels", type=int, default=10)
    ap.add_argument("--drop-path", type=float, default=0.1)
    args = ap.parse_args()
    global HCP_LH_DIR, HCP_RH_DIR, COHORT_DIR
    HCP_LH_DIR, HCP_RH_DIR, COHORT_DIR = args.lh_dir, args.rh_dir, args.cohort_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)

    exp_cfg = dict(dataset="paired", hemi="bilateral", dist=args.dist,
                   nan=args.nan, path_mode=args.path_mode)
    ckpt_paths = [args.lh_ckpt, args.rh_ckpt]
    run_name = args.run_name
    global _PROBE_ONLY
    _PROBE_ONLY = args.probe_only
    print(f"[task] {run_name} seed={args.seed} pheno={args.phenotype} "
          f"probe_only={args.probe_only}", flush=True)
    print(f"[data] LH={HCP_LH_DIR} RH={HCP_RH_DIR} path_mode={exp_cfg.get('path_mode', 'dual')}", flush=True)

    cdf = pd.read_csv(f"{COHORT_DIR}/cohort.csv", dtype={"Subject": str})
    cdf["Subject"] = cdf["Subject"].str.zfill(6)
    sdf = pd.read_csv(f"{COHORT_DIR}/splits.csv", dtype={"subject_id": str})
    sdf["subject_id"] = sdf["subject_id"].str.zfill(6)
    splits_seed = {
        part: sdf[(sdf["seed"] == args.seed) & (sdf["partition"] == part)]["subject_id"].tolist()
        for part in ("train", "val", "test")
    }
    assert len(splits_seed["train"]) == 653
    assert len(splits_seed["val"]) == 73
    assert len(splits_seed["test"]) == 81

    df_train = cdf[cdf["Subject"].isin(splits_seed["train"])].copy()
    df_val   = cdf[cdf["Subject"].isin(splits_seed["val"])].copy()
    df_test  = cdf[cdf["Subject"].isin(splits_seed["test"])].copy()

    mu_p, sigma_p = compute_train_zscore(df_train, args.phenotype)
    print(f"[target] raw phenotype train stats: mu={mu_p:.4f} sigma={sigma_p:.4f}", flush=True)

    z_map_tr  = build_z_map(df_train, mu_p, sigma_p, args.phenotype)
    z_map_va  = build_z_map(df_val,   mu_p, sigma_p, args.phenotype)
    z_map_te  = build_z_map(df_test,  mu_p, sigma_p, args.phenotype)
    z_map_all = {**z_map_tr, **z_map_va, **z_map_te}

    ds_dict = build_datasets(exp_cfg, splits_seed, z_map_all)
    train_loader = make_loader(ds_dict["train"][0], args.batch, args.n_worker, shuffle=True)
    val_loader   = make_loader(ds_dict["val"][0],   args.batch, args.n_worker, shuffle=False)

    model_kw = dict(channels=args.channels, num_heads=args.heads,
                    encoder_depth=args.encoder_depth, embed_dim=args.embed_dim,
                    patch_size=args.patch_size, drop_path=args.drop_path,
                    path_mode=exp_cfg.get("path_mode", "dual"))
    pairs = build_model_pairs(ckpt_paths, model_kw, device, load_ckpt=not args.from_scratch)
    nets = [p[0] for p in pairs]
    print(f"[ckpt] loaded {len(nets)} encoder(s) "
          f"(from_scratch={args.from_scratch})", flush=True)

    global _fusion_mlp
    for net in nets:
        net.head = nn.Identity()
    _fusion_mlp = nn.Sequential(
        nn.Linear(args.embed_dim * 2, 128), nn.ReLU(), nn.Linear(128, 1)
    ).to(device)
    print(f"[fusion] concat MLP: {sum(p.numel() for p in _fusion_mlp.parameters())} params", flush=True)

    crit = HuberLoss()

    out_dir = None
    if not args.smoke:
        out_dir = Path(args.results_root) / f"{run_name}__seed{args.seed}__{args.phenotype}"
        out_dir.mkdir(parents=True, exist_ok=True)

    epoch_log = []
    best_val_huber = float("inf")
    best_epoch = -1
    best_phase = ""
    probe_best_val_huber = None
    probe_best_epoch = None
    ft_best_val_huber = None
    ft_best_epoch = None
    global_best_state = None

    n_tr = set_trainable_head_only(nets)
    params = trainable_parameters(nets)
    opt = torch.optim.AdamW(params, lr=args.probe_lr, weight_decay=args.weight_decay)
    max_probe = 3 if args.smoke else args.probe_epochs
    print(f"== Phase 1 (Probe): head-only trainable={n_tr} lr={args.probe_lr} "
          f"max_ep={max_probe} patience={args.probe_patience} ==", flush=True)

    since = 0
    probe_best_val_huber = float("inf")
    probe_best_epoch = -1
    probe_best_state = None
    for ep in range(max_probe):
        tr_loss, _, _ = run_epoch(nets, train_loader, device, crit, opt=opt)
        va_loss, _, _ = run_epoch(nets, val_loader, device, crit)
        epoch_log.append({
            "epoch": ep, "phase": "probe",
            "train_huber": tr_loss["huber"], "train_total": tr_loss["total"],
            "val_huber":   va_loss["huber"], "val_total":   va_loss["total"],
        })
        mark = ""
        if va_loss["huber"] < probe_best_val_huber - args.min_delta:
            probe_best_val_huber = va_loss["huber"]
            probe_best_epoch = ep
            probe_best_state = capture_training_state(nets)
            since = 0
            mark = " *best probe*"
            if out_dir is not None:
                torch.save(probe_best_state, out_dir / "ckpt_probe_best.pt")
        else:
            since += 1
        print(f"ep{ep:02d}(probe) tr_h={tr_loss['huber']:.4f} "
              f"| VAL h={va_loss['huber']:.4f} {mark}", flush=True)
        if (not args.smoke) and since >= args.probe_patience:
            print(f"  probe early stop @ ep{ep} (patience={args.probe_patience})", flush=True)
            break

    if probe_best_state is None:
        raise RuntimeError("probe phase did not produce a valid checkpoint")
    print(f"[probe] best val={probe_best_val_huber:.4f} "
          f"@ ep{probe_best_epoch}", flush=True)

    restore_training_state(nets, probe_best_state)

    n_tr = set_trainable_texstyle(nets)
    params = trainable_parameters(nets)
    ft_lr = args.lr
    max_ft = 0 if args.probe_only else (3 if args.smoke else args.ft_epochs)
    warmup_epochs = min(max(args.ft_warmup_epochs, 0), max_ft)
    initial_lr = args.ft_warmup_start_lr if warmup_epochs > 0 else ft_lr
    opt = torch.optim.AdamW(params, lr=initial_lr, weight_decay=args.weight_decay)
    scheduler = None
    print(f"== Phase 2 (Finetune): texstyle trainable={n_tr} lr={ft_lr} "
          f"max_ep={max_ft} warmup={warmup_epochs} patience={args.ft_patience} ==",
          flush=True)
    print(f"   probe reference: val={probe_best_val_huber:.4f}", flush=True)

    since = 0
    ft_best_val_huber = float("inf")
    ft_best_epoch = -1
    ft_best_state = None
    for ep in range(max_ft):
        if ep < warmup_epochs:
            if warmup_epochs == 1:
                current_lr = ft_lr
            else:
                fraction = ep / max(warmup_epochs - 1, 1)
                current_lr = (
                    args.ft_warmup_start_lr
                    + fraction * (ft_lr - args.ft_warmup_start_lr)
                )
            for group in opt.param_groups:
                group["lr"] = current_lr
        elif scheduler is None:
            for group in opt.param_groups:
                group["lr"] = ft_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, max_ft - warmup_epochs)
            )

        tr_loss, _, _ = run_epoch(nets, train_loader, device, crit, opt=opt)
        va_loss, _, _ = run_epoch(nets, val_loader, device, crit)
        if scheduler is not None:
            scheduler.step()
        epoch_log.append({
            "epoch": ep, "phase": "ft",
            "train_huber": tr_loss["huber"], "train_total": tr_loss["total"],
            "val_huber":   va_loss["huber"], "val_total":   va_loss["total"],
            "lr": opt.param_groups[0]["lr"],
        })
        mark = ""
        if va_loss["huber"] < ft_best_val_huber - args.min_delta:
            ft_best_val_huber = va_loss["huber"]
            ft_best_epoch = ep
            ft_best_state = capture_training_state(nets)
            since = 0
            mark = " *best ft*"
            if out_dir is not None:
                torch.save(ft_best_state, out_dir / "ckpt_ft_best.pt")
        elif ep >= warmup_epochs:
            since += 1
        print(f"ep{ep:02d}(ft) tr_h={tr_loss['huber']:.4f} "
              f"| VAL h={va_loss['huber']:.4f} "
              f"| lr={opt.param_groups[0]['lr']:.3g}{mark}", flush=True)
        if (not args.smoke) and ep >= warmup_epochs and since >= args.ft_patience:
            print(f"  ft early stop @ ep{ep} (patience={args.ft_patience})", flush=True)
            break

    if ft_best_state is None and not args.probe_only:
        raise RuntimeError("finetune phase did not produce a valid checkpoint")

    if ft_best_val_huber < probe_best_val_huber - args.min_delta:
        global_best_state = ft_best_state
        best_val_huber = ft_best_val_huber
        best_epoch = ft_best_epoch
        best_phase = "ft"
    else:
        global_best_state = probe_best_state
        best_val_huber = probe_best_val_huber
        best_epoch = probe_best_epoch
        best_phase = "probe"

    restore_training_state(nets, global_best_state)
    if out_dir is not None:
        torch.save(global_best_state, out_dir / "ckpt_global_best.pt")
        torch.save(global_best_state, out_dir / "ckpt.pt")
    print(f"[selection] probe val={probe_best_val_huber:.4f} @ ep{probe_best_epoch}; "
          f"ft val={ft_best_val_huber:.4f} @ ep{ft_best_epoch}; "
          f"selected={best_phase}", flush=True)

    rows = []
    test_pred_z = []; test_true_z = []
    for part, (ds, _) in ds_dict.items():
        loader = make_loader(ds, args.batch, args.n_worker, shuffle=False)
        _, preds, sids = run_epoch(nets, loader, device, crit)
        for sid_raw, pred_z in zip(sids, preds):
            sid = str(sid_raw)
            sid = sid.zfill(6) if sid.isdigit() and len(sid) < 6 else sid
            y_raw, y_true_z = z_map_all[sid]
            y_pred_raw = float(pred_z) * sigma_p + mu_p
            rows.append({
                "experiment_id": run_name,
                "split_seed":    args.seed,
                "phenotype":     args.phenotype,
                "partition":     part,
                "subject_id":    sid,
                "phenotype_mean_train": mu_p,
                "phenotype_std_train":  sigma_p,
                "y_true_raw":    float(y_raw),
                "y_true_z":      float(y_true_z),
                "y_pred_z":      float(pred_z),
                "y_pred_raw":    y_pred_raw,
                "best_epoch":    best_epoch,
                "best_phase":    best_phase,
                "checkpoint_path": str(out_dir / "ckpt.pt") if out_dir else "",
            })
            if part == "test":
                test_pred_z.append(float(pred_z)); test_true_z.append(float(y_true_z))

    y_t = np.asarray(test_true_z); y_p = np.asarray(test_pred_z)
    pcc, p_raw = pearsonr(y_t, y_p)
    ss_res = float(((y_p - y_t) ** 2).sum())
    ss_tot = float(((y_t - y_t.mean()) ** 2).sum())
    cod = float(1.0 - ss_res / max(ss_tot, 1e-12))
    srmse = float(np.sqrt(((y_p - y_t) ** 2).mean()))
    metrics = {
        "experiment_id": run_name, "split_seed": args.seed,
        "phenotype":     args.phenotype,
        "fusion": "concat_mlp",
        "n_test": len(y_t), "pcc": float(pcc), "p_raw_pcc": float(p_raw),
        "cod": cod, "standardized_rmse": srmse,
        "best_epoch": best_epoch, "best_phase": best_phase, "best_val_huber": best_val_huber,
        "probe_best_epoch": probe_best_epoch,
        "probe_best_val_huber": probe_best_val_huber,
        "ft_best_epoch": ft_best_epoch,
        "ft_best_val_huber": ft_best_val_huber,
        "phenotype_mean_train": mu_p, "phenotype_std_train": sigma_p,
        "n_train": len(splits_seed["train"]),
        "ckpt": str(out_dir / "ckpt.pt") if out_dir else "",
    }
    print(f"\n[TEST] PCC={pcc:.4f} (p={p_raw:.4g}) COD={cod:.4f} sRMSE={srmse:.4f} "
          f"(best_ep={best_epoch})", flush=True)

    if out_dir is not None:
        pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
        pd.DataFrame(epoch_log).to_csv(out_dir / "epoch_log.csv", index=False)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[save] {out_dir}/", flush=True)
    else:
        print(f"[smoke] no artifacts saved.", flush=True)


if __name__ == "__main__":
    main()
