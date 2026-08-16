import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_fc_offline import PairedFCDataset, _norm_sid
from meshmae_fc import BilateralFCNet


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def grouped_params(model, wd):
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n.lower() for nd in ["bias", "norm"]):
            nodecay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}]


def move_batch(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, dev, sbar=1.0):
    model.eval()
    preds_z, targs_z = [], []
    for b in loader:
        b = move_batch(b, dev)
        out = model(b["l_feats"], b["l_center"], b["l_hop"],
                    b["r_feats"], b["r_center"], b["r_hop"])
        preds_z.append(out.cpu().numpy())
        targs_z.append(b["target"].cpu().numpy())
    pz = np.concatenate(preds_z)
    tz = np.concatenate(targs_z)
    sp = [float(np.corrcoef(pz[i], tz[i])[0, 1]) for i in range(pz.shape[0])
          if pz[i].std() > 0 and tz[i].std() > 0]
    sample_pcc = float(np.mean(sp)) if sp else float("nan")
    rmse = float(sbar * np.sqrt(np.mean((pz - tz) ** 2)))
    return {"sample_pcc": sample_pcc, "rmse": rmse}


def set_trainable(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fusion.parameters():
        p.requires_grad = True
    if hasattr(model, "pool_l"):
        for p in model.pool_l.parameters():
            p.requires_grad = True
        for p in model.pool_r.parameters():
            p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [trainable] head-only params={n/1e6:.2f}M", flush=True)


def run_phase(model, train_loader, val_loader, dev, epochs, lr, wd, best, save_path, sbar=1.0, patience=5):
    set_trainable(model)
    opt = AdamW(grouped_params(model, wd), lr=lr)
    sch = CosineAnnealingLR(opt, T_max=max(1, epochs))
    since = 0
    best_val = best.get("val", -9.0)
    for ep in range(1, epochs + 1):
        model.train()
        model.left.eval(); model.right.eval()
        tl = 0.0
        for b in train_loader:
            b = move_batch(b, dev)
            out = model(b["l_feats"], b["l_center"], b["l_hop"],
                        b["r_feats"], b["r_center"], b["r_hop"])
            loss = F.mse_loss(out, b["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tl += float(loss.item()) * b["target"].shape[0]
        sch.step()
        m = evaluate(model, val_loader, dev, sbar)
        improved = m["sample_pcc"] > best_val
        if improved:
            best_val = m["sample_pcc"]
            best["val"] = m["sample_pcc"]
            best["val_loss"] = m["rmse"]
            torch.save(model.state_dict(), save_path)
            since = 0
        else:
            since += 1
        tag = "***BEST" if improved else ""
        print(f"  [head] ep{ep:03d} train_mse={tl/len(train_loader.dataset):.5f} "
              f"val sample_pcc={m['sample_pcc']:.4f} "
              f"rmse={m['rmse']:.4f} {tag}", flush=True)
        if since >= patience:
            print(f"  [head] early stop @ ep{ep} (patience={patience})", flush=True)
            break
    return best


def build_subject_folds(subjects, n_folds, seed):
    rng = random.Random(seed); sh = list(subjects); rng.shuffle(sh)
    chunks = np.array_split(np.asarray(sh, dtype=object), n_folds)
    folds = []
    for vi in range(n_folds):
        val = set(chunks[vi].tolist())
        tr = [s for s in sh if s not in val]
        folds.append((tr, list(chunks[vi].tolist())))
    return folds


def cohort_split(subjects, splits_csv, seed):
    sdf = pd.read_csv(splits_csv, dtype={"subject_id": str})
    sdf = sdf[sdf["seed"] == seed]
    if sdf.empty:
        raise SystemExit(f"seed {seed} not found in {splits_csv}")
    part = {}
    for _, r in sdf.iterrows():
        part[r["subject_id"]] = r["partition"]
        part[str(r["subject_id"]).zfill(6)] = r["partition"]
    tr = [s for s in subjects if part.get(s) == "train"]
    va = [s for s in subjects if part.get(s) == "val"]
    te = [s for s in subjects if part.get(s) == "test"]
    if not (tr and va and te):
        raise SystemExit(f"cohort split empty for seed {seed} "
                         f"(train={len(tr)} val={len(va)} test={len(te)} of {len(subjects)}); "
                         f"check that the cohort covers this task's subjects")
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lh-dir", required=True)
    ap.add_argument("--rh-dir", required=True)
    ap.add_argument("--fc-dir", required=True, help="FC label directory (with fc.npy, subject_ids.npy)")
    ap.add_argument("--lh-ckpt", required=True)
    ap.add_argument("--rh-ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", required=True,
                    help="frozen cohort splits csv; --seed selects the train/val/test partition")
    ap.add_argument("--cv-folds", type=int, default=1, help=">1 runs N-fold CV (head-only)")
    ap.add_argument("--random-encoder", action="store_true", help="skip pretrained weights (random encoder ablation)")
    ap.add_argument("--n-queries", type=int, default=8, help="attention pool query count")
    ap.add_argument("--save", required=True)
    ap.add_argument("--path-mode", choices=["dual", "local", "global"], default="dual",
                    help="path mode for BilateralFCNet HopEncoder Block (must match ckpt)")
    ap.add_argument("--dist", choices=["hop", "geodesic"], default="geodesic",
                    help="attention distance: hop | geodesic (geodesic requires geodesic pretrained ckpt)")
    ap.add_argument("--nan", action="store_true",
                    help="apply NAN normalization to feats (paired-set mu/sigma); use with NAN pretrained ckpt")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load existing ckpts and re-run evaluation (strict RMSE, writes .eval.json/.cv_eval.json)")
    args = ap.parse_args()
    BilateralFCNetCls = BilateralFCNet

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)

    fc = np.load(args.fc_dir + "/fc.npy")
    fc_sids_raw = np.load(args.fc_dir + "/subject_ids.npy", allow_pickle=True)
    fc_sids = {}
    for i, s in enumerate(fc_sids_raw):
        fc_sids.setdefault(_norm_sid(s), i)
    print(f"FC: {fc.shape} n_sids={len(fc_sids)}", flush=True)

    l_sids = {_norm_sid(s) for s in np.load(args.lh_dir + "/subject_ids.npy", allow_pickle=True)}
    r_sids = {_norm_sid(s) for s in np.load(args.rh_dir + "/subject_ids.npy", allow_pickle=True)}
    common = sorted((l_sids & r_sids) & set(fc_sids.keys()))
    print(f"common(L&R&FC)={len(common)}", flush=True)

    if args.cv_folds and args.cv_folds > 1:
        full = PairedFCDataset(args.lh_dir, args.rh_dir, fc, fc_sids, common, "sample_z", ram_cache=True, dist=args.dist, nan=args.nan)
        folds = build_subject_folds(common, args.cv_folds, args.seed)
        kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
        all_s, all_rmse = [], []
        for fi, (tr_sids, va_sids) in enumerate(folds):
            print(f"\n=== FC fold {fi+1}/{args.cv_folds} train={len(tr_sids)} val={len(va_sids)} ===", flush=True)
            tr_idx = [full.sids.index(s) for s in tr_sids]
            va_idx = [full.sids.index(s) for s in va_sids]
            sbar_f = float(full.fc_std[tr_idx].mean())
            tl = DataLoader(Subset(full, tr_idx), shuffle=True, drop_last=True, **kw)
            vl = DataLoader(Subset(full, va_idx), shuffle=False, **kw)
            model = BilateralFCNetCls(n_out=full.n_out, lh_ckpt=args.lh_ckpt, rh_ckpt=args.rh_ckpt,
                                      load_pretrained=not args.random_encoder,
                                      n_queries=args.n_queries,
                                      path_mode=args.path_mode).to(dev)
            if args.eval_only:
                model.load_state_dict(torch.load(f"{args.save}_fold{fi+1}.pkl", map_location=dev))
                print(f"  [eval-only] loaded {args.save}_fold{fi+1}.pkl", flush=True)
            else:
                ckpt = f"{args.save}_fold{fi+1}.pkl"
                run_phase(model, tl, vl, dev, args.epochs, args.lr, args.wd, {"val": -9.0}, ckpt, sbar=sbar_f)
                model.load_state_dict(torch.load(ckpt, map_location=dev))
            m = evaluate(model, vl, dev, sbar_f)
            all_s.append(m["sample_pcc"])
            all_rmse.append(m["rmse"])
            print(f"  fold{fi+1} sample_pcc={m['sample_pcc']:.4f} "
                  f"rmse={m['rmse']:.4f}", flush=True)
            torch.cuda.empty_cache()
        s = np.array(all_s); r = np.array(all_rmse)
        print("=" * 60, flush=True)
        print(f"[{args.cv_folds}-fold FC] "
              f"sample_pcc={s.mean():.4f}+/-{s.std():.4f} "
              f"rmse={r.mean():.4f}+/-{r.std():.4f}", flush=True)
        out_json = args.save + (".cv_eval.json" if args.eval_only else ".cv_result.json")
        with open(out_json, "w") as f:
            json.dump({"args": vars(args),
                       "sample_pcc_mean": float(s.mean()), "sample_pcc_std": float(s.std()),
                       "rmse_mean": float(r.mean()), "rmse_std": float(r.std())}, f, indent=2)
        print("done.", flush=True)
        return

    tr, va, te = cohort_split(common, args.splits, args.seed)
    print(f"cohort split seed={args.seed}: train={len(tr)} val={len(va)} test={len(te)} of {len(common)}", flush=True)

    ram = True
    ds_tr = PairedFCDataset(args.lh_dir, args.rh_dir, fc, fc_sids, tr, "sample_z", ram_cache=ram, dist=args.dist, nan=args.nan)
    _ns = ds_tr.nan_stats if args.nan else None
    ds_va = PairedFCDataset(args.lh_dir, args.rh_dir, fc, fc_sids, va, "sample_z", ram_cache=ram, dist=args.dist, nan=False, nan_stats=_ns)
    ds_te = PairedFCDataset(args.lh_dir, args.rh_dir, fc, fc_sids, te, "sample_z", ram_cache=ram, dist=args.dist, nan=False, nan_stats=_ns)
    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(ds_tr, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(ds_va, shuffle=False, **kw)
    test_loader = DataLoader(ds_te, shuffle=False, **kw)
    sbar_tr = float(ds_tr.fc_std.mean())

    model = BilateralFCNetCls(n_out=ds_tr.n_out, lh_ckpt=args.lh_ckpt, rh_ckpt=args.rh_ckpt,
                              load_pretrained=not args.random_encoder,
                              n_queries=args.n_queries,
                              path_mode=args.path_mode).to(dev)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params={nparams:.1f}M n_out={ds_tr.n_out}", flush=True)

    if args.eval_only:
        model.load_state_dict(torch.load(args.save, map_location=dev))
        vm = evaluate(model, val_loader, dev, sbar_tr)
        tm = evaluate(model, test_loader, dev, sbar_tr)
        print("=" * 60, flush=True)
        print(f"VAL   [eval-only] sample_pcc={vm['sample_pcc']:.4f} rmse={vm['rmse']:.4f}", flush=True)
        print(f"TEST  [eval-only] sample_pcc={tm['sample_pcc']:.4f} rmse={tm['rmse']:.4f}", flush=True)
        with open(args.save + ".eval.json", "w") as f:
            json.dump({"args": vars(args), "val": vm, "test": tm}, f, indent=2)
        print("done.", flush=True)
        return

    best = {"val": -9.0}
    print("=== head-only training (frozen encoder + attention pool) ===", flush=True)
    best = run_phase(model, train_loader, val_loader, dev, args.epochs, args.lr, args.wd, best, args.save, sbar=sbar_tr)
    print(f"best_val={best['val']:.4f}", flush=True)

    model.load_state_dict(torch.load(args.save, map_location=dev))
    vm = evaluate(model, val_loader, dev, sbar_tr)
    tm = evaluate(model, test_loader, dev, sbar_tr)
    print("=" * 60, flush=True)
    print(f"VAL   sample_pcc={vm['sample_pcc']:.4f} rmse={vm['rmse']:.4f}", flush=True)
    print(f"TEST  sample_pcc={tm['sample_pcc']:.4f} rmse={tm['rmse']:.4f}", flush=True)
    with open(args.save + ".result.json", "w") as f:
        json.dump({"args": vars(args), "val": vm, "test": tm}, f, indent=2)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
