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
from dataset_tfmri_paired import PairedTfMRIDataset, _norm_sid
from meshmae_tfmri_bilateral import TfMRIBilateralNet


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def grouped_params(model, wd):
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay if any(nd in n.lower() for nd in ["bias", "norm"]) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}]


def move(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def pcc(a, b):
    a = a.astype(np.float64).flatten(); b = b.astype(np.float64).flatten()
    fin = np.isfinite(a) & np.isfinite(b)
    if a[fin].std() == 0 or b[fin].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[fin], b[fin])[0, 1])


@torch.no_grad()
def evaluate(model, loader, dev, sbar_l, sbar_r, vm_l, vm_r):
    model.eval()
    Pl, Tl, Ml, Sl, Pr, Tr, Mr, Sr = [], [], [], [], [], [], [], []
    for b in loader:
        b = move(b, dev)
        pl, pr = model(b["lf"], b["lc"], b["lhop"], b["rf"], b["rc"], b["rhop"])
        Pl.append(pl.cpu().numpy()); Tl.append(b["l_tz"].cpu().numpy())
        Ml.append(b["l_mean"].cpu().numpy()); Sl.append(b["l_std"].cpu().numpy())
        Pr.append(pr.cpu().numpy()); Tr.append(b["r_tz"].cpu().numpy())
        Mr.append(b["r_mean"].cpu().numpy()); Sr.append(b["r_std"].cpu().numpy())
    Pl = np.concatenate(Pl); Tl = np.concatenate(Tl)
    Ml = np.concatenate(Ml); Sl = np.concatenate(Sl)
    Pr = np.concatenate(Pr); Tr = np.concatenate(Tr)
    Mr = np.concatenate(Mr); Sr = np.concatenate(Sr)

    def native(P, T, M, S):
        Tn = T * S[:, :, None] + M[:, :, None]
        Pn = P * S[:, :, None] + M[:, :, None]
        return Pn, Tn

    Pln, Tln = native(Pl, Tl, Ml, Sl)
    Prn, Trn = native(Pr, Tr, Mr, Sr)

    def per_sub_masked(Pn, Tn, vm):
        return [pcc(Pn[:, s][:, vm], Tn[:, s][:, vm]) for s in range(Pn.shape[1])]
    def rmse_sub_strict(P, T, sbar, vm):
        return [float(np.sqrt(np.nanmean(((P[:, s][:, vm] - T[:, s][:, vm]) * sbar[s]) ** 2))) for s in range(P.shape[1])]

    l_mask = per_sub_masked(Pln, Tln, vm_l)
    r_mask = per_sub_masked(Prn, Trn, vm_r)
    l_rmse = rmse_sub_strict(Pl, Tl, sbar_l, vm_l)
    r_rmse = rmse_sub_strict(Pr, Tr, sbar_r, vm_r)
    return {"l_mean": float(np.nanmean(l_mask)), "r_mean": float(np.nanmean(r_mask)),
            "l_rmse_mean": float(np.nanmean(l_rmse)), "r_rmse_mean": float(np.nanmean(r_rmse))}


def set_trainable(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fusion.parameters():
        p.requires_grad = True
    for p in model.head_l.parameters():
        p.requires_grad = True
    for p in model.head_r.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [trainable] head-only {n/1e6:.2f}M", flush=True)


def build_subject_folds(subjects, n_folds, seed):
    rng = random.Random(seed); sh = list(subjects); rng.shuffle(sh)
    chunks = np.array_split(np.asarray(sh, dtype=object), n_folds)
    return [([s for s in sh if s not in set(chunks[vi].tolist())], list(chunks[vi].tolist()))
            for vi in range(n_folds)]


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
    ap.add_argument("--lh-dir", required=True); ap.add_argument("--rh-dir", required=True)
    ap.add_argument("--l-target-prefix", required=True); ap.add_argument("--r-target-prefix", required=True)
    ap.add_argument("--lh-ckpt", required=True); ap.add_argument("--rh-ckpt", required=True)
    ap.add_argument("--cc-l", default=None); ap.add_argument("--cc-r", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", required=True,
                    help="frozen cohort splits csv; --seed selects the train/val/test partition")
    ap.add_argument("--cv-folds", type=int, default=1,
                    help="1 = paper protocol (frozen split selected by --splits + --seed); >1 runs N-fold subject CV")
    ap.add_argument("--random-encoder", action="store_true")
    ap.add_argument("--save", required=True)
    ap.add_argument("--path-mode", choices=["dual", "local", "global"], default="dual",
                    help="ablation: passed to TfMRIBilateralNet encoder Block (must match ckpt)")
    ap.add_argument("--dist", choices=["hop", "geodesic"], default="geodesic",
                    help="ablation: attention distance hop|geodesic (geodesic requires geodesic pretrained ckpt)")
    ap.add_argument("--nan", action="store_true",
                    help="apply NAN normalization to feats (stats from the split's training subjects); use with NAN pretrained ckpt")
    ap.add_argument("--exclude-subtasks", default="",
                    help="comma-separated subtask indices to exclude (0-based), e.g. '2' excludes GAM_3")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load existing ckpts and re-run evaluation (strict RMSE uses train-fold std)")
    args = ap.parse_args()
    exclude_subs = [int(x) for x in args.exclude_subtasks.split(",") if x.strip()] if args.exclude_subtasks else None
    Cls = TfMRIBilateralNet

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} cv_folds={args.cv_folds}", flush=True)

    vm_l = np.ones(32492, dtype=bool)
    if args.cc_l:
        cc = np.loadtxt(args.cc_l, dtype=int); vm_l[cc[cc < 32492]] = False
    vm_r = np.ones(32492, dtype=bool)
    if args.cc_r:
        cc = np.loadtxt(args.cc_r, dtype=int); vm_r[cc[cc < 32492]] = False
    print(f"CC mask: L valid={vm_l.sum()} R valid={vm_r.sum()}", flush=True)

    l_sids = {_norm_sid(s) for s in np.load(args.l_target_prefix + "_subject_ids.npy", allow_pickle=True)}
    r_sids = {_norm_sid(s) for s in np.load(args.r_target_prefix + "_subject_ids.npy", allow_pickle=True)}
    off_l = {_norm_sid(s) for s in np.load(args.lh_dir + "/subject_ids.npy", allow_pickle=True)}
    off_r = {_norm_sid(s) for s in np.load(args.rh_dir + "/subject_ids.npy", allow_pickle=True)}
    common = sorted(l_sids & r_sids & off_l & off_r)
    print(f"common={len(common)}", flush=True)

    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    if args.cv_folds <= 1:
        tr_s, va_s, te_s = cohort_split(common, args.splits, args.seed)
        print(f"cohort split seed={args.seed}: train={len(tr_s)} val={len(va_s)} test={len(te_s)} of {len(common)}", flush=True)
        full = PairedTfMRIDataset(args.lh_dir, args.rh_dir, args.l_target_prefix, args.r_target_prefix,
                                  common, ram_cache=True, dist=args.dist, exclude_subtasks=exclude_subs,
                                  nan=args.nan, nan_subjects=tr_s)
        ti = [full.sids.index(s) for s in tr_s]
        vi = [full.sids.index(s) for s in va_s]
        ei = [full.sids.index(s) for s in te_s]
        sbar_l = full.l_std[ti].mean(0).numpy(); sbar_r = full.r_std[ti].mean(0).numpy()
        tl = DataLoader(Subset(full, ti), shuffle=True, drop_last=True, **kw)
        vl = DataLoader(Subset(full, vi), shuffle=False, **kw)
        el = DataLoader(Subset(full, ei), shuffle=False, **kw)
        model = Cls(n_sub=full.n_sub, n_vertices=full.V, lh_ckpt=args.lh_ckpt,
                    rh_ckpt=args.rh_ckpt, load_pretrained=not args.random_encoder,
                    path_mode=args.path_mode).to(dev)
        if args.eval_only:
            model.load_state_dict(torch.load(args.save + '.pkl', map_location=dev))
            vm = evaluate(model, vl, dev, sbar_l, sbar_r, vm_l, vm_r)
            tm = evaluate(model, el, dev, sbar_l, sbar_r, vm_l, vm_r)
            print('=' * 60, flush=True)
            print('[single eval-only] VAL  L=%.4f R=%.4f rmse L=%.4f R=%.4f' %
                  (vm['l_mean'], vm['r_mean'], vm['l_rmse_mean'], vm['r_rmse_mean']), flush=True)
            print('[single eval-only] TEST L=%.4f R=%.4f rmse L=%.4f R=%.4f' %
                  (tm['l_mean'], tm['r_mean'], tm['l_rmse_mean'], tm['r_rmse_mean']), flush=True)
            with open(args.save + '.eval.json', 'w') as f:
                json.dump({'args': vars(args), 'val': vm, 'test': tm}, f, indent=2, default=float)
            print('done.', flush=True)
            return
        set_trainable(model)
        opt = AdamW(grouped_params(model, args.wd), lr=args.lr)
        sch = CosineAnnealingLR(opt, T_max=max(1, args.epochs))
        best = float("inf"); best_ckpt = args.save + '.pkl'; since_es = 0
        for ep in range(1, args.epochs + 1):
            model.train(); tl0 = 0.0
            model.l_enc.eval(); model.r_enc.eval()
            for b in tl:
                b = move(b, dev)
                pl, pr = model(b['lf'], b['lc'], b['lhop'], b['rf'], b['rc'], b['rhop'])
                loss = F.mse_loss(pl, b['l_tz']) + F.mse_loss(pr, b['r_tz'])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); tl0 += float(loss.item()) * b['l_tz'].shape[0]
            sch.step()
            vm = evaluate(model, vl, dev, sbar_l, sbar_r, vm_l, vm_r)
            model.eval()
            with torch.no_grad():
                vl_sum = 0.0; vl_n = 0
                for b in vl:
                    b = move(b, dev)
                    pl, pr = model(b['lf'], b['lc'], b['lhop'], b['rf'], b['rc'], b['rhop'])
                    vl_sum += float(F.mse_loss(pl, b['l_tz']) + F.mse_loss(pr, b['r_tz'])) * b['l_tz'].shape[0]
                    vl_n += b['l_tz'].shape[0]
                vl_loss = vl_sum / max(vl_n, 1)
            imp = vl_loss < best
            if imp:
                best = vl_loss; since_es = 0; torch.save(model.state_dict(), best_ckpt)
            else:
                since_es += 1
            print('  ep%03d train_mse=%.5f val_loss=%.5f val_L=%.4f val_R=%.4f %s' %
                  (ep, tl0/len(tl.dataset), vl_loss, vm['l_mean'], vm['r_mean'],
                   '***BEST' if imp else ''), flush=True)
            if since_es >= 5:
                print('  early stop @ ep%d (patience=5 val_loss)' % ep, flush=True)
                break
        model.load_state_dict(torch.load(best_ckpt, map_location=dev))
        vm = evaluate(model, vl, dev, sbar_l, sbar_r, vm_l, vm_r)
        tm = evaluate(model, el, dev, sbar_l, sbar_r, vm_l, vm_r)
        print('=' * 60, flush=True)
        print('[single] VAL  L_unmask=%.4f R_unmask=%.4f' % (vm['l_mean'], vm['r_mean']), flush=True)
        print('[single] TEST L_unmask=%.4f R_unmask=%.4f' % (tm['l_mean'], tm['r_mean']), flush=True)
        with open(args.save + '.result.json', 'w') as f:
            json.dump({'args': vars(args), 'val': vm, 'test': tm}, f, indent=2, default=float)
        print('done.', flush=True)
        return

    folds = build_subject_folds(common, args.cv_folds, args.seed)
    full = PairedTfMRIDataset(args.lh_dir, args.rh_dir, args.l_target_prefix, args.r_target_prefix,
                              common, ram_cache=True, dist=args.dist, exclude_subtasks=exclude_subs)
    Lm, Rm = [], []
    Lbr, Rbr = [], []
    for fi, (tr, va) in enumerate(folds):
        print(f"\n=== bilateral fold {fi+1}/{args.cv_folds} train={len(tr)} val={len(va)} ===", flush=True)
        if args.nan:
            full.refit_nan(tr)
        ti = [full.sids.index(s) for s in tr]; vi = [full.sids.index(s) for s in va]
        sbar_l = full.l_std[ti].mean(0).numpy(); sbar_r = full.r_std[ti].mean(0).numpy()
        tl = DataLoader(Subset(full, ti), shuffle=True, drop_last=True, **kw)
        vl = DataLoader(Subset(full, vi), shuffle=False, **kw)
        model = Cls(n_sub=full.n_sub, n_vertices=full.V, lh_ckpt=args.lh_ckpt,
                    rh_ckpt=args.rh_ckpt, load_pretrained=not args.random_encoder,
                    path_mode=args.path_mode).to(dev)

        if args.eval_only:
            model.load_state_dict(torch.load(f"{args.save}_fold{fi+1}.pkl", map_location=dev))
            print(f"  [eval-only] loaded {args.save}_fold{fi+1}.pkl", flush=True)
            m = evaluate(model, vl, dev, sbar_l, sbar_r, vm_l, vm_r)
            Lm.append(m["l_mean"]); Rm.append(m["r_mean"])
            Lbr.append(m["l_rmse_mean"]); Rbr.append(m["r_rmse_mean"])
            print(f"  fold{fi+1} [eval-only] mask: L={m['l_mean']:.4f} R={m['r_mean']:.4f} "
                  f"rmse L={m['l_rmse_mean']:.4f} R={m['r_rmse_mean']:.4f}", flush=True)
            torch.cuda.empty_cache()
            continue

        set_trainable(model)
        opt = AdamW(grouped_params(model, args.wd), lr=args.lr)
        sch = CosineAnnealingLR(opt, T_max=max(1, args.epochs))
        best = -9.0; best_ckpt = f"{args.save}_fold{fi+1}.pkl"
        for ep in range(1, args.epochs + 1):
            model.train(); tl0 = 0.0
            model.l_enc.eval(); model.r_enc.eval()
            for b in tl:
                b = move(b, dev)
                pl, pr = model(b["lf"], b["lc"], b["lhop"], b["rf"], b["rc"], b["rhop"])
                loss = F.mse_loss(pl, b["l_tz"]) + F.mse_loss(pr, b["r_tz"])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); tl0 += float(loss.item()) * b["l_tz"].shape[0]
            sch.step()
            m = evaluate(model, vl, dev, sbar_l, sbar_r, vm_l, vm_r)
            cur = (m["l_mean"] + m["r_mean"]) / 2
            if cur > best:
                best = cur; torch.save(model.state_dict(), best_ckpt)
            print(f"  ep{ep:03d} loss={tl0/len(tl.dataset):.4f} "
                  f"L_mask={m['l_mean']:.4f} R_mask={m['r_mean']:.4f} {'**' if cur==best else ''}", flush=True)
        model.load_state_dict(torch.load(best_ckpt, map_location=dev))
        m = evaluate(model, vl, dev, sbar_l, sbar_r, vm_l, vm_r)
        Lm.append(m["l_mean"]); Rm.append(m["r_mean"])
        Lbr.append(m["l_rmse_mean"]); Rbr.append(m["r_rmse_mean"])
        print(f"  fold{fi+1} best: L={m['l_mean']:.4f} R={m['r_mean']:.4f} "
              f"rmse L={m['l_rmse_mean']:.4f} R={m['r_rmse_mean']:.4f}", flush=True)
        torch.cuda.empty_cache()
    print("=" * 60, flush=True)
    print(f"[{args.cv_folds}-fold bilateral] masked PCC "
          f"L={np.mean(Lm):.4f} R={np.mean(Rm):.4f} avg={(np.mean(Lm)+np.mean(Rm))/2:.4f} | "
          f"rmse L={np.mean(Lbr):.4f} R={np.mean(Rbr):.4f}", flush=True)
    out_json = args.save + (".cv_eval.json" if args.eval_only else ".cv_result.json")
    with open(out_json, "w") as f:
        json.dump({"args": vars(args),
                   "L_masked_mean": float(np.mean(Lm)), "R_masked_mean": float(np.mean(Rm)),
                   "L_rmse_mean": float(np.mean(Lbr)), "R_rmse_mean": float(np.mean(Rbr))}, f, indent=2)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
