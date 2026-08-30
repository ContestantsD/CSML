"""compute_ig.py -- Integrated Gradients attribution for the bilateral phenotype model.

Method (as used for the attribution results reported in the paper):
  - Path: linear interpolation from an all-zero baseline, steps alpha = k/steps.
    Patch features, patch centers AND the geodesic hop matrix are jointly scaled
    by alpha: the baseline is an "empty surface", not zero features on the true
    geometry. Do not replace this loop with a features-only IG helper -- the two
    paths yield different attributions.
  - Attribution target: the final standardized scalar prediction.
  - Patch importance: sum of |IG| over (channels x within-patch positions).
  - Per subject, the bilateral (LH+RH) importance vector is normalized to unit
    sum; per-seed group maps are test-subject means; per-subject maps are saved.
  - Consistency check: a gradient-free forward pass is compared against the
    predictions.csv written at training time (Pearson r, max |diff|). A missing
    or malformed predictions.csv only disables the check, it never aborts IG.

Inputs:
  --lh-dir / --rh-dir : offline feature directories containing
        data.npy (N, C, P, S), centers.npy (N, P, S, 3), geodesic.npy (N, P, P),
        subject_ids.npy, canonical_mapping.npy (N, P)
  --cohort-dir : cohort.csv (Subject + phenotype columns, z-scored) and
        splits.csv (subject_id, seed, partition)
  --ckpt-template : directory template containing {seed} and {phenotype}; each
        directory must contain the checkpoint (net0_state_dict /
        net1_state_dict / fusion_mlp_state_dict) and, optionally, predictions.csv.

Outputs:
  <out-dir>/<phenotype>__seed<seed>/ig_seed<seed>_{LH,RH}.npy            (P,)
  <out-dir>/<phenotype>__seed<seed>/per_subject_ig_seed<seed>_{LH,RH}.npy (N, P)
  <out-dir>/<phenotype>__seed<seed>/report.json
  <out-dir>/ig_mean_{LH,RH}.npy, ig_seed_stability.csv, ig_top_patches.csv
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

_CT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CT)
_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _patched_load

from dataset_offline import compute_nan_stats   # noqa: E402
from meshmae_regressor import Mesh_regressor    # noqa: E402

TOP_FRAC = 0.10


def load_sids(path):
    return [str(int(x)) for x in np.load(path, allow_pickle=True)]


def get_nan_stats(seed, lh_dir, rh_dir, cohort_dir, cache_dir):
    """Training-split NAN statistics (per channel/patch), cached per seed."""
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, "nan_seed%d.npz" % seed)
    if os.path.exists(p):
        z = np.load(p)
        return {"LH": {"mu": z["LH_mu"], "sigma": z["LH_sigma"]},
                "RH": {"mu": z["RH_mu"], "sigma": z["RH_sigma"]}}
    sdf = pd.read_csv(os.path.join(cohort_dir, "splits.csv"), dtype={"subject_id": str})
    sdf["subject_id"] = sdf["subject_id"].str.zfill(6)
    train = sdf[(sdf.seed == seed) & (sdf.partition == "train")].subject_id.tolist()
    out = {}
    for name, d in (("LH", lh_dir), ("RH", rh_dir)):
        sids = load_sids(os.path.join(d, "subject_ids.npy"))
        pos = {s: i for i, s in enumerate(sids)}
        idx = [pos[s] for s in train]
        data = np.load(os.path.join(d, "data.npy"), mmap_mode="r")
        cmap = np.load(os.path.join(d, "canonical_mapping.npy"))
        mu, sigma = compute_nan_stats(data, cmap, idx)
        out[name] = {"mu": mu, "sigma": sigma}
    np.savez(p, LH_mu=out["LH"]["mu"], LH_sigma=out["LH"]["sigma"],
             RH_mu=out["RH"]["mu"], RH_sigma=out["RH"]["sigma"])
    return out


def build_model(ckpt_path, device):
    """Two frozen hemispheric encoders + fusion MLP (heads removed)."""
    kw = dict(channels=10, num_heads=6, encoder_depth=6, embed_dim=384,
              patch_size=64, drop_path=0.1, path_mode="dual")
    nets = [Mesh_regressor(**kw).to(device) for _ in range(2)]
    for net in nets:
        net.head = nn.Identity()
    fusion = nn.Sequential(nn.Linear(384 * 2, 128), nn.ReLU(),
                           nn.Linear(128, 1)).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    nets[0].load_state_dict(sd["net0_state_dict"])
    nets[1].load_state_dict(sd["net1_state_dict"])
    fusion.load_state_dict(sd["fusion_mlp_state_dict"])
    for m in nets + [fusion]:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return nets, fusion


def load_subject(feat_dir, sid, st):
    """Canonical re-ordering + NAN normalization. Returns (feats, centers, hop)."""
    sids = load_sids(os.path.join(feat_dir, "subject_ids.npy"))
    i = sids.index(sid)
    f = np.asarray(np.load(os.path.join(feat_dir, "data.npy"), mmap_mode="r")[i], dtype=np.float32)
    c = np.asarray(np.load(os.path.join(feat_dir, "centers.npy"), mmap_mode="r")[i], dtype=np.float32)
    hop = np.asarray(np.load(os.path.join(feat_dir, "geodesic.npy"), mmap_mode="r")[i], dtype=np.float32)
    inv = np.argsort(np.load(os.path.join(feat_dir, "canonical_mapping.npy"))[i])
    f, c, hop = f[:, inv, :], c[inv], hop[inv][:, inv]
    f = (f - st["mu"][:, :, None]) / st["sigma"][:, :, None]
    return (np.ascontiguousarray(f), np.ascontiguousarray(c), np.ascontiguousarray(hop))


def ig_batch(nets, fusion, bl, br, device, steps):
    """Manual IG. bl/br: lists of (feats, centers, hop) triples.

    feats, centers and hop are jointly scaled by alpha (empty-surface baseline).
    Returns per-subject IG tensors of shape (B, C, P, S) for both hemispheres.
    """
    z = torch.zeros_like
    stack = lambda side, k: torch.from_numpy(np.stack([t[k] for t in side])).to(device)  # noqa: E731
    F_l, C_l, H_l = stack(bl, 0), stack(bl, 1), stack(bl, 2)
    F_r, C_r, H_r = stack(br, 0), stack(br, 1), stack(br, 2)
    acc_l, acc_r = torch.zeros_like(F_l), torch.zeros_like(F_r)
    for k in range(1, steps + 1):
        a = k / steps
        x_l = (a * F_l).requires_grad_(True)
        x_r = (a * F_r).requires_grad_(True)
        o_l = nets[0](z(F_l), x_l, a * C_l, z(C_l), a * H_l)
        o_r = nets[1](z(F_r), x_r, a * C_r, z(C_r), a * H_r)
        out = fusion(torch.cat([o_l, o_r], dim=1)).reshape(-1)
        g_l, g_r = torch.autograd.grad(out, (x_l, x_r),
                                       grad_outputs=torch.ones_like(out))
        acc_l += g_l.detach()
        acc_r += g_r.detach()
    return (acc_l / steps) * F_l, (acc_r / steps) * F_r


def forward_plain(nets, fusion, bl, br, device):
    """Gradient-free forward pass -> standardized predictions (consistency check)."""
    outs = []
    for s in range(0, len(bl), 8):
        stack = lambda side, k: torch.from_numpy(  # noqa: E731
            np.stack([t[k] for t in side[s:s + 8]])).to(device)
        F_l, C_l, H_l = stack(bl, 0), stack(bl, 1), stack(bl, 2)
        F_r, C_r, H_r = stack(br, 0), stack(br, 1), stack(br, 2)
        z = torch.zeros_like
        o_l = nets[0](z(F_l), F_l, C_l, z(C_l), H_l)
        o_r = nets[1](z(F_r), F_r, C_r, z(C_r), H_r)
        outs.append(fusion(torch.cat([o_l, o_r], dim=1)).reshape(-1).cpu().numpy())
    return np.concatenate(outs)


def consistency_check(nets, fusion, bl, br, test, ckpt_dir, device):
    """Compare a fresh forward pass against the training-time predictions.csv."""
    try:
        pred = pd.read_csv(os.path.join(ckpt_dir, "predictions.csv"),
                           dtype={"subject_id": str})
        pred = pred[pred.partition == "test"].copy()
        pred["subject_id"] = pred["subject_id"].str.zfill(6)
        ref = dict(zip(pred.subject_id, pred.y_pred_z))
        ours = forward_plain(nets, fusion, bl, br, device)
        refv = np.array([ref[s] for s in test])
        return {"corr_with_saved": float(np.corrcoef(ours, refv)[0, 1]),
                "max_abs_diff": float(np.max(np.abs(ours - refv))),
                "pcc_ours_vs_ytrue": float(np.corrcoef(
                    ours, pred.set_index("subject_id").loc[test, "y_true_z"].values)[0, 1])}
    except Exception as e:                       # check is best-effort only
        return {"error": str(e)}


def run_seed(args, seed, device):
    phenotype = args.phenotype
    ckpt_dir = args.ckpt_template.format(seed=seed, phenotype=phenotype)
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        print("[SKIP seed %d] checkpoint not found: %s" % (seed, ckpt_path))
        return None
    out = Path(args.out_dir) / ("%s__seed%d" % (phenotype, seed))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    st = get_nan_stats(seed, args.lh_dir, args.rh_dir, args.cohort_dir,
                       os.path.join(args.out_dir, "_nan_cache"))
    sdf = pd.read_csv(os.path.join(args.cohort_dir, "splits.csv"), dtype={"subject_id": str})
    sdf["subject_id"] = sdf["subject_id"].str.zfill(6)
    test = sdf[(sdf.seed == seed) & (sdf.partition == "test")].subject_id.tolist()
    n_test = len(test)

    nets, fusion = build_model(ckpt_path, device)
    bl = [load_subject(args.lh_dir, s, st["LH"]) for s in test]
    br = [load_subject(args.rh_dir, s, st["RH"]) for s in test]
    print("[%s seed%d] model + %d subjects loaded in %.1fs" %
          (phenotype, seed, n_test, time.time() - t0), flush=True)

    check = consistency_check(nets, fusion, bl, br, test, ckpt_dir, device)
    print("  [check] %s" % check, flush=True)

    ti = time.time()
    P = bl[0][0].shape[1]
    ps_l = np.zeros((n_test, P), np.float32)
    ps_r = np.zeros((n_test, P), np.float32)
    for i in range(0, n_test, args.batch):
        ig_l, ig_r = ig_batch(nets, fusion, bl[i:i + args.batch],
                              br[i:i + args.batch], device, args.steps)
        pl = np.abs(ig_l.cpu().numpy()).sum(axis=(1, 3))
        pr = np.abs(ig_r.cpu().numpy()).sum(axis=(1, 3))
        for j in range(pl.shape[0]):
            tot = pl[j].sum() + pr[j].sum()
            ps_l[i + j] = pl[j] / max(tot, 1e-12)
            ps_r[i + j] = pr[j] / max(tot, 1e-12)
        print("  [ig] %d/%d (%.1fs)" % (min(i + args.batch, n_test), n_test,
                                        time.time() - ti), flush=True)

    mean_l, mean_r = ps_l.mean(0), ps_r.mean(0)
    np.save(out / ("ig_seed%d_LH.npy" % seed), mean_l)
    np.save(out / ("ig_seed%d_RH.npy" % seed), mean_r)
    np.save(out / ("per_subject_ig_seed%d_LH.npy" % seed), ps_l)
    np.save(out / ("per_subject_ig_seed%d_RH.npy" % seed), ps_r)

    report = {"phenotype": phenotype, "seed": seed, "steps": args.steps,
              "batch": args.batch, "n_test": n_test,
              "total_s": round(time.time() - t0, 1),
              "ig_s": round(time.time() - ti, 1),
              "sum_lh": float(mean_l.sum()), "sum_rh": float(mean_r.sum()),
              "per_subject_sum_min": float(min(ps_l.sum(1).min(), ps_r.sum(1).min())),
              "check": check}
    if torch.cuda.is_available():
        report["vram_peak_gib"] = round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 2)
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("[done] %s seed%d total %.1fs" % (phenotype, seed, report["total_s"]), flush=True)
    return mean_l, mean_r


def stability_analysis(seed_results, out_dir):
    """Pairwise cross-seed agreement per hemisphere (top-10% within each hemi)."""
    seeds = sorted(seed_results)
    rows = []
    for hemi in ("LH", "RH"):
        hi = 0 if hemi == "LH" else 1
        n_patches = len(seed_results[seeds[0]][hi])
        top_k = max(1, int(n_patches * TOP_FRAC))
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                va, vb = seed_results[seeds[i]][hi], seed_results[seeds[j]][hi]
                rho, p = spearmanr(va, vb)
                ta = set(np.argsort(va)[-top_k:])
                tb = set(np.argsort(vb)[-top_k:])
                rows.append({"hemi": hemi, "seed_a": seeds[i], "seed_b": seeds[j],
                             "spearman": round(float(rho), 4), "p": float(p),
                             "top10pct_jaccard": round(len(ta & tb) / len(ta | tb), 4)})
        tops = [set(np.argsort(seed_results[s][hi])[-top_k:]) for s in seeds]
        rows.append({"hemi": hemi, "seed_a": "ALL", "seed_b": "ALL",
                     "spearman": "", "p": "",
                     "top10pct_jaccard": "",
                     "common_top10pct": float(len(set.intersection(*tops)))})
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "ig_seed_stability.csv", index=False)
    print(df.to_string(index=False))
    return df


def save_top_patches(mean_l, mean_r, out_dir):
    rows = []
    for hemi, mean in (("LH", mean_l), ("RH", mean_r)):
        top_k = max(1, int(len(mean) * TOP_FRAC))
        for rank, p in enumerate(np.argsort(mean)[::-1][:top_k]):
            rows.append({"hemi": hemi, "patch": int(p), "rank": rank + 1,
                         "importance": float(mean[p])})
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "ig_top_patches.csv", index=False)
    print("saved ig_top_patches.csv (%d patches)" % len(df))


def main():
    ap = argparse.ArgumentParser(
        description="Integrated Gradients for the bilateral CSML phenotype model")
    ap.add_argument("--lh-dir", required=True,
                    help="left-hemisphere offline feature directory")
    ap.add_argument("--rh-dir", required=True,
                    help="right-hemisphere offline feature directory")
    ap.add_argument("--cohort-dir", required=True,
                    help="cohort directory (cohort.csv / splits.csv)")
    ap.add_argument("--phenotype", required=True,
                    help="phenotype column the checkpoints were trained on")
    ap.add_argument("--ckpt-template", required=True,
                    help="checkpoint dir template containing {seed} and {phenotype}")
    ap.add_argument("--ckpt-name", default="ckpt.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 11, 16])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device] %s" % device)
    seed_results = {}
    for seed in args.seeds:
        res = run_seed(args, seed, device)
        if res is not None:
            seed_results[seed] = res

    if len(seed_results) < 2:
        print("[WARN] fewer than 2 seeds completed, skipping stability analysis")
        return
    mean_l = np.mean([v[0] for v in seed_results.values()], axis=0)
    mean_r = np.mean([v[1] for v in seed_results.values()], axis=0)
    np.save(Path(args.out_dir) / "ig_mean_LH.npy", mean_l.astype(np.float32))
    np.save(Path(args.out_dir) / "ig_mean_RH.npy", mean_r.astype(np.float32))
    stability_analysis(seed_results, args.out_dir)
    save_top_patches(mean_l, mean_r, args.out_dir)
    print("done -> %s" % args.out_dir)


if __name__ == "__main__":
    main()
